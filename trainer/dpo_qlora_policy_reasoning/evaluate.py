"""Preference evaluation, generation metrics and reports."""


# Eval / metrics
import math
import re
from collections import Counter
from statistics import mean

from src.components.agentic_workflow.contracts import CandidateSelections, FactSelections
from trainer.dpo_qlora_policy_reasoning.data_prep import strict_json


def sanitize_output(text):
    # Evaluator accepts only registered synthetic cases, never private invoices.
    text = str(text)[:16384]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b(?:hf_|sk-|AIza)[A-Za-z0-9_-]+", "[TOKEN]", text)
    return re.sub(r"(?i)(?:bearer\s+|(?:api[_ -]?key|password|secret)\s*[:=]\s*)[^\s,}\"]+", "[REDACTED]", text)


def preference_score(chosen, rejected, tolerance=1e-6):
    if not all(math.isfinite(value) for value in (chosen, rejected)):
        raise ValueError("Nonfinite conditional log probability")
    margin = chosen - rejected
    return {"chosen_logp": chosen, "rejected_logp": rejected, "margin": margin,
            "outcome": "tied" if abs(margin) <= tolerance else "correct" if margin > 0 else "incorrect"}


def assess_generation(raw, record, annotation, *, truncated=False):
    """No repair and no postprocessed agent fallback. Expected labels were validated.

    Field/set comparisons follow the actual CandidateSelections/FactSelections
    contracts. Facts and conditions are supplied by canonical Python records,
    never inferred from a generated amount or invented citation.
    """
    role = annotation["agent_role"]
    schema = CandidateSelections if role == "agent_1" else FactSelections
    expected = schema.model_validate(strict_json(record["chosen"][0]["content"]))
    context = strict_json(record["prompt"][1]["content"])
    result = {"valid_json": False, "valid_schema": False, "correct_scoped_references": False,
              "eligible_completion": False, "semantic_correct": False, "truncated": truncated,
              "unsupported_reference_count": 0, "wrong_item_reference_count": 0,
              "unsupported_approval_count": 0, "missing_required_concern_count": 0,
              "unsupported_review_count": 0, "items": [], "failures": [],
              "correctness_status": "INCORRECT"}
    try:
        value = strict_json(raw)
        result["valid_json"] = True
    except (ValueError, TypeError):
        result["failures"].append("invalid_json")
        return result
    # Explicit citations/amounts are forbidden in these raw selection schemas.
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        for row in value["items"]:
            if isinstance(row, dict):
                citations = row.get("policy_ids", [])
                result["unsupported_reference_count"] += (len(citations) if isinstance(citations, list) else 1) if citations else 0
                if role == "agent_1" and row.get("suggested_status") == "COVERED":
                    index = row.get("item_index")
                    truth = next((r for r in expected.items if type(index) is int and r.item_index == index), None)
                    if truth is None or truth.suggested_status != "COVERED" or row.get("candidate_index") != truth.candidate_index or row.get("basis") != truth.basis:
                        result["unsupported_approval_count"] += 1
    try:
        output = schema.model_validate(value)
        result["valid_schema"] = True
    except ValueError:
        result["failures"].append("invalid_schema_or_forbidden_fields")
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            for row in value["items"]:
                if isinstance(row, dict) and row.get("candidate_index") is not None and type(row.get("candidate_index")) is not int:
                    result["unsupported_reference_count"] += 1
        return result
    if [r.item_index for r in output.items] != [r.item_index for r in expected.items]:
        result["failures"].append("row_identity_or_order")
        return result
    all_facts = {f["index"] for row in context["items"] for f in row.get("facts", [])}
    all_concerns = {f["index"] for row in context["items"] for f in row.get("concern_facts", [])}
    references_ok = True
    for selected, truth, row in zip(output.items, expected.items, context["items"]):
        details = {"item_index": truth.item_index}
        if role == "agent_1":
            allowed = {c["index"] for c in row["candidates"]}
            scope = selected.candidate_index is None or selected.candidate_index in allowed
            if not scope:
                result["unsupported_reference_count"] += 1
            details.update(classification_correct=selected.suggested_status == truth.suggested_status,
                candidate_correct=selected.candidate_index == truth.candidate_index,
                basis_correct=selected.basis == truth.basis,
                expected_status=truth.suggested_status,
                scoped_reference_correct=scope)
            details["correct"] = all(details[k] for k in ("classification_correct", "candidate_correct", "basis_correct")) and scope
        else:
            facts = {f["index"] for f in row["facts"]}
            concerns = {f["index"] for f in row["concern_facts"]}
            wrong_facts = set(selected.fact_indices)-facts
            wrong_concerns = set(selected.concerns)-concerns
            scope = not wrong_facts and not wrong_concerns
            result["unsupported_reference_count"] += len(wrong_facts)+len(wrong_concerns)
            result["wrong_item_reference_count"] += len(wrong_facts & all_facts)+len(wrong_concerns & all_concerns)
            result["missing_required_concern_count"] += len(set(truth.concerns)-set(selected.concerns))
            facts_correct = len(selected.fact_indices) == len(set(selected.fact_indices)) and set(selected.fact_indices) == set(truth.fact_indices)
            concerns_correct = len(selected.concerns) == len(set(selected.concerns)) and set(selected.concerns) == set(truth.concerns)
            details.update(facts_correct=facts_correct, concerns_correct=concerns_correct,
                status_correct=selected.status == truth.status, expected_status=truth.status,
                scoped_reference_correct=scope)
            details["correct"] = facts_correct and concerns_correct and details["status_correct"] and scope
            result["unsupported_review_count"] += int(not details["correct"])
        references_ok &= scope
        result["items"].append(details)
        if not details["correct"]:
            result["failures"].append(f"unsupported_selection:item_{truth.item_index}")
    result["correct_scoped_references"] = references_ok
    result["eligible_completion"] = references_ok and not truncated
    result["semantic_correct"] = all(row["correct"] for row in result["items"]) and not truncated
    if truncated:
        result["failures"].append("generation_did_not_reach_eos")
    result["correctness_status"] = "CORRECT" if result["semantic_correct"] else "INCORRECT"
    return result


def rate(successes, total):
    return {"successes": successes, "denominator": total,
            "percent": 100*successes/total if total else None}


def summarize(predictions, scores):
    """Missing/invalid answers remain in denominators; no invented zero-sample rate."""
    result = {"samples": len(predictions)}
    for field in ("valid_json", "valid_schema", "correct_scoped_references", "eligible_completion", "semantic_correct"):
        result[field] = rate(sum(p["assessment"][field] for p in predictions), len(predictions))
    for field in ("unsupported_reference_count", "wrong_item_reference_count", "unsupported_approval_count",
                  "missing_required_concern_count", "unsupported_review_count"):
        result[field] = sum(p["assessment"][field] for p in predictions)
    for key in ("classification_correct", "candidate_correct", "basis_correct", "facts_correct", "concerns_correct", "status_correct"):
        eligible = [(p, e) for p in predictions for e in p["expected_items"]
                    if (p["agent_role"] == "agent_1") == (key in {"classification_correct", "candidate_correct", "basis_correct"})]
        successes = sum(next((row.get(key, False) for row in p["assessment"]["items"] if row["item_index"] == e["item_index"]), False) for p, e in eligible)
        result[key] = rate(successes, len(eligible))
    for name, status in (("coverage_selection", "COVERED"), ("exclusion_identification", "EXCLUDED"), ("appropriate_review", "REVIEW_REQUIRED")):
        eligible = [(p, e) for p in predictions if p["agent_role"] == "agent_1" for e in p["expected_items"] if e["suggested_status"] == status]
        successes = sum(next((r["correct"] for r in p["assessment"]["items"] if r["item_index"] == e["item_index"]), False) for p, e in eligible)
        result[name] = rate(successes, len(eligible))
    agent1 = [p for p in predictions if p["agent_role"] == "agent_1"]
    result["unsupported_approval_rate"] = rate(sum(p["assessment"]["unsupported_approval_count"] > 0 for p in agent1), len(agent1))
    result["wrong_item_evidence_rate"] = rate(sum(p["assessment"]["wrong_item_reference_count"] > 0 for p in predictions if p["agent_role"] == "agent_2"), sum(p["agent_role"] == "agent_2" for p in predictions))
    outcomes = Counter(s["outcome"] for s in scores)
    result["preference"] = {"correct": outcomes["correct"], "incorrect": outcomes["incorrect"],
        "tied": outcomes["tied"], "accuracy": rate(outcomes["correct"], len(scores)),
        "mean_margin": mean(s["margin"] for s in scores) if scores else None,
        "failures": len(predictions)-len(scores), "definition": "raw summed assistant-completion logp; ties are not correct; NOT DPO reward accuracy"}
    result["generation_failures"] = sum(not p["assessment"]["semantic_correct"] for p in predictions)
    result["average_generation_seconds"] = mean(p["generation_seconds"] for p in predictions) if predictions else None
    return result


def aggregate(predictions, pairwise):
    metrics = {}
    for model in ("baseline", "dpo"):
        rows = predictions[model]
        scores = [s[model] for s in pairwise]
        metrics[model] = {"overall": summarize(rows, scores), "by_role": {}, "by_scenario": {}}
        for group, field in (("by_role", "agent_role"), ("by_scenario", "scenario_family")):
            for value in sorted({r[field] for r in rows}):
                subset = [r for r in rows if r[field] == value]
                ids = {r["pair_id"] for r in subset}
                metrics[model][group][value] = summarize(subset, [s[model] for s in pairwise if s["pair_id"] in ids])
    return metrics


# Eval / report
import csv
import json


def comparison_rows(metrics):
    directions = {"unsupported_reference_count": False, "wrong_item_reference_count": False,
                  "unsupported_approval_count": False, "missing_required_concern_count": False,
                  "unsupported_review_count": False, "unsupported_approval_rate": False,
                  "wrong_item_evidence_rate": False, "generation_failures": False,
                  "average_generation_seconds": False}
    rows = []
    for group in ("overall", "agent_1", "agent_2"):
        left = metrics["baseline"]["overall"] if group == "overall" else metrics["baseline"]["by_role"].get(group, {})
        right = metrics["dpo"]["overall"] if group == "overall" else metrics["dpo"]["by_role"].get(group, {})
        for name in left:
            if name in {"samples", "preference"}:
                continue
            a, b = left[name], right[name]
            if isinstance(a, dict):
                a, b = a["percent"], b["percent"]
            direction = directions.get(name, True)
            delta = b-a if a is not None and b is not None else None
            verdict = "not measured" if delta is None else "tied" if abs(delta) <= 1e-9 else (
                "improved" if (delta > 0) == direction else "worsened")
            rows.append(dict(group=group, metric=name, baseline=a, dpo=b, delta=delta, dpo_result=verdict))
        if left:
            for name in ("accuracy", "mean_margin"):
                a, b = left["preference"][name], right["preference"][name]
                if isinstance(a, dict):
                    a, b = a["percent"], b["percent"]
                delta = b-a if a is not None and b is not None else None
                verdict = "not measured" if delta is None else "tied" if abs(delta) <= 1e-9 else "improved" if delta > 0 else "worsened"
                rows.append(dict(group=group, metric="raw_preference_"+name, baseline=a, dpo=b, delta=delta, dpo_result=verdict))
    return rows


def write_reports(directory, predictions, scores, metrics, metadata):
    ids = [p["pair_id"] for p in predictions["baseline"]]
    if (metadata.get("status") != "completed" or len(ids) != metadata["dataset"]["pairs"] or
            len(ids) != len(set(ids)) or ids != [p["pair_id"] for p in predictions["dpo"]] or
            ids != [s["pair_id"] for s in scores]):
        raise ValueError("Refuse fabricated/incomplete comparison report")
    def write_json(name, value):
        (directory/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    for model in ("baseline", "dpo"):
        (directory/(model+"_predictions.jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in predictions[model]), encoding="utf-8")
    write_json("pairwise_scores.json", scores)
    write_json("metrics.json", metrics)
    rows = comparison_rows(metrics)
    with (directory/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "metric", "baseline", "dpo", "delta", "dpo_result"])
        writer.writeheader()
        writer.writerows(rows)
    reviews = []
    for baseline, dpo in zip(predictions["baseline"], predictions["dpo"]):
        reviews.append({"pair_id": baseline["pair_id"], "agent_role": baseline["agent_role"],
            "prompt_summary": baseline["scenario_family"], "canonical_policy_ids": baseline["source_policy_ids"],
            "verified_expected": baseline["expected_items"],
            "baseline_response": baseline["raw_output_sanitized"], "dpo_response": dpo["raw_output_sanitized"],
            "baseline_status": baseline["assessment"]["correctness_status"],
            "dpo_status": dpo["assessment"]["correctness_status"],
            "baseline_failures": baseline["assessment"]["failures"], "dpo_failures": dpo["assessment"]["failures"],
            "human_review_status": "PENDING"})
    (directory/"human_review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in reviews), encoding="utf-8")
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"
    cases = metadata["dataset"].get("underlying_cases", len({p["scenario_family"] for p in predictions["baseline"]}))
    role_counts = {role: sum(p["agent_role"] == role for p in predictions["baseline"]) for role in ("agent_1", "agent_2")}
    lines = ["# Independent baseline vs DPO evaluation", "",
        f"Both modes: `{metadata['model_id']}`. DPO adapter SHA256: `{metadata['adapter']['adapter_sha256']}`.",
        f"Checkpoint: `{metadata['adapter']['adapter']}`.",
        f"Split: **test**, {len(ids)} pairs ({role_counts}), from {cases} underlying synthetic cases. Test SHA256: `{metadata['dataset']['test_sha256']}`.",
        "Identical base object, tokenizer, chat template, prompts and greedy settings. Adapter state is audited for every mode.", "",
        "## Generation and raw preference ranking", "",
        "Percentages are percentages of explicit denominators in metrics.json; counts are counts. Missing denominators are N/A.", "",
        "eligible_completion means a nontruncated schema-valid answer with row-owned references, not approval or semantic correctness. Safety rates count affected prompts; reference/approval counts count offending selections.", "",
        "| Group | Metric | Baseline | DPO | Change | DPO result |",
        "|---|---|---:|---:|---:|---|"]
    lines += [f"| {r['group']} | {r['metric']} | {number(r['baseline'])} | {number(r['dpo'])} | {number(r['delta'])} | {r['dpo_result']} |" for r in rows]
    lines += ["", "## Preference outcome counts", ""]
    for mode in ("baseline", "dpo"):
        p = metrics[mode]["overall"]["preference"]
        lines.append(f"- {mode}: correct={p['correct']}, incorrect={p['incorrect']}, tied={p['tied']}, scoring failures={p['failures']}.")
    lines += ["", "Raw preference accuracy ranks summed chosen/rejected completion log probabilities; it is neither DPO reference-relative reward accuracy nor claim accuracy.",
        "Per-scenario results and their small denominators are in metrics.json.", "", "## Representative actual generation failures", ""]
    failures = [(mode, p) for mode in ("baseline", "dpo") for p in predictions[mode] if p["assessment"]["failures"]]
    lines += [f"- {mode}, {p['pair_id']}: {', '.join(p['assessment']['failures'])}. See sanitized output in predictions JSONL." for mode, p in failures[:6]] or ["No automatic generation-contract failures observed in this run."]
    lines += ["", "## Runtime", "",
        f"Total evaluation time: {metadata.get('total_seconds', 0):.2f}s. Peak tracked CUDA allocation: {metadata.get('peak_cuda_allocated_gib', 0):.2f} GiB.",
        "Per-generation latency and pairwise scoring time are recorded separately.", "", "## Limitations and conclusion", "",
        f"Only {len(ids)} synthetic preference pairs ({cases} correlated case groups) were evaluated. These labels are automatically canonical-validated, not independently human verified.",
        "Negatives are deliberately constructed single-field/scoped-reference errors, often easy; they do not represent the full difficulty of clinical reasoning or real claim review.",
        "Both modes use the same NF4 quantized base; this comparison does not measure full-precision baseline behavior.",
        "Agent 2 receives the same fixed verified decisions/Agent 1 context in both modes; this is independent reasoning evaluation, not a new end-to-end deployment.",
        "No free-text explanation quality is claimed because raw agent contracts contain selections, not prose. Semantics are scored only where canonical annotation replay establishes the expected decision.",
        "Training did not record a base-weight revision/hash; evaluation records its resolved base revision/config hash. Confirm the local base snapshot matches the training snapshot before drawing conclusions.",
        "Metric changes above report improved/worsened/tied independently. No overall superiority or deployment decision is inferred. Do not tune prompts, checkpoints or parameters from these test results."]
    (directory/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json("metadata.json", metadata)  # last file; parent publishes directory only after completion


# Eval / evaluate
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

from src.utils.policy import load_policy
from trainer.dpo_qlora_policy_reasoning.provenance import verify_provenance
from trainer.dpo_qlora_policy_reasoning.data_prep import strict_json, validate_records
from trainer.dpo_qlora_policy_reasoning.train import MODEL_ID, OUTPUT_ROOT, PINNED
from trainer.dpo_qlora_policy_reasoning.data_prep import serialize_and_audit
from trainer.dpo_qlora_policy_reasoning.train import verify_latest
from src.utils.paths import ROOT, project_path

CONTEXT_LENGTH = 2048


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_weight_hashes(revision):
    # Resolve only already-loaded cached files; never download while hashing.
    from transformers.utils.hub import cached_file
    options = dict(revision=revision, local_files_only=True, _raise_exceptions_for_missing_entries=False)
    index = cached_file(MODEL_ID, "model.safetensors.index.json", **options)
    names = sorted(set(strict_json(Path(index).read_text())["weight_map"].values())) if index else ["model.safetensors"]
    result = {}
    for name in names:
        path = cached_file(MODEL_ID, name, **options)
        if not path:
            raise RuntimeError("Cannot fingerprint the cached base checkpoint")
        result[name] = sha(path)
    return result


def load_test(directory="data/policy_reasoning_dpo_qlora"):
    """Never open train/validation. Reuse canonical validation for test only."""
    directory = project_path(directory).resolve()
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    provenance = verify_provenance(manifest)
    if manifest["pair_count"] != 80 or manifest["flagged_cases"] or manifest["human_verified"] is not False:
        raise ValueError("Expected complete, automatically validated 80-pair synthetic dataset")
    if manifest["base_model_id"] != "Qwen/Qwen2.5-VL-3B-Instruct":
        raise ValueError("Unexpected Phase 9A annotation provenance; evaluation backbone remains the TEXT model")
    families, cases = {}, {}
    for annotation in manifest["annotations"]:
        for mapping, key in ((families, annotation["scenario_family"]), (cases, annotation["synthetic_case_id"])):
            if key in mapping and mapping[key] != annotation["data_split"]:
                raise ValueError("Cross-split case/family leakage in manifest")
            mapping[key] = annotation["data_split"]
    annotations = [a for a in manifest["annotations"] if a["data_split"] == "test"]
    if any(a["generation_seed"] != manifest["generation_seed"] or a["validation_status"] != "automatically_validated_human_review_pending" for a in annotations):
        raise ValueError("Test annotation provenance/status mismatch")
    path = directory/"test.jsonl"
    records = [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 10 or len(annotations) != 10:
        raise ValueError("Expected exactly 10 held-out test pairs")
    entries = list(zip(records, annotations))
    counts = validate_records(entries, load_policy(ROOT/"data/assignment_dataset/policy.json"))
    if counts["test"] != {"agent_1": 5, "agent_2": 5}:
        raise ValueError("Expected 5 test pairs per agent")
    return entries, {"split": "test", "pairs": 10, "by_role": counts["test"],
        "underlying_cases": len({a["synthetic_case_id"] for a in annotations}),
        "test_sha256": sha(path), "manifest_sha256": sha(directory/"manifest.json"),
        "policy_sha256": manifest["policy_sha256"], "source_sha256": manifest["source_sha256"],
        "source_provenance": provenance,
        "human_verified": False, "train_validation_opened": False,
        "annotation_source_model": manifest["base_model_id"],
        "rejected_error_types": sorted({a["rejected_error_type"] for a in annotations})}


def output_root(value):
    root = project_path(value).resolve()
    if not root.is_relative_to((ROOT/"reports/dpo").resolve()):
        raise ValueError("Evaluation output must be inside reports/dpo; data/model directories are protected")
    return root


def preflight(args):
    if args.split != "test" or not 1 <= args.max_new_tokens < CONTEXT_LENGTH or args.seed < 0:
        raise ValueError("Only test is supported; invalid generation token budget/seed")
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != (ROOT.parent/".venv").resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the original ../.venv Python 3.12 training/application environment")
    versions = {name: metadata.version(name) for name in (*PINNED, "torch")}
    for name, version in PINNED.items():
        if versions[name] != version:
            raise RuntimeError(f"Expected inspected {name}=={version}; found {versions[name]}; no automatic upgrades")
    entries, audit = load_test(args.dataset_dir)
    adapter = verify_latest(OUTPUT_ROOT)  # existing checksum, identity and tensor verifier; CPU only
    path = Path(adapter["adapter"])
    config = strict_json((path.parent/"training_config.json").read_text())
    if config.get("model_id") != MODEL_ID or config.get("model") != MODEL_ID:
        raise ValueError("Adapter was not trained from the required TEXT backbone")
    training_audit = strict_json((path.parent/"dataset_manifest.json").read_text())
    if training_audit.get("test_accessed") is not False or training_audit.get("manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("Training provenance does not match this untouched test manifest")
    return entries, {"model_id": MODEL_ID, "adapter": adapter,
        "adapter_config_sha256": sha(path/"adapter_config.json"),
        "training_config_sha256": sha(path.parent/"training_config.json"),
        "dataset": audit, "versions": versions, "interpreter": sys.executable,
        "output_root": str(output_root(args.output_dir)),
        "chat_template": "checked against saved adapter tokenizer at GPU-evaluation startup",
        "gpu_kernels_and_inference": "not verified by CPU preflight"}


def completion_batch(prompt, completion):
    if not prompt or not completion:
        raise ValueError("Prompt and assistant completion must be nonempty")
    return {"input_ids": list(prompt)+list(completion), "labels": [-100]*len(prompt)+list(completion),
            "prediction_positions": list(range(len(prompt)-1, len(prompt)+len(completion)-1))}


def adapter_layers(model):
    layers = [module for module in model.modules() if hasattr(module, "lora_A")]
    if not layers or set(model.peft_config) != {"dpo"}:
        raise RuntimeError("Expected exactly the verified DPO adapter; no extraction or other adapter")
    if any(layer.merged for layer in layers):
        raise RuntimeError("Merged adapters cannot provide an unbiased disabled-adapter baseline")
    return layers


@contextmanager
def model_mode(model, mode):
    if mode not in {"baseline", "dpo"}:
        raise ValueError("Unknown evaluation model mode")
    model.set_adapter("dpo")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = adapter_layers(model)
    if mode == "baseline":
        with model.disable_adapter():
            if any(not layer.disable_adapters for layer in layers):
                raise RuntimeError("Baseline still has an enabled LoRA adapter")
            yield {"mode": mode, "adapter_enabled": False, "verified_lora_layers": len(layers)}
    else:
        if any(layer.disable_adapters or layer.active_adapters != ["dpo"] or layer.scaling.get("dpo", 0) <= 0 for layer in layers):
            raise RuntimeError("DPO adapter is not active on every LoRA layer")
        yield {"mode": mode, "adapter_enabled": True, "verified_lora_layers": len(layers)}


def generation_parameters(tokenizer, max_new_tokens):
    # No temperature/top_p/top_k sampling arguments: identical greedy decoding.
    return dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                use_cache=True)


class Backend:
    def __init__(self, model, tokenizer, max_new_tokens):
        self.model, self.tokenizer = model, tokenizer
        self.generation = generation_parameters(tokenizer, max_new_tokens)

    def logp(self, prompt, completion):
        import torch
        batch = completion_batch(prompt, completion)
        ids = torch.tensor([batch["input_ids"]], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            # Last C+1 logits contain the C completion prediction positions.
            # The final next-token logit is dropped; prompt logits never scored.
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, logits_to_keep=len(completion)+1).logits[0, :-1].float()
            labels = torch.tensor(batch["labels"][len(prompt):], dtype=torch.long, device=ids.device)
            if logits.shape[0] != labels.shape[0]:
                raise RuntimeError("Qwen completion logit/label alignment mismatch")
            value = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).sum().item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite assistant log probability")
        return value

    def generate(self, prompt):
        import torch
        if len(prompt)+self.generation["max_new_tokens"] > CONTEXT_LENGTH:
            raise ValueError("Generation budget exceeds 2048; refuse prompt truncation")
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **self.generation)
        suffix = output[0, len(prompt):].tolist()
        finished = bool(suffix and suffix[-1] == self.tokenizer.eos_token_id)
        return self.tokenizer.decode(suffix, skip_special_tokens=True), not finished


def evaluate_entries(entries, prepared, backend, mode_context):
    """Injectable CPU test seam; live path always uses actual HF forwards."""
    if len(prepared) != len(entries):
        raise ValueError("Incomplete prepared evaluation cases")
    predictions = {"baseline": [], "dpo": []}
    pairwise = []
    for (record, annotation), tokens in zip(entries, prepared):
        scores = {key: annotation[key] for key in ("pair_id", "agent_role", "scenario_family")}
        scores["token_counts"] = {key: len(tokens[key+"_input_ids"]) for key in ("prompt", "chosen", "rejected")}
        for mode in ("baseline", "dpo"):
            with mode_context(mode) as adapter_state:
                scoring_start = time.monotonic()
                chosen = backend.logp(tokens["prompt_input_ids"], tokens["chosen_input_ids"])
                rejected = backend.logp(tokens["prompt_input_ids"], tokens["rejected_input_ids"])
                scores[mode] = {**preference_score(chosen, rejected), "scoring_seconds": time.monotonic()-scoring_start}
                generation_start = time.monotonic()
                # Only prompt token IDs reach generation; no completions/labels.
                raw, truncated = backend.generate(tokens["prompt_input_ids"])
                latency = time.monotonic()-generation_start
            assessment = assess_generation(raw, record, annotation, truncated=truncated)
            predictions[mode].append({**{key: annotation[key] for key in ("pair_id", "agent_role", "scenario_family")},
                "scenario": annotation["scenario"], "source_policy_ids": annotation["source_policy_ids"],
                "expected_items": strict_json(record["chosen"][0]["content"])["items"],
                "raw_output_sanitized": sanitize_output(raw), "assessment": assessment,
                "generation_seconds": latency, "adapter_state": adapter_state})
        pairwise.append(scores)
        print(f"Evaluated {len(pairwise)}/{len(entries)} held-out pairs (both modes)", flush=True)
    if len(prepared) != len(entries) or len(pairwise) != len(entries):
        raise ValueError("Incomplete prepared evaluation cases")
    return predictions, pairwise


def run(args):
    entries, details = preflight(args)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required; use CUDA_VISIBLE_DEVICES=0")
    if torch.cuda.mem_get_info()[0] < 18*2**30:
        raise RuntimeError("Stop vLLM/other GPU work manually; require at least 18 GiB free")
    set_seed(args.seed)
    started = time.monotonic()
    adapter_path = Path(details["adapter"]["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=args.local_files_only)
    saved_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=False, local_files_only=True)
    if (tokenizer.get_vocab() != saved_tokenizer.get_vocab() or tokenizer.get_chat_template() != saved_tokenizer.get_chat_template()
            or tokenizer.special_tokens_map != saved_tokenizer.special_tokens_map):
        raise RuntimeError("Base and saved training tokenizer/template differ; cannot fairly evaluate")
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    saved_config = strict_json((adapter_path.parent/"training_config.json").read_text())
    if template_hash != saved_config.get("chat_template_sha256"):
        raise RuntimeError("Text chat template differs from successful DPO training")
    prepared = [serialize_and_audit(record, tokenizer, CONTEXT_LENGTH)[1] for record, _ in entries]
    if any(len(row["prompt_input_ids"])+args.max_new_tokens > CONTEXT_LENGTH for row in prepared):
        raise ValueError("Prompt plus generation budget exceeds 2048; choose a smaller max-new-tokens BEFORE final evaluation")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Loading one NF4 text backbone and the verified reasoning adapter", flush=True)
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=args.local_files_only,
        trust_remote_code=False, device_map={"": 0}, dtype=dtype, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    if base.config.model_type != "qwen2":
        raise RuntimeError("Expected TEXT Qwen2ForCausalLM, never Qwen2.5-VL")
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name="dpo", is_trainable=False,
        autocast_adapter_dtype=True)
    model.eval()
    backend = Backend(model, tokenizer, args.max_new_tokens)
    base_hashes = base_weight_hashes(getattr(base.config, "_commit_hash", None))
    if sha(adapter_path/"adapter_model.safetensors") != details["adapter"]["adapter_sha256"]:
        raise RuntimeError("Adapter changed after checkpoint verification")
    root = output_root(args.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8]
    staging = root/(run_id+".pending")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        predictions, scores = evaluate_entries(entries, prepared, backend, lambda mode: model_mode(model, mode))
        details.update(status="completed", run_id=run_id, seed=args.seed, context_length=CONTEXT_LENGTH,
            generation_settings=backend.generation, chat_template_sha256=template_hash,
            base_revision=getattr(base.config, "_commit_hash", None), base_config_sha256=hashlib.sha256(base.config.to_json_string().encode()).hexdigest(),
            base_weight_sha256=base_hashes, torch_runtime_version=torch.__version__,
            evaluator_source_sha256={name: sha(Path(__file__).parent/name) for name in ("evaluate.py",)},
            quantization="4-bit NF4, double quantization; same base object and tokenizer for both modes",
            dtype=str(dtype), gpu_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            scoring="summed completion-only conditional logp, including stock TRL completion EOS; prompt/padding excluded",
            pairwise_tie_tolerance=1e-6, total_seconds=time.monotonic()-started,
            metric_definitions={"eligible_completion": "nontruncated, valid-schema answer with row-owned references; not approval or semantic correctness",
                "unsupported_approval_rate": "percent of Agent 1 prompts containing any unsupported approval",
                "wrong_item_evidence_rate": "percent of Agent 2 prompts containing any cross-item reference"},
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            created_utc=datetime.now(timezone.utc).isoformat())
        write_reports(staging, predictions, scores, aggregate(predictions, scores), details)
        final = root/run_id
        staging.rename(final)
        print(f"Completed evaluation reports: {final}", flush=True)
    except BaseException as exc:
        (staging/"failure.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__,
            "message": sanitize_output(str(exc)), "run_id": run_id}, indent=2)+"\n", encoding="utf-8")
        print(f"Evaluation failed; no completed report published. Diagnostic directory: {staging}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--dataset-dir", default="data/policy_reasoning_dpo_qlora")
    parser.add_argument("--output-dir", default="reports/dpo/evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="CPU data/environment/checkpoint verification; no model load/download")
    args = parser.parse_args()
    if args.check:
        _, details = preflight(args)
        print(json.dumps(details, indent=2), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
