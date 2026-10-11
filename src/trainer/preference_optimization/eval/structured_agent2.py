"""Manual, schema-constrained Agent 2 diagnostic evaluation; no deployment changes."""
import argparse
import copy
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shlex
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

from src.adjudication.policy import load_policy
from src.agents.contracts import AgentSafetyError, FactSelections
from src.agents.reasoning_reviewer import ReasoningReviewAgent
from src.inference.remote_client import RemoteVLMClient, RemoteVLMConfig, RemoteVLMError
from src.trainer.preference_optimization.annotation.cases import (
    AnnotationClient, agent1, calculate, prepare_case, rejected_selection, source_hashes,
)
from src.trainer.preference_optimization.annotation.validate import strict_json, validate_records
from src.trainer.preference_optimization.train.config import MODEL_ID, OUTPUT_ROOT
from src.trainer.preference_optimization.train.train_dpo import verify_latest
from src.utils.paths import ROOT, project_path
from .evaluate import CONTEXT_LENGTH, output_root, sha
from .metrics import assess_generation, sanitize_output, summarize

BASE_NAME = "reasoning-base"
DPO_NAME = "reasoning-dpo"


def load_cases(split="validation", limit=None, acknowledge=False):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unsupported diagnostic split")
    if split == "test" and not acknowledge:
        raise ValueError("Observed test requires --acknowledge-observed-test; it is a diagnostic retest, not independent evaluation")
    directory = ROOT/"data/dpo"
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hashes() or manifest["policy_sha256"] != sha(ROOT/"data/policy.json"):
        raise ValueError("Canonical policy/contracts changed; stop rather than alter held-out labels")
    annotations = [a for a in manifest["annotations"] if a["data_split"] == split]
    records = [strict_json(line) for line in (directory/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != len(annotations) or len(records) != {"train": 60, "validation": 10, "test": 10}[split]:
        raise ValueError("Unexpected preference split count")
    entries = list(zip(records, annotations))
    validate_records(entries, load_policy(ROOT/"data/policy.json"))
    agent2 = [(record, annotation) for record, annotation in entries if annotation["agent_role"] == "agent_2"]
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        agent2 = agent2[:limit]
    return agent2, {"split": split, "pairs": len(agent2), "split_sha256": sha(directory/f"{split}.jsonl"),
        "manifest_sha256": sha(directory/"manifest.json"), "policy_sha256": manifest["policy_sha256"],
        "evaluation_kind": "observed_test_diagnostic_retest" if split == "test" else "non_test_implementation_diagnostic",
        "independent_generalization_evidence": False, "human_verified": False}


def response_format():
    # Generic wire schema only: NO gold indices, statuses, concerns, row counts,
    # enum narrowing or default selections. Semantic correctness remains open.
    return {"type": "json_schema", "json_schema": {
        "name": "FactSelections", "strict": True, "schema": FactSelections.model_json_schema()}}


def replay_review(raw, annotation):
    """Feed the ACTUAL generation to the original runtime validator offline.

    AnnotationClient here is only a response carrier for canonical validation;
    no response from it is ever used as a model generation or replacement answer.
    """
    output = FactSelections.model_validate(strict_json(raw))
    policy = load_policy(ROOT/"data/policy.json")
    spec = annotation["case"]
    invoice, evidence, resolutions = prepare_case(spec, policy)
    first, suggestions, meta = agent1(invoice, evidence, resolutions, policy)
    bad, _ = rejected_selection("agent_1", first.output.model_dump(), first.data)
    _, _, bad_meta = agent1(invoice, evidence, resolutions, policy, bad)
    claim = calculate(invoice, resolutions, suggestions, meta, policy)
    before = claim.model_dump(mode="json")
    reviewed, _ = ReasoningReviewAgent(AnnotationClient(output.model_dump())).review(
        invoice, evidence, suggestions, claim, deadline=float("inf"), policy=policy,
        agent_metadata=bad_meta if spec["prior_disagreement"] else meta)
    if claim.model_dump(mode="json") != before or before != annotation["deterministic_claim"]:
        raise AgentSafetyError("Canonical financial result changed during review validation")
    return reviewed.model_dump(mode="json")


def selection_diagnostics(raw, record):
    """Inspect selections even when forbidden fields cause schema rejection."""
    try:
        value = strict_json(raw)
        rows = value.get("items", [])
        context = strict_json(record["prompt"][1]["content"])
        if not isinstance(rows, list):
            return {"field": "items", "reason": "not_a_list"}
        result = []
        def safe_indices(value):
            if not isinstance(value, list):
                return None
            return [entry if type(entry) is int else "INVALID_TYPE" for entry in value[:50]]
        for expected in context["items"]:
            selected = next((r for r in rows if isinstance(r, dict) and type(r.get("item_index")) is int and r["item_index"] == expected["item_index"]), {})
            result.append({"item_index": expected["item_index"],
                "allowed_fact_indices": [f["index"] for f in expected["facts"]],
                "selected_fact_indices": safe_indices(selected.get("fact_indices")),
                "allowed_concern_indices": [c["index"] for c in expected["concern_facts"]],
                "required_concern_indices": expected["required_concerns"],
                "selected_concern_indices": safe_indices(selected.get("concerns")),
                "expected_status": expected["facts"][0]["status"],
                "selected_status": selected.get("status") if selected.get("status") in {"approved", "partial", "denied", "manual_review"} else "INVALID_OR_MISSING",
                "forbidden_fields": [sanitize_output(field) for field in sorted(set(selected)-{"item_index", "status", "fact_indices", "concerns"})]})
        return {"forbidden_root_fields": [sanitize_output(field) for field in sorted(set(value)-{"items"})], "items": result}
    except (ValueError, TypeError, AttributeError):
        return {"field": "response", "reason": "invalid_json_object"}


class StructuredClient:
    """Reuse the existing transport and proven vLLM JSON-schema mechanism."""
    def __init__(self, endpoint, adapter, max_tokens=512, seed=42, timeout=180, http_client=None):
        parsed = urlsplit(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Use a loopback evaluation endpoint")
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        self.transport = RemoteVLMClient(RemoteVLMConfig(base_url=endpoint, api_key="", model=BASE_NAME,
            timeout_seconds=timeout, max_tokens=max_tokens, require_api_key=False), http_client)
        self.adapter = Path(adapter).resolve()
        self.max_tokens, self.seed = max_tokens, seed
        self.format = response_format()

    def check_models(self):
        version = self.transport._request("GET", "/version", api_root=True)
        if version.get("version") != "0.11.2":
            raise RemoteVLMError("Expected the inspected vLLM 0.11.2 structured-output API")
        cards = self.transport._request("GET", "/models").get("data", [])
        if not isinstance(cards, list):
            raise RemoteVLMError("Malformed served model list")
        base = next((c for c in cards if isinstance(c, dict) and c.get("id") == BASE_NAME), None)
        dpo = next((c for c in cards if isinstance(c, dict) and c.get("id") == DPO_NAME), None)
        if not base or base.get("parent") or base.get("root") != MODEL_ID:
            raise RemoteVLMError("reasoning-base is absent or is not the original TEXT backbone")
        if not dpo or dpo.get("parent") != BASE_NAME or not isinstance(dpo.get("root"), str) or Path(dpo["root"]).resolve() != self.adapter:
            raise RemoteVLMError("reasoning-dpo does not map to the pointer-verified text adapter")
        return {"version": version, "base": base, "dpo": dpo,
                "verification": "vLLM model-card routing; server loads base without LoRA and adapter by separate served ID"}

    def token_ids(self, messages, model):
        result = self.transport._request("POST", "/tokenize", {"model": model,
            "messages": copy.deepcopy(messages), "add_generation_prompt": True}, api_root=True)
        ids = result.get("tokens")
        count, maximum = result.get("count"), result.get("max_model_len")
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or count != len(ids) or type(maximum) is not int:
            raise RemoteVLMError("Invalid vLLM chat tokenization response")
        if count+self.max_tokens > min(CONTEXT_LENGTH, maximum):
            raise RemoteVLMError("Prompt plus output budget exceeds 2048; refuse truncation")
        return ids

    def generate(self, messages, mode):
        if mode not in {"baseline", "dpo"}:
            raise ValueError("Invalid evaluation mode")
        if (not isinstance(messages, list) or len(messages) != 2 or
                any(not isinstance(m, dict) or set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages) or
                [m["role"] for m in messages] != ["system", "user"]):
            raise ValueError("Generation accepts only the original system/user prompt, never assistant answers")
        model = BASE_NAME if mode == "baseline" else DPO_NAME
        payload = {"model": model, "messages": copy.deepcopy(messages), "temperature": 0,
            "seed": self.seed, "max_tokens": self.max_tokens, "response_format": copy.deepcopy(self.format)}
        start = time.monotonic()
        response = self.transport._request("POST", "/chat/completions", payload)
        if response.get("model") != model:
            raise RemoteVLMError("Server responded with the wrong baseline/adapter model ID")
        try:
            choice = response["choices"][0]
            raw = choice["message"]["content"]
            tokens = response["usage"]["completion_tokens"]
            if not isinstance(raw, str) or type(tokens) is not int or not 0 <= tokens <= self.max_tokens:
                raise TypeError("Invalid output/token usage")
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteVLMError("Missing real model output or measured completion token count") from exc
        return raw, {"model": model, "mode": mode, "structured_decoding": True,
            "finish_reason": choice.get("finish_reason"), "generation_tokens": tokens,
            "generation_seconds": time.monotonic()-start}


def assess(raw, record, annotation, generation):
    result = assess_generation(raw, record, annotation, truncated=generation["finish_reason"] != "stop")
    result["runtime_validator_accepted"] = None
    if generation["finish_reason"] != "stop" and "generation_did_not_reach_eos" not in result["failures"]:
        result["failures"].append("generation_did_not_reach_eos")
    if result["valid_schema"] and not result["truncated"]:
        try:
            replay_review(raw, annotation)
            result["runtime_validator_accepted"] = True
        except (ValueError, ArithmeticError) as exc:
            result["semantic_correct"] = False
            result["runtime_validator_accepted"] = False
            result["correctness_status"] = "INCORRECT"
            result["failures"].append("runtime_validator_rejected")
            result["runtime_diagnostics"] = getattr(exc, "diagnostics", {"category": type(exc).__name__})
    result["selection_diagnostics"] = selection_diagnostics(raw, record)
    return result


def checked_adapter():
    verified = verify_latest(OUTPUT_ROOT)
    adapter = Path(verified["adapter"])
    config = strict_json((adapter.parent/"training_config.json").read_text())
    if config.get("model_id") != MODEL_ID or config.get("model") != MODEL_ID:
        raise ValueError("Expected DPO adapter trained on Qwen2.5-3B TEXT model")
    return verified


def server_command(adapter):
    # Print only; NEVER subprocess/exec a service from this module.
    python = Path.home()/".venv-vllm/bin/python"
    return shlex.join([str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL_ID,
        "--tokenizer", adapter["adapter"], "--served-model-name", BASE_NAME,
        "--host", "127.0.0.1", "--port", "8002", "--dtype", "bfloat16", "--max-model-len", "2048",
        "--max-num-seqs", "1", "--gpu-memory-utilization", "0.75", "--generation-config", "vllm",
        "--enable-lora", "--max-lora-rank", str(adapter["rank"]), "--max-loras", "1",
        "--lora-modules", f"{DPO_NAME}={adapter['adapter']}"])


def run(args):
    if not 1 <= args.max_new_tokens < CONTEXT_LENGTH or args.seed < 0:
        raise ValueError("Invalid token budget or seed")
    if Path(sys.prefix).resolve() != (ROOT.parent/".venv").resolve() or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Activate original ../.venv Python 3.12; server uses the separate ~/.venv-vllm")
    verified = checked_adapter()
    if args.print_server_command:
        print(server_command(verified), flush=True)
        return
    if args.manual_review_smoke:
        if args.mode != "both" or args.split != "validation" or args.limit not in (None, 1):
            raise ValueError("Standalone manual-review smoke requires both modes and cannot select a dataset split/limit")
        from .manual_review_fixture import load_manual_review_fixture
        entries, audit = load_manual_review_fixture()
    else:
        entries, audit = load_cases(args.split, args.limit, args.acknowledge_observed_test)
    adapter = Path(verified["adapter"])
    provenance = strict_json((adapter.parent/"dataset_manifest.json").read_text())
    if (provenance.get("test_accessed") is not False or
            (not args.manual_review_smoke and provenance.get("manifest_sha256") != audit["manifest_sha256"])):
        raise ValueError("Dataset/training provenance mismatch")
    meta = {"model_id": MODEL_ID, "adapter": verified, "dataset": audit,
        "schema": response_format(), "schema_sha256": sha(ROOT/"src/agents/contracts.py"),
        "endpoint": args.endpoint, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "decoding": "identical temperature=0 and generic strict JSON schema; no gold constraints",
        "versions": {name: metadata.version(name) for name in ("httpx", "pydantic", "transformers", "peft")}}
    if args.check:
        print(json.dumps({**meta, "server_and_inference": "not checked"}, indent=2), flush=True)
        return
    client = StructuredClient(args.endpoint, adapter, args.max_new_tokens, args.seed, args.timeout)
    meta["server"] = client.check_models()
    # Saved tokenizer is local-only, never downloads or loads model weights.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    saved_config = strict_json((adapter.parent/"training_config.json").read_text())
    import hashlib
    template_sha = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    if template_sha != saved_config.get("chat_template_sha256"):
        raise ValueError("Saved tokenizer/template differs from successful training")
    for record, _ in entries:
        ids = client.token_ids(record["prompt"], BASE_NAME)
        if ids != client.token_ids(record["prompt"], DPO_NAME) or ids != tokenizer.apply_chat_template(record["prompt"], tokenize=True, add_generation_prompt=True):
            raise ValueError("Baseline/DPO/server/training chat-template boundaries differ")
    if args.check_server:
        print(json.dumps({**meta, "chat_template_sha256": template_sha, "inference": "not performed"}, indent=2), flush=True)
        return
    modes = ("baseline", "dpo") if args.mode == "both" else (args.mode,)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8]
    root = output_root(args.output_dir)
    pending = root/(run_id+".pending")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        predictions = {mode: [] for mode in modes}
        for record, annotation in entries:
            for mode in modes:
                # Nothing except ORIGINAL system/user messages goes to the model.
                raw, generation = client.generate(record["prompt"], mode)
                assessment = assess(raw, record, annotation, generation)
                predictions[mode].append({"pair_id": annotation["pair_id"], "agent_role": "agent_2",
                    "scenario_family": annotation["scenario_family"], "expected_items": strict_json(record["chosen"][0]["content"])["items"],
                    "raw_output_sanitized": sanitize_output(raw), "assessment": assessment, **generation})
                print(f"{mode}: Agent 2 {len(predictions[mode])}/{len(entries)} semantic_correct={assessment['semantic_correct']}", flush=True)
        metrics = {mode: summarize(rows, []) for mode, rows in predictions.items()}
        for values in metrics.values():
            values.pop("preference")  # generation-only diagnostic; no invented ranking scores
        for mode, rows in predictions.items():
            (pending/f"{mode}_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
        reviews = []
        for index, (_, annotation) in enumerate(entries):
            reviews.append({"pair_id": annotation["pair_id"], "agent_role": "agent_2", "human_review": "PENDING",
                **{mode: {"output": predictions[mode][index]["raw_output_sanitized"],
                    "failures": predictions[mode][index]["assessment"]["failures"],
                    "diagnostics": predictions[mode][index]["assessment"]["selection_diagnostics"]} for mode in modes}})
        (pending/"human_review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in reviews), encoding="utf-8")
        (pending/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
        meta.update(status="completed", run_id=run_id, chat_template_sha256=template_sha,
            base_weight_identity="same loaded vLLM base for both served model IDs; server model-card routing audited",
            conclusion="Diagnostic only; schema constraints do not establish semantic correctness or deployment readiness")
        (pending/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
        lines = ["# Agent 2 constrained-generation diagnostic", "", audit["evaluation_kind"],
            "Not a new independent held-out evaluation. No preference ranking recalculated.", "",
            "| Mode | JSON % | Schema % | Scoped references % | Semantic % |", "|---|---:|---:|---:|---:|"]
        for mode, values in metrics.items():
            percentages = [values[key]["percent"] for key in ("valid_json", "valid_schema", "correct_scoped_references", "semantic_correct")]
            lines.append(f"| {mode} | "+" | ".join("N/A" if p is None else f"{p:.1f}" for p in percentages)+" |")
        lines += ["", "Inspect actual generations, allowed/selected references and runtime diagnostics in prediction JSONL.",
            "Remaining semantic failures require investigation on non-test cases; do not assume additional training will help without evidence.",
            "Before Phase 9D, require schema and runtime-validator success on non-test approved/capped/excluded/manual-review cases, preserved amounts, and new independently held-out evaluation."]
        (pending/"comparison.md").write_text("\n".join(lines)+"\n")
        pending.rename(root/run_id)
        print(f"Diagnostic reports: {root/run_id}", flush=True)
    except BaseException as exc:
        (pending/"failure.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": sanitize_output(str(exc))})+"\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=["both", "baseline", "dpo"], default="both")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002")
    parser.add_argument("--output-dir", default="reports/dpo/agent2_structured")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--acknowledge-observed-test", action="store_true")
    parser.add_argument("--manual-review-smoke", action="store_true", help="One NEW canonical medical-review fixture outside all preference splits; both models mandatory")
    parser.add_argument("--check", action="store_true", help="CPU preflight, no network/model inference")
    parser.add_argument("--check-server", action="store_true", help="Model-card and tokenization checks, no generation")
    parser.add_argument("--print-server-command", action="store_true", help="Print only; never start vLLM")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
