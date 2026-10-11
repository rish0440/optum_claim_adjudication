"""Manual Phase 9C test-only evaluation of the SAME text base with/without DPO."""
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

from src.adjudication.policy import load_policy
from src.trainer.preference_optimization.annotation.cases import source_hashes
from src.trainer.preference_optimization.annotation.validate import strict_json, validate_records
from src.trainer.preference_optimization.train.config import MODEL_ID, OUTPUT_ROOT, PINNED
from src.trainer.preference_optimization.train.dataset import serialize_and_audit
from src.trainer.preference_optimization.train.train_dpo import verify_latest
from src.utils.paths import ROOT, project_path
from .metrics import aggregate, assess_generation, preference_score, sanitize_output
from .report import write_reports

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


def load_test(directory="data/dpo"):
    """Never open train/validation. Reuse canonical validation for test only."""
    directory = project_path(directory).resolve()
    manifest = strict_json((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hashes() or manifest["policy_sha256"] != sha(ROOT/"data/policy.json"):
        raise ValueError("Policy/contracts differ from validated Phase 9A data; do not relabel the held-out test")
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
    counts = validate_records(entries, load_policy(ROOT/"data/policy.json"))
    if counts["test"] != {"agent_1": 5, "agent_2": 5}:
        raise ValueError("Expected 5 test pairs per agent")
    return entries, {"split": "test", "pairs": 10, "by_role": counts["test"],
        "underlying_cases": len({a["synthetic_case_id"] for a in annotations}),
        "test_sha256": sha(path), "manifest_sha256": sha(directory/"manifest.json"),
        "policy_sha256": manifest["policy_sha256"], "source_sha256": manifest["source_sha256"],
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
            evaluator_source_sha256={name: sha(Path(__file__).parent/name) for name in ("evaluate.py", "metrics.py", "report.py")},
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
    parser.add_argument("--dataset-dir", default="data/dpo")
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
