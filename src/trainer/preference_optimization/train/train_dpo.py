"""Manual Qwen2.5-3B TEXT-model DPO/QLoRA; no application or serving changes."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

from src.utils.paths import ROOT
from .config import MODEL_ID, PINNED, TARGETS, TrainingConfig, dpo_arguments, language_targets, safe_output_root
from .dataset import load_preferences, prepare_preferences


def log(text):
    print(time.strftime("[%H:%M:%S] ")+str(text), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def preflight(config, *, check_gpu=False, check_tokenizer=False):
    config.validate()
    expected = (ROOT.parent/".venv").resolve()
    if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
        raise RuntimeError(f"Activate the existing application/training environment: {expected}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python 3.12 is required")
    versions = {name: metadata.version(name) for name in (*PINNED, "torch")}
    for name, required in PINNED.items():
        if versions[name] != required:
            raise RuntimeError(f"Expected installed {name}=={required}; found {versions[name]}. Do not upgrade automatically")
    from trl import DPOTrainer, DPOConfig
    from transformers import Qwen2ForCausalLM, AutoTokenizer
    fields = set(DPOConfig.__dataclass_fields__)
    if not set(dpo_arguments(config, Path("unused"), True)) <= fields:
        raise RuntimeError("Installed DPOConfig differs from the inspected TRL 0.24.0 API")
    if not {"ref_model", "peft_config", "processing_class"} <= set(inspect.signature(DPOTrainer.__init__).parameters):
        raise RuntimeError("Installed DPOTrainer lacks the required PEFT/reference API")
    # Guard the implementation assumptions against a locally modified trainer.
    source = inspect.getsource(DPOTrainer.null_ref_context)
    if "disable_adapter" not in source or not hasattr(DPOTrainer, "tokenize_row"):
        raise RuntimeError("Adapter-disabled reference or text tokenization differs from inspected source")
    data, audit = load_preferences(config.dataset_dir)
    result = {"python": platform.python_version(), "interpreter": sys.executable, "prefix": sys.prefix,
              "versions": versions, "training_model": MODEL_ID, "dataset": audit,
              "output_root": str(safe_output_root(config.output_dir)),
              "reference": "frozen NF4 base with new LoRA disabled; reference_free=False",
              "chat_template": "pending manual cached-tokenizer check or training startup",
              "gpu_kernels_and_training": "not verified by preflight"}
    if check_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=False)
        prepare_preferences(data, tokenizer, config.max_length)
        result["chat_template"] = "cached text-model template and all train/validation token lengths verified"
    if check_gpu:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU required; use CUDA_VISIBLE_DEVICES=0")
        free, total = torch.cuda.mem_get_info()
        result["gpu"] = {"name": torch.cuda.get_device_name(0), "free_gib": free/2**30,
                         "total_gib": total/2**30, "cuda": torch.version.cuda,
                         "bf16_supported": torch.cuda.is_bf16_supported()}
    return data, result


def build_trainer(model, tokenizer, datasets, config, run_dir, bf16, *, trainer_cls=None,
                  dpo_config_cls=None, lora_config_cls=None, callbacks=None):
    if trainer_cls is None or dpo_config_cls is None:
        from trl import DPOConfig
        trainer_cls = trainer_cls or audited_trainer_class()
        dpo_config_cls = dpo_config_cls or DPOConfig
    if lora_config_cls is None:
        from peft import LoraConfig
        lora_config_cls = LoraConfig
    return trainer_cls(model=model, ref_model=None,
        args=dpo_config_cls(**dpo_arguments(config, run_dir, bf16)),
        peft_config=lora_config_cls(r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=TARGETS, modules_to_save=None,
            use_dora=False, use_rslora=False), processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"], callbacks=callbacks or [])


def audited_trainer_class():
    import torch
    from trl import DPOTrainer

    class AuditedDPOTrainer(DPOTrainer):
        """Audit the standard TRL objective/reference; do not replace its math."""
        policy_calls = 0
        reference_calls = 0
        gradient_checks = 0

        def concatenated_forward(self, model, batch, is_ref_model=False):
            if "pixel_values" in batch:
                raise RuntimeError("Reasoning DPO must contain text-only preference batches")
            if is_ref_model:
                modules = [m for m in self.accelerator.unwrap_model(model).modules() if hasattr(m, "lora_A")]
                if not modules or any(not m.disable_adapters for m in modules):
                    raise RuntimeError("Reference forward did not disable all trainable LoRA adapters")
                self.reference_calls += 1
            else:
                self.policy_calls += 1
            output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
            for key in ("chosen_logps", "rejected_logps"):
                if not torch.isfinite(output[key]).all().item():
                    raise RuntimeError("Nonfinite DPO preference log probabilities")
            return output

        def training_step(self, model, inputs, *args, **kwargs):
            loss = super().training_step(model, inputs, *args, **kwargs)
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not torch.isfinite(loss).all().item():
                raise RuntimeError("No real LoRA backward gradients or nonfinite DPO loss")
            if not self.args.fp16 and any(not torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError("Nonfinite LoRA gradients")
            self.gradient_checks += 1
            return loss
    return AuditedDPOTrainer


def progress_callback():
    import torch
    from transformers import TrainerCallback
    class Progress(TrainerCallback):
        def __init__(self):
            self.started = time.monotonic()
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = logs or {}
            if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
                raise RuntimeError("Training/evaluation metric is nonfinite")
            log(f"step={state.global_step}/{state.max_steps} elapsed={time.monotonic()-self.started:.1f}s "
                f"allocated={torch.cuda.memory_allocated()/2**30:.2f}GiB reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB "
                + json.dumps(values))
        def on_save(self, args, state, control, **kwargs):
            log(f"Adapter checkpoint saved: {args.output_dir}/checkpoint-{state.global_step}")
    return Progress()


def verify_adapter(adapter):
    """CPU tensor checks for TEXT-model adapters (SFT helpers expect VL keys)."""
    import torch
    from safetensors import safe_open
    adapter = Path(adapter)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json"):
        if not (adapter/name).is_file() or not (adapter/name).stat().st_size:
            raise RuntimeError(f"Missing/empty adapter artifact: {name}")
    config = json.loads((adapter/"adapter_config.json").read_text())
    rank = config.get("r")
    if (config.get("base_model_name_or_path") != MODEL_ID or config.get("peft_type") != "LORA"
            or set(config.get("target_modules", [])) != set(TARGETS) or config.get("bias") != "none"
            or config.get("modules_to_save") or type(rank) is not int or rank < 1):
        raise RuntimeError("Unexpected text-model adapter configuration")
    weights = adapter/"adapter_model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise RuntimeError("Adapter has no tensors")
        for key in keys:
            if (".model.layers." not in key or ".self_attn." not in key or
                    not any(f".{target}.lora_" in key for target in TARGETS)):
                raise RuntimeError("Adapter contains tensors outside text attention LoRA")
            shape = handle.get_slice(key).get_shape()
            is_a = ".lora_A." in key
            if not is_a and ".lora_B." not in key:
                raise RuntimeError("Unexpected LoRA tensor")
            partner = key.replace(".lora_A.", ".lora_B.") if is_a else key.replace(".lora_B.", ".lora_A.")
            if (len(shape) != 2 or min(shape) < 1 or shape[0 if is_a else 1] != rank
                    or partner not in keys or not torch.isfinite(handle.get_tensor(key)).all().item()):
                raise RuntimeError("Invalid, unpaired or nonfinite LoRA tensor")
    return {"adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "rank": rank,
            "tensor_count": len(keys), "artifacts": sorted(p.name for p in adapter.iterdir() if p.is_file())}


def publish_success(root, run_dir, config, metrics, artifact_info):
    """Never publish an incomplete/failed run or overwrite extraction pointers."""
    root = safe_output_root(root)
    if run_dir.parent != root/"runs" or not (run_dir/"adapter").is_dir():
        raise RuntimeError("Unsafe DPO checkpoint directory")
    if (metrics.get("global_steps_completed") != config.max_steps or
            not all(math.isfinite(float(metrics[key])) for key in ("training_loss", "eval_loss")) or
            min(metrics.get(key, 0) for key in ("policy_calls", "reference_calls", "gradient_checks")) < 1 or
            not metrics.get("adapter_changed")):
        raise RuntimeError("Cannot mark an unverified optimization run successful")
    for name in ("training_config.json", "training_args.json", "dataset_manifest.json", "metrics.json", "training_log.json"):
        json.loads((run_dir/name).read_text())
    saved = json.loads((run_dir/"training_config.json").read_text())
    audit = json.loads((run_dir/"dataset_manifest.json").read_text())
    if (json.loads((run_dir/"metrics.json").read_text()) != metrics or
            saved.get("model_id") != MODEL_ID or saved.get("max_steps") != config.max_steps or
            audit.get("train_pairs") != 60 or audit.get("validation_pairs") != 10 or
            audit.get("test_accessed") is not False):
        raise RuntimeError("Saved metadata does not verify the completed training run")
    actual = verify_adapter(run_dir/"adapter")
    if actual != artifact_info or actual["rank"] != config.rank:
        raise RuntimeError("Adapter verification differs from saved run")
    write_json(run_dir/"training_result.json", {"status": "success", "model_id": MODEL_ID,
        "run_id": run_dir.name, "finished_utc": datetime.now(timezone.utc).isoformat(), "adapter": actual})
    pointer = {"model_id": MODEL_ID, "run_id": run_dir.name,
               "adapter_path": (run_dir/"adapter").relative_to(root).as_posix(), "adapter_sha256": actual["adapter_sha256"]}
    temporary = root/f".latest-{run_dir.name}.json"
    write_json(temporary, pointer)
    os.replace(temporary, root/"latest_successful.json")


def verify_latest(root):
    root = safe_output_root(root)
    pointer = json.loads((root/"latest_successful.json").read_text())
    adapter = (root/pointer["adapter_path"]).resolve()
    if not adapter.is_relative_to(root/"runs") or adapter.name != "adapter" or adapter.parent.name != pointer["run_id"]:
        raise RuntimeError("Unsafe adapter pointer")
    result = json.loads((adapter.parent/"training_result.json").read_text())
    metrics = json.loads((adapter.parent/"metrics.json").read_text())
    if result.get("status") != "success" or result.get("run_id") != pointer["run_id"] or result.get("model_id") != MODEL_ID or pointer.get("model_id") != MODEL_ID or metrics.get("global_steps_completed", 0) < 1:
        raise RuntimeError("Pointer does not identify a completed reasoning run")
    artifacts = verify_adapter(adapter)
    if artifacts != result.get("adapter") or artifacts["adapter_sha256"] != pointer["adapter_sha256"]:
        raise RuntimeError("Adapter checksum mismatch")
    return {"adapter": str(adapter), **artifacts}


def train(config):
    data, preflight_report = preflight(config, check_gpu=True)
    log(json.dumps(preflight_report))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    # Manual prerequisite; do not terminate other GPU processes automatically.
    free, total = torch.cuda.mem_get_info()
    if free < 18*2**30:
        raise RuntimeError("Less than 18 GiB GPU memory free. Stop vLLM and other GPU work manually")
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    set_seed(config.seed)
    root = safe_output_root(config.output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8]
    run_dir = root/"runs"/run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    saved_config = {**asdict(config), "model_id": MODEL_ID, "versions": preflight_report["versions"],
        "run_id": run_id, "dtype": str(dtype), "reference": preflight_report["reference"],
        "effective_lora_dropout": 0, "selection": "lowest validation DPO loss; test never opened"}
    write_json(run_dir/"training_config.json", saved_config)
    write_json(run_dir/"dataset_manifest.json", preflight_report["dataset"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        log("Loading NEW text-model tokenizer; cache/download behavior is controlled by --local-files-only")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=config.local_files_only)
        datasets = prepare_preferences(data, tokenizer, config.max_length)
        log("Text Qwen template and exact JSON completion tokens validated; no truncation")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=False,
            local_files_only=config.local_files_only, device_map={"": 0}, dtype=dtype, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        if model.config.model_type != "qwen2" or model.config.is_encoder_decoder:
            raise RuntimeError("Expected Qwen2ForCausalLM text architecture")
        model.config.use_cache = False
        saved_config["resolved_targets"] = language_targets([name for name, _ in model.named_modules()])
        saved_config["chat_template_sha256"] = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
        write_json(run_dir/"training_config.json", saved_config)
        trainer = build_trainer(model, tokenizer, datasets, config, run_dir, bf16, callbacks=[progress_callback()])
        if trainer.ref_model is not None or not trainer.is_peft_model or trainer.reference_free:
            raise RuntimeError("Expected TRL adapter-disabled frozen-base reference")
        if trainer.is_vision_model or trainer.accelerator.num_processes != 1:
            raise RuntimeError("Only single-process TEXT DPO is supported")
        trainable = {name: p for name, p in trainer.model.named_parameters() if p.requires_grad}
        if not trainable or any(".lora_" not in name or ".model.layers." not in name or ".self_attn." not in name for name in trainable):
            raise RuntimeError("Only text-attention LoRA parameters may be trained")
        initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        log(f"Trainable LoRA parameters={sum(p.numel() for p in trainable.values()):,}; steps={config.max_steps}; run={run_dir}")
        write_json(run_dir/"training_args.json", trainer.args.to_dict())
        result = trainer.train()
        evaluation = trainer.evaluate()  # validation only, selected best checkpoint
        selected_parameters = dict(trainer.model.named_parameters())
        changed = any(not torch.equal(value, selected_parameters[name].detach().cpu()) for name, value in initial.items())
        metrics = {"training_loss": float(result.training_loss), "eval_loss": float(evaluation["eval_loss"]),
            "training": result.metrics, "evaluation": evaluation, "global_steps_completed": result.global_step,
            "policy_calls": trainer.policy_calls, "reference_calls": trainer.reference_calls,
            "gradient_checks": trainer.gradient_checks, "adapter_changed": changed,
            "best_checkpoint": trainer.state.best_model_checkpoint, "duration_seconds": time.monotonic()-started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        trainer.model.save_pretrained(run_dir/"adapter", safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(run_dir/"adapter")
        write_json(run_dir/"metrics.json", metrics)
        write_json(run_dir/"training_log.json", trainer.state.log_history)
        info = verify_adapter(run_dir/"adapter")
        publish_success(root, run_dir, config, metrics, info)
        log("Successful reasoning adapter: "+str(run_dir/"adapter"))
        log(json.dumps(metrics))
        return run_dir/"adapter"
    except BaseException as exc:
        write_json(run_dir/"training_result.json", {"status": "failed", "run_id": run_id,
            "model_id": MODEL_ID, "error_type": type(exc).__name__})
        log(f"Run failed ({type(exc).__name__}); successful pointer unchanged. Partial artifacts retained: {run_dir}")
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for field in TrainingConfig.__dataclass_fields__.values():
        option = "--"+field.name.replace("_", "-")
        if isinstance(field.default, bool):
            p.add_argument(option, action="store_true", default=field.default)
        else:
            p.add_argument(option, type=type(field.default), default=field.default)
    p.add_argument("--check", action="store_true", help="CPU/API/data preflight; no model/tokenizer download")
    p.add_argument("--check-gpu", action="store_true", help="With --check, inspect CUDA availability/memory")
    p.add_argument("--check-tokenizer", action="store_true", help="With --check, audit cached tokenizer; never download")
    p.add_argument("--smoke", action="store_true", help="Require max-steps=1 and isolate under output-dir/smoke")
    p.add_argument("--verify-latest", action="store_true", help="CPU saved-adapter verification only")
    return p


def main():
    args = parser().parse_args()
    values = {key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__}
    config = TrainingConfig(**values)
    if args.smoke:
        if config.max_steps != 1:
            raise ValueError("Smoke mode requires --max-steps 1")
        config.output_dir = str(safe_output_root(config.output_dir)/"smoke")
    if args.verify_latest:
        log(json.dumps(verify_latest(config.output_dir), indent=2))
    elif args.check:
        _, report = preflight(config, check_gpu=args.check_gpu, check_tokenizer=args.check_tokenizer)
        log(json.dumps(report, indent=2))
    else:
        if args.check_gpu or args.check_tokenizer:
            raise ValueError("Check options require --check")
        train(config)


if __name__ == "__main__":
    main()
