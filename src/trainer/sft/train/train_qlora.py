"""Manual single-GPU Qwen2.5-VL QLoRA SFT; --dry-run uses only CPU dataset code."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

from src.trainer.sft.config import (
    MODEL_ID, MAX_STEPS, LEARNING_RATE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    GRAD_ACCUM, MIN_PIXELS, MAX_PIXELS, SEED,
    dataset_root, output_dir,
)
from src.trainer.sft.training_data import make_hf_dataset, read_records
from src.utils.paths import project_path
from .checkpoints import TARGET_MODULES, verify_adapter

PACKAGES = ("torch", "torchvision", "transformers", "trl", "peft", "bitsandbytes",
            "accelerate", "datasets", "Pillow", "safetensors")


def versions():
    result = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", "--base-model", default=MODEL_ID)
    cli.add_argument("--dataset-root", type=Path, default=dataset_root())
    cli.add_argument("--train-manifest", type=Path)
    cli.add_argument("--validation-manifest", type=Path)
    cli.add_argument("--output-dir", type=Path, default=output_dir())
    cli.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    cli.add_argument("--max-steps", type=int, default=MAX_STEPS)
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--grad-accum", "--gradient-accumulation", type=int, default=GRAD_ACCUM)
    cli.add_argument("--lora-rank", type=int, default=LORA_RANK)
    cli.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    cli.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    cli.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    cli.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    cli.add_argument("--logging-steps", type=int, default=1)
    cli.add_argument("--save-steps", type=int, default=4)
    cli.add_argument("--seed", type=int, default=SEED)
    cli.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    cli.add_argument("--optimizer", choices=("paged_adamw_8bit", "adamw_torch"), default="paged_adamw_8bit")
    cli.add_argument("--gpu-memory-fraction", type=float)
    cli.add_argument("--log-gpu-utilization", action="store_true",
                     help="Query nvidia-smi at log events (one-second timeout); no background process")
    cli.add_argument("--no-evaluation", action="store_true", help="Skip final validation loss; keep splits separate")
    cli.add_argument("--dry-run", action="store_true", help="CPU dataset checks only; no CUDA access or model download")
    return cli


def check_arguments(args):
    for name in ("max_steps", "batch_size", "grad_accum", "lora_rank", "lora_alpha",
                 "min_pixels", "max_pixels", "logging_steps", "save_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_pixels > args.max_pixels:
        raise ValueError("min_pixels must be <= max_pixels")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be in (0, 1]")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")


def run(args):
    check_arguments(args)
    library_versions = versions()  # No Torch import or CUDA initialization.
    print("Library versions: " + json.dumps(library_versions), flush=True)
    root = project_path(args.dataset_root).resolve()
    manifests = {
        "train": project_path(args.train_manifest) if args.train_manifest else root / "train/annotations.jsonl",
        "validation": project_path(args.validation_manifest) if args.validation_manifest else root / "validation/annotations.jsonl",
    }
    from src.trainer.sft.annotation.validation import verify_dataset
    validation_report = verify_dataset(root, check_hf=True, manifests=manifests)
    print("Dataset: " + json.dumps(validation_report), flush=True)
    records = {split: read_records(root, split, manifest=path) for split, path in manifests.items()}
    selected = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    selected.update(dataset_root=str(root), train_manifest=str(manifests["train"]),
                    validation_manifest=str(manifests["validation"]), target_modules=TARGET_MODULES,
                    train_images=len(records["train"]), validation_images=len(records["validation"]),
                    gradient_checkpointing=True, quantization="4-bit NF4, double quantization",
                    trainable_scope="language attention LoRA only; vision tower/merger and base weights frozen")
    print("Configuration: " + json.dumps(selected), flush=True)
    if args.dry_run:
        print("CPU dataset dry-run PASS. No processor/model loaded or output/checkpoints created. "
              "Model-specific collator and forward/backward checks require the manual training run.", flush=True)
        return None
    missing = [name for name, value in library_versions.items() if value == "MISSING"]
    if missing:
        raise RuntimeError("Missing training dependencies: " + ", ".join(missing))
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from trl import SFTConfig
    from .runtime import AuditedSFTTrainer, CheckedInvoiceCollator, LiveProgress, log

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required; use CUDA_VISIBLE_DEVICES=0")
    supported = torch.cuda.is_bf16_supported()
    if args.dtype == "bf16" and not supported:
        raise RuntimeError("BF16 requested but unsupported on the visible GPU")
    bf16 = args.dtype == "bf16" or (args.dtype == "auto" and supported)
    dtype = torch.bfloat16 if bf16 else torch.float16
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        log("Allocator fraction limits only PyTorch allocations; it is not a cap on total GPU usage")
    free, total = torch.cuda.mem_get_info(0)
    log(f"CUDA available=True runtime={torch.version.cuda} GPU={torch.cuda.get_device_name(0)} "
        f"total={total/2**30:.2f}GiB free={free/2**30:.2f}GiB dtype={dtype}")
    log("vLLM compatibility preference: language q/k/v/o projections only. "
        "Future vLLM must support Qwen2.5-VL LoRA and Transformers 4.57 language_model tensor mapping.")
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    output = project_path(args.output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir = run_dir / "adapter"
    selected.update(run_id=run_id, output_dir=str(output), dtype=str(dtype), versions=library_versions,
                    model_id=args.model, started_utc=datetime.now(timezone.utc).isoformat(),
                    manifest_sha256={split: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for split, path in manifests.items()})
    write_json(run_dir / "run_config.json", selected)
    log(f"Run directory: {run_dir}; periodic checkpoints: {run_dir / 'checkpoints'}")
    started = time.monotonic()
    try:
        log("Loading Qwen processor/chat template and NF4 base model; uncached files download from Hugging Face")
        loading = time.monotonic()
        processor = AutoProcessor.from_pretrained(
            args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
            use_fast=False, trust_remote_code=False,
        )
        if not processor.chat_template or processor.tokenizer.pad_token_id is None:
            raise RuntimeError("Qwen processor must have a chat template and padding token")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, quantization_config=quantization, device_map={"": 0},
            dtype=dtype, attn_implementation="sdpa", trust_remote_code=False,
        )
        log(f"Processor/base-model loading completed in {time.monotonic()-loading:.1f}s")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        targets = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in TARGET_MODULES]
        if not targets or any(".language_model.layers." not in name or ".self_attn." not in name for name in targets):
            raise RuntimeError("LoRA suffixes target unexpected modules; review the model architecture")
        if {name.rsplit(".", 1)[-1] for name in targets} != set(TARGET_MODULES):
            raise RuntimeError("Some language attention projections are missing")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
            modules_to_save=None, use_dora=False, use_rslora=False,
        ))
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or any(".language_model." not in name or ".lora_" not in name for name in trainable_names):
            raise RuntimeError("Only language-model LoRA weights may be trainable")
        trainable, total_parameters = model.get_nb_trainable_parameters()
        selected.update(total_parameters=total_parameters, trainable_parameters=trainable,
                        resolved_target_modules=targets, processor_class=type(processor).__name__,
                        tokenizer_class=type(processor.tokenizer).__name__)
        write_json(run_dir / "run_config.json", selected)
        log(f"Total parameters={total_parameters:,}; trainable LoRA={trainable:,} "
            f"({100*trainable/total_parameters:.4f}%); vision/merger frozen")
        datasets = {split: make_hf_dataset(value) for split, value in records.items()}
        collator = CheckedInvoiceCollator(processor)
        for split, dataset in datasets.items():
            for example in dataset:
                collator([example])
            log(f"Multimodal collator preflight PASS: {split}, {len(dataset)} images; exact CSV loss targets")
        training_args = SFTConfig(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum, max_steps=args.max_steps,
            learning_rate=args.learning_rate, lr_scheduler_type="constant", warmup_steps=0,
            fp16=not bf16, bf16=bf16, optim=args.optimizer,
            logging_steps=args.logging_steps, logging_first_step=True,
            save_strategy="steps", save_steps=args.save_steps, save_safetensors=True,
            eval_strategy="no", prediction_loss_only=True, eval_accumulation_steps=1,
            report_to="none", remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=True,
            max_length=None, packing=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0, dataloader_pin_memory=False,
            seed=args.seed, data_seed=args.seed, disable_tqdm=False,
        )
        write_json(run_dir / "training_args.json", training_args.to_dict())
        trainer = AuditedSFTTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets["validation"], processing_class=processor,
            data_collator=collator, callbacks=[LiveProgress(args.log_gpu_utilization)],
        )
        log(f"Training starts: {args.max_steps} optimizer steps; adapter destination: {adapter_dir}")
        training_started = time.monotonic()
        result = trainer.train()
        training_duration = time.monotonic() - training_started
        if result.global_step != args.max_steps or not math.isfinite(float(result.training_loss)):
            raise RuntimeError("Training did not complete all requested steps with a finite loss")
        eval_metrics = {"status": "skipped_by_user"} if args.no_evaluation else trainer.evaluate()
        if not args.no_evaluation and not math.isfinite(float(eval_metrics["eval_loss"])):
            raise RuntimeError("Validation loss is non-finite")
        log("Saving PEFT adapter and Qwen processor/tokenizer")
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        processor.save_pretrained(adapter_dir)
        metrics = {
            "training": result.metrics, "evaluation": eval_metrics,
            "training_loss": float(result.training_loss), "global_steps_completed": int(result.global_step),
            "training_duration_seconds": training_duration, "total_duration_seconds": time.monotonic()-started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(0)/2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(0)/2**30,
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "training_log.json", trainer.state.log_history)
        artifacts = verify_adapter(adapter_dir, args.model)
        write_json(run_dir / "training_result.json", {
            "status": "success", "model_id": args.model, "run_id": run_id,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "versions": library_versions, "metrics": metrics, "adapter": artifacts,
        })
        # Only verified genuine adapters update the atomic, repository-relative pointer.
        pointer = {"model_id": args.model, "run_id": run_id,
                   "adapter_path": adapter_dir.relative_to(output).as_posix(),
                   "adapter_sha256": artifacts["sha256"]}
        temporary_pointer = output / f".latest-{run_id}.json"
        write_json(temporary_pointer, pointer)
        os.replace(temporary_pointer, output / "latest_successful.json")
        log("Training complete: " + json.dumps(metrics))
        log(f"Verified adapter saved: {adapter_dir}")
        log("Saved artifact files: " + ", ".join(artifacts["artifacts"]))
        return adapter_dir
    except BaseException as exc:
        write_json(run_dir / "training_result.json", {
            "status": "failed", "run_id": run_id, "error_type": type(exc).__name__,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        log(f"Run failed ({type(exc).__name__}); checkpoints retained at {run_dir}. Successful pointer unchanged.")
        raise


def train(dataset_root: Path, output_dir: Path, **kwargs) -> Path:
    """Preserve the previous Python API and legacy notebook entry points."""
    args = parser().parse_args([])
    args.dataset_root, args.output_dir = dataset_root, output_dir
    for name, value in kwargs.items():
        name = {"model_name": "model"}.get(name, name)
        if not hasattr(args, name):
            raise TypeError(f"Unknown training argument: {name}")
        setattr(args, name, value)
    return run(args)


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
