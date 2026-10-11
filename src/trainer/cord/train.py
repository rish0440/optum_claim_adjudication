"""Isolated CORD QLoRA SFT. Manual execution only; no held-out test access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from src.trainer.sft.train.checkpoints import TARGET_MODULES, latest_adapter, verify_adapter
from src.trainer.sft.train.train_qlora import versions, write_json
from src.utils.paths import project_path
from .schema import sha
from .training_data import (BoundedCollator, INSTRUCTION, INSTRUCTION_VERSION,
                            load_training_data, make_dataset)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT = "models/qwen25_vl_cord_sft"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, default=Path("data/cord_200"))
    cli.add_argument("--output-dir", type=Path, default=Path(OUTPUT))
    cli.add_argument("--epochs", type=float, default=2)
    cli.add_argument("--max-steps", type=int, default=-1, help="Override epochs; 1 is always isolated smoke mode")
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--grad-accum", type=int, default=4)
    cli.add_argument("--learning-rate", type=float, default=1e-4)
    cli.add_argument("--lora-rank", type=int, default=8)
    cli.add_argument("--lora-alpha", type=int, default=16)
    cli.add_argument("--lora-dropout", type=float, default=0.05)
    cli.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    cli.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    cli.add_argument("--max-sequence-length", type=int, default=2048)
    cli.add_argument("--min-free-gib", type=float, default=18)
    cli.add_argument("--optimizer", choices=("paged_adamw_8bit", "adamw_torch"), default="paged_adamw_8bit")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--logging-steps", type=int, default=1)
    cli.add_argument("--instruction-file", type=Path, help="Optional UTF-8 general extraction instruction")
    cli.add_argument("--local-files-only", action="store_true")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="CPU data/HF/config checks; no processor/model loading")
    mode.add_argument("--smoke", action="store_true", help="Exactly one optimizer step; never publish latest pointer")
    mode.add_argument("--verify-latest", action="store_true", help="CPU verify the latest completed CORD adapter")
    mode.add_argument("--verify-adapter", type=Path, help="CPU verify an explicit CORD run adapter, including smoke")
    return cli


def check_arguments(args):
    for name in ("batch_size", "grad_accum", "lora_rank", "lora_alpha", "min_pixels", "max_pixels",
                 "max_sequence_length", "logging_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps != -1 and args.max_steps < 1:
        raise ValueError("max_steps must be -1 (epochs) or positive")
    for name in ("epochs", "learning_rate", "min_free_gib"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= args.lora_dropout < 1 or args.seed < 0 or args.min_pixels > args.max_pixels:
        raise ValueError("Invalid dropout, seed or image pixel limits")
    if args.smoke and args.max_steps not in (-1, 1):
        raise ValueError("--smoke only permits --max-steps 1")
    output = project_path(args.output_dir).resolve()
    allowed = project_path(OUTPUT).resolve()
    if not output.is_relative_to(allowed):
        raise ValueError(f"Output must be within {allowed}; existing extraction/DPO adapters are protected")
    if project_path(args.dataset_root).resolve().is_relative_to(output):
        raise ValueError("Dataset cannot be inside the model output directory")
    return output


def expected_steps(samples, batch_size, accumulation, epochs):
    return math.ceil(epochs * math.ceil(math.ceil(samples / batch_size) / accumulation))


def language_targets(model):
    targets = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in TARGET_MODULES]
    if (not targets or any(".language_model.layers." not in name or ".self_attn." not in name for name in targets)
            or {name.rsplit(".", 1)[-1] for name in targets} != set(TARGET_MODULES)):
        raise RuntimeError("Unexpected Qwen-VL attention layout; cannot safely select language-only LoRA")
    return targets


def adapter_digest(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            if ".language_model." not in name or ".lora_" not in name:
                raise RuntimeError("Only language attention LoRA parameters may be trained")
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
            count += parameter.numel()
    if not count:
        raise RuntimeError("No trainable language LoRA parameters")
    return digest.hexdigest(), count


def publish_success(output, run_dir, config, metrics, artifacts):
    if (metrics.get("global_steps_completed", 0) < 1 or not metrics.get("adapter_parameters_changed")
            or not math.isfinite(metrics["training_loss"])
            or not math.isfinite(metrics["evaluation"]["eval_loss"])):
        raise RuntimeError("Cannot mark an incomplete/unchanged/non-finite training run successful")
    hashes = {name: sha(run_dir / name) for name in
              ("training_config.json", "manifest.json", "metrics.json", "training_args.json")}
    write_json(run_dir / "training_result.json", {
        "status": "success", "model_id": MODEL_ID, "run_id": config["run_id"],
        "finished_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
        "adapter": artifacts, "metadata_sha256": hashes, "smoke": config["smoke"],
    })
    verify_saved(run_dir / "adapter", output)
    if not config["smoke"]:
        pointer = {"model_id": MODEL_ID, "run_id": config["run_id"],
                   "adapter_path": (run_dir / "adapter").relative_to(output).as_posix(),
                   "adapter_sha256": artifacts["sha256"]}
        temporary = output / f".latest-{config['run_id']}.json"
        write_json(temporary, pointer)
        os.replace(temporary, output / "latest_successful.json")


def verify_saved(adapter, output):
    adapter, output = Path(adapter).resolve(), Path(output).resolve()
    if (adapter.name != "adapter" or not (adapter.is_relative_to(output / "smoke/runs")
            or adapter.is_relative_to(output / "runs"))):
        raise ValueError("Unsafe CORD adapter path")
    artifacts = verify_adapter(adapter, MODEL_ID)
    run = adapter.parent
    result = json.loads((run / "training_result.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    config = json.loads((run / "training_config.json").read_text())
    if (result.get("status") != "success" or result.get("model_id") != MODEL_ID
            or result.get("run_id") != run.name or config.get("model_id") != MODEL_ID
            or result.get("adapter", {}).get("sha256") != artifacts["sha256"]
            or result.get("metrics") != metrics or metrics.get("global_steps_completed", 0) < 1
            or not metrics.get("adapter_parameters_changed")
            or metrics.get("initial_adapter_digest") == metrics.get("final_adapter_digest")
            or not math.isfinite(metrics["training_loss"])
            or not math.isfinite(metrics["evaluation"]["eval_loss"])):
        raise RuntimeError("CORD run lacks verified successful optimizer updates")
    for name in ("training_config.json", "manifest.json", "metrics.json", "training_args.json"):
        if result["metadata_sha256"].get(name) != sha(run / name):
            raise RuntimeError("CORD training metadata checksum mismatch")
    if hashlib.sha256(config["instruction"].encode()).hexdigest() != config["instruction_sha256"]:
        raise RuntimeError("Training instruction checksum mismatch")
    return {"adapter": str(adapter), "run_id": run.name, "smoke": config["smoke"], **artifacts}


def run(args):
    output = check_arguments(args)
    if args.verify_latest or args.verify_adapter:
        adapter = latest_adapter(output) if args.verify_latest else project_path(args.verify_adapter)
        verified = verify_saved(adapter, output)
        if args.verify_latest and verified["smoke"]:
            raise RuntimeError("Smoke adapter cannot be the latest training adapter")
        print(json.dumps(verified, indent=2), flush=True)
        return adapter
    libraries = versions()
    print("Versions: " + json.dumps(libraries), flush=True)
    if Path(sys.prefix).name != ".venv" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the existing application/training ../.venv, not .venv-vllm")
    missing = [name for name, version in libraries.items() if version == "MISSING"]
    if missing:
        raise RuntimeError("Missing existing SFT dependencies: " + ", ".join(missing))
    if libraries["trl"] != "0.24.0" or libraries["transformers"] != "4.57.1":
        raise RuntimeError("This entry point uses the existing TRL 0.24.0 / Transformers 4.57.1 SFT APIs; "
                           "inspect the environment instead of upgrading it")
    records, dataset_metadata = load_training_data(project_path(args.dataset_root))
    instruction = (project_path(args.instruction_file).read_text(encoding="utf-8").strip()
                   if args.instruction_file else INSTRUCTION)
    if not instruction or "Item,Quantity,Rate,Amount" not in instruction or len(instruction) > 2000:
        raise ValueError("Instruction must be bounded, nonempty and include the exact CSV header")
    datasets = {split: make_dataset(rows, instruction) for split, rows in records.items()}
    for split, dataset in datasets.items():
        for example, record in zip(dataset, records[split]):
            image = example["images"][0]
            image.load()
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest()
            if pixel_hash != record["pixel_sha256"]:
                raise RuntimeError("HF image paired with incorrect CSV")
            if example["completion"][0]["content"][0]["text"] != record["target_csv"]:
                raise RuntimeError("HF completion changed the canonical CSV")
    smoke = args.smoke or args.max_steps == 1
    steps = 1 if smoke else args.max_steps if args.max_steps > 0 else expected_steps(
        len(records["train"]), args.batch_size, args.grad_accum, args.epochs)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_id=MODEL_ID, versions=libraries, smoke=smoke, expected_optimizer_steps=steps,
                  dataset_root=str(project_path(args.dataset_root).resolve()), output_dir=str(output),
                  instruction=instruction, instruction_version=INSTRUCTION_VERSION if not args.instruction_file else "custom",
                  instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                  target_modules=TARGET_MODULES, quantization="NF4 double quantization", trainable_scope="language attention only",
                  interpreter=sys.executable, environment=sys.prefix, dataset=dataset_metadata)
    print("Configuration: " + json.dumps(config), flush=True)
    if args.preflight:
        print("CPU preflight PASS: 161 train / 20 validation; 19 test artifacts untouched. "
              "Processor token budget, CUDA and forward/backward require the manual GPU smoke.", flush=True)
        return None

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from trl import SFTConfig
    from src.trainer.sft.train.runtime import AuditedSFTTrainer, LiveProgress, log
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One visible BF16-capable CUDA GPU is required (NVIDIA L4; CUDA_VISIBLE_DEVICES=0)")
    free, total = torch.cuda.mem_get_info(0)
    if free / 2**30 < args.min_free_gib:
        raise RuntimeError(f"Only {free/2**30:.2f} GiB free; need {args.min_free_gib}. "
                           "Manually stop BOTH vLLM servers (8000/8002) before training")
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(0)
    config.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  total_gpu_gib=total/2**30, free_gpu_gib=free/2**30, dtype="bfloat16")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output / ("smoke/runs" if smoke else "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config["run_id"] = run_id
    started = time.monotonic()
    try:
        write_json(run_dir / "manifest.json", dataset_metadata)
        write_json(run_dir / "training_config.json", config)
        log(f"Loading original pretrained {MODEL_ID}; no previous SFT/DPO adapter is loaded")
        loading = time.monotonic()
        processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=args.min_pixels,
            max_pixels=args.max_pixels, use_fast=False, trust_remote_code=False,
            local_files_only=args.local_files_only)
        if not processor.chat_template or processor.tokenizer.pad_token_id is None:
            raise RuntimeError("Missing Qwen processor chat template/padding token")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=False, local_files_only=args.local_files_only)
        config["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
        log(f"Processor/model loaded in {time.monotonic()-loading:.1f}s")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        config["resolved_target_modules"] = language_targets(model)
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES, modules_to_save=None, use_dora=False, use_rslora=False))
        initial_digest, trainable = adapter_digest(model)
        config["trainable_parameters"] = trainable
        log(f"Trainable language LoRA parameters={trainable:,}; vision/merger/base weights frozen")
        collator = BoundedCollator(processor, args.max_sequence_length)
        for split, dataset in datasets.items():
            for example in dataset:
                collator([example])
            log(f"Exact image/CSV collator preflight PASS: {split}, {len(dataset)} examples")
        training_args = SFTConfig(output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=args.epochs, max_steps=1 if smoke else args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
            lr_scheduler_type="constant", warmup_steps=0, optim=args.optimizer, bf16=True, fp16=False,
            logging_steps=args.logging_steps, logging_first_step=True, save_strategy="epoch",
            eval_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, prediction_loss_only=True,
            eval_accumulation_steps=1, save_safetensors=True, report_to="none", remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True}, completion_only_loss=True, max_length=None,
            packing=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0, dataloader_pin_memory=False, seed=args.seed, data_seed=args.seed)
        write_json(run_dir / "training_config.json", config)
        write_json(run_dir / "run_config.json", config)  # Existing pointer verifier compatibility.
        write_json(run_dir / "training_args.json", training_args.to_dict())
        trainer = AuditedSFTTrainer(model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets["validation"], processing_class=processor,
            data_collator=collator, callbacks=[LiveProgress()])
        log(f"Training starts: expected {steps} optimizer steps; output {run_dir}")
        training_started = time.monotonic()
        result = trainer.train()
        training_seconds = time.monotonic() - training_started
        if (result.global_step != steps or result.global_step < 1
                or not math.isfinite(float(result.training_loss))):
            raise RuntimeError("Incomplete optimizer steps or non-finite training loss")
        final_digest, _ = adapter_digest(model)
        if final_digest == initial_digest:
            raise RuntimeError("LoRA tensors did not change during optimization")
        evaluation = trainer.evaluate()
        if not math.isfinite(float(evaluation["eval_loss"])):
            raise RuntimeError("Non-finite validation loss")
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        processor.save_pretrained(adapter_dir)
        metrics = {"training": result.metrics, "evaluation": evaluation,
            "training_loss": float(result.training_loss), "global_steps_completed": int(result.global_step),
            "epochs_completed": trainer.state.epoch, "training_duration_seconds": training_seconds,
            "total_duration_seconds": time.monotonic()-started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(0)/2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(0)/2**30,
            "trainable_parameters": trainable, "adapter_parameters_changed": True,
            "initial_adapter_digest": initial_digest, "final_adapter_digest": final_digest,
            "best_validation_checkpoint": trainer.state.best_model_checkpoint}
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "training_log.json", trainer.state.log_history)
        artifacts = verify_adapter(adapter_dir, MODEL_ID)
        publish_success(output, run_dir, config, metrics, artifacts)
        log("Training complete: " + json.dumps(metrics))
        log(f"Verified adapter: {adapter_dir}; weights SHA256={artifacts['sha256']}")
        return adapter_dir
    except BaseException as exc:
        write_json(run_dir / "training_result.json", {"status": "failed", "run_id": run_id,
                   "error_type": type(exc).__name__})
        log(f"Run failed ({type(exc).__name__}); checkpoints retained. Latest successful pointer unchanged")
        raise


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
