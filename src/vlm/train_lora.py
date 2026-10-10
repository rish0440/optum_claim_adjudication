"""Actual supervised QLoRA training for Qwen2.5-VL-3B on Google Colab T4.

Run from Colab project root AFTER unpacking the synthetic-only bundle:
    python -m src.vlm.train_lora --dataset-root data/synthetic --max-steps 2

The model is downloaded from Hugging Face. GPU/CUDA and external network are
required. This module is not imported by the local FastAPI application.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .training_data import make_hf_dataset, read_records

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def train(dataset_root: Path, output_dir: Path, *, model_name: str = MODEL_ID,
          max_steps: int = 2, max_pixels: int = 512*28*28, learning_rate: float = 2e-4,
          lora_rank: int = 8, grad_accum: int = 2, seed: int = 42) -> Path:
    """Train LoRA adapters on 4-bit frozen base model; save real artifacts."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Run this module in Google Colab (T4 or better).")
    if max_steps < 1 or grad_accum < 1:
        raise ValueError("max_steps and grad_accum must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    records = read_records(dataset_root, "train")
    validation = read_records(dataset_root, "validation")
    train_names = {r["image_path"].name for r in records}
    if train_names & {r["image_path"].name for r in validation}:
        raise ValueError("Overlapping validation and training data")
    dataset = make_hf_dataset(records)
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28,
                                              max_pixels=max_pixels)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, quantization_config=quantization, device_map={"": 0},
        dtype=torch.float16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=lora_rank, lora_alpha=lora_rank*2, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # Prompt-completion supervision masks user/image tokens; only CSV output is a target.
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, max_length=None, completion_only_loss=True,
    )
    # Fail fast if a package-version change removes the loss or image tensors.
    check = collator([dataset[0]])
    if "pixel_values" not in check or not torch.any(check["labels"] != -100).item():
        raise RuntimeError("Invalid multimodal batch: missing image tensors or CSV loss labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name, "dataset_root": str(dataset_root),
        "train_examples": len(records), "validation_examples": len(validation),
        "max_steps": max_steps, "max_pixels": max_pixels,
        "learning_rate": learning_rate, "lora_rank": lora_rank,
        "gradient_accumulation_steps": grad_accum, "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        fp16=True, bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="no", eval_strategy="no",
        report_to="none", remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None, packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, seed=seed,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=processor, data_collator=collator)
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    config["finished_utc"] = datetime.now(timezone.utc).isoformat()
    config["train_loss"] = float(result.training_loss)
    config["global_steps_completed"] = int(result.global_step)
    config["adapter_dir"] = str(adapter_dir)
    (output_dir / "training_result.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Adapter saved to {adapter_dir} | completed steps: {result.global_step}")
    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen25-vl-invoice-lora"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=512*28*28)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()
    train(args.dataset_root, args.output_dir, model_name=args.model,
          max_steps=args.max_steps, max_pixels=args.max_pixels,
          learning_rate=args.learning_rate, lora_rank=args.lora_rank,
          grad_accum=args.grad_accum)


if __name__ == "__main__":
    main()
