"""Multimodal safety checks and live training progress (loaded only for training)."""
import copy
import math
import subprocess
import time

import torch
from transformers import TrainerCallback
from trl import SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


def log(message):
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


class CheckedInvoiceCollator:
    """Reuse TRL's Qwen chat/vision processing and audit every batch's CSV loss."""

    def __init__(self, processor):
        self.processor = processor
        self.delegate = DataCollatorForVisionLanguageModeling(
            processor=processor, max_length=None, completion_only_loss=True,
        )
        vocab = processor.tokenizer.get_vocab()
        self.image_tokens = [vocab[token] for token in
                             ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>") if token in vocab]
        if len(self.image_tokens) != 3:
            raise RuntimeError("Expected Qwen vision special tokens are missing")

    def __call__(self, examples):
        if not examples:
            raise ValueError("Empty invoice training batch")
        for example in examples:
            if len(example.get("images", [])) != 1 or example["images"][0] is None:
                raise ValueError("Each invoice example must contain exactly one image")
            if not example.get("prompt") or not example.get("completion"):
                raise ValueError("Invoice prompt/completion messages are missing")
        # TRL mutates message content while preparing images; retain the source dataset.
        batch = self.delegate(copy.deepcopy(examples))
        if any(key not in batch or batch[key].numel() == 0 for key in
               ("pixel_values", "image_grid_thw", "input_ids", "attention_mask", "labels")):
            raise RuntimeError("Invalid multimodal batch: missing pixels/grid/tokens/labels")
        if len(batch["image_grid_thw"]) != len(examples):
            raise RuntimeError("Image grid count does not match invoice batch size")
        if not torch.isfinite(batch["pixel_values"]).all().item():
            raise RuntimeError("Non-finite image tensors")
        for index, example in enumerate(examples):
            labels = batch["labels"][index]
            ids = batch["input_ids"][index]
            supervised = labels != -100
            if not supervised.any().item() or not torch.equal(labels[supervised], ids[supervised]):
                raise RuntimeError("CSV completion loss labels are empty or mismatched")
            if (labels[batch["attention_mask"][index] == 0] != -100).any().item():
                raise RuntimeError("Padding tokens must not contribute to loss")
            positions = torch.nonzero(supervised).flatten()
            if positions[0].item() == 0 or not torch.all(labels[:positions[0]] == -100).item():
                raise RuntimeError("Prompt tokens must be masked from completion loss")
            for token in self.image_tokens:
                if not torch.all(labels[ids == token] == -100).item():
                    raise RuntimeError("Vision tokens must be masked from loss")
            if not (ids == self.image_tokens[0]).any().item():
                raise RuntimeError("Image token is absent; text-only training is forbidden")
            expected = example["completion"][0]["content"][0]["text"]
            actual = self.processor.tokenizer.decode(labels[supervised].tolist(), skip_special_tokens=True)
            if actual.strip() != expected.strip():
                raise RuntimeError("Supervised completion tokens do not decode to the exact target CSV")
        return batch


class LiveProgress(TrainerCallback):
    def __init__(self, query_utilization=False):
        self.started = time.monotonic()
        self.query_utilization = query_utilization

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs and not math.isfinite(float(logs["loss"])):
            raise RuntimeError("Training loss became non-finite")
        utilization = ""
        if self.query_utilization:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1, check=True,
                )
                utilization = " GPU utilization [index,%]=" + result.stdout.strip().replace("\n", "; ")
            except (OSError, subprocess.SubprocessError):
                utilization = " GPU utilization=unavailable"
        log(f"step={state.global_step}/{state.max_steps} epoch={state.epoch or 0:.3f} "
            f"loss={logs.get('loss', logs.get('eval_loss', 'n/a'))} "
            f"lr={logs.get('learning_rate', 'n/a')} elapsed={time.monotonic()-self.started:.1f}s "
            f"VRAM allocated={torch.cuda.memory_allocated()/2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB{utilization}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps:
            log(f"optimizer step={state.global_step}/{state.max_steps} epoch={state.epoch or 0:.3f} "
                f"elapsed={time.monotonic()-self.started:.1f}s")

    def on_save(self, args, state, control, **kwargs):
        log(f"Periodic checkpoint saved: {args.output_dir}/checkpoint-{state.global_step}")


class AuditedSFTTrainer(SFTTrainer):
    """Check that Trainer really calls the vision tower and backpropagates to LoRA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_calls = 0
        base_model = self.model.get_base_model()
        self._vision_audit_hook = base_model.visual.register_forward_hook(self._vision_used)

    def _vision_used(self, module, inputs, output):
        self.vision_calls += 1

    def training_step(self, model, inputs, *args, **kwargs):
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Trainer dropped the multimodal forward inputs")
        previous = self.vision_calls
        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.vision_calls <= previous:
            raise RuntimeError("The invoice image did not pass through the vision tower")
        gradients = [parameter.grad for name, parameter in model.named_parameters()
                     if parameter.requires_grad and ".lora_" in name and parameter.grad is not None]
        if not gradients:
            raise RuntimeError("LoRA backward gradients are missing")
        # FP16's GradScaler checks scaled overflows during the optimizer update.
        if not self.args.fp16 and any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError("LoRA backward gradients are non-finite")
        if not getattr(self, "_audit_logged", False):
            log("First training batch PASS: vision forward executed and LoRA backward gradients present")
            self._audit_logged = True
        return loss
