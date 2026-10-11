"""Conservative single-L4 defaults, isolated from the extraction model."""
from dataclasses import dataclass
import math
from src.utils.paths import project_path
from src.config.models import sft_output_root

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_ROOT = "models/qwen25_3b_reasoning_dpo"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
PINNED = {"trl": "0.24.0", "transformers": "4.57.1", "peft": "0.17.1",
          "accelerate": "1.12.0", "datasets": "4.3.0", "bitsandbytes": "0.48.1",
          "safetensors": "0.8.0"}


@dataclass
class TrainingConfig:
    model: str = MODEL_ID
    dataset_dir: str = "data/dpo"
    output_dir: str = OUTPUT_ROOT
    learning_rate: float = 1e-5
    beta: float = .1
    max_steps: int = 20
    batch_size: int = 1
    grad_accum: int = 4
    rank: int = 8
    alpha: int = 16
    dropout: float = .05
    max_length: int = 2048
    eval_steps: int = 5
    seed: int = 42
    optimizer: str = "paged_adamw_8bit"
    precompute_reference: bool = False
    local_files_only: bool = False

    def validate(self):
        if self.model != MODEL_ID:
            raise ValueError("Phase 9B requires the NEW Qwen2.5-3B text backbone")
        for name in ("max_steps", "batch_size", "grad_accum", "rank", "alpha", "max_length", "eval_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.seed < 0 or self.max_length > 2048 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid seed/dropout or sequence length above 2048")
        if not all(math.isfinite(v) and v > 0 for v in (self.learning_rate, self.beta)):
            raise ValueError("Learning rate and beta must be positive and finite")
        if self.optimizer not in {"paged_adamw_8bit", "adamw_torch"}:
            raise ValueError("Unsupported optimizer")
        safe_output_root(self.output_dir)
        return self


def safe_output_root(value):
    root = project_path(value).resolve()
    protected = [sft_output_root().resolve(), project_path("models/qwen25_vl_sft").resolve(),
                 project_path("models/qwen25-vl-invoice-lora").resolve()]
    if any(root == path or root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("DPO output must be separate from extraction checkpoints and their parent directories")
    return root


def language_targets(module_names):
    selected = [name for name in module_names if name.rsplit(".", 1)[-1] in TARGETS]
    if (not selected or {name.rsplit(".", 1)[-1] for name in selected} != set(TARGETS)
            or any(not name.startswith("model.layers.") or ".self_attn." not in name for name in selected)):
        raise ValueError("Expected Qwen2 text-model attention projections only")
    return selected


def dpo_arguments(config, run_dir, bf16):
    config.validate()
    interval = min(config.eval_steps, config.max_steps)
    return dict(output_dir=str(run_dir/"checkpoints"), max_steps=config.max_steps,
        learning_rate=config.learning_rate, beta=config.beta, loss_type="sigmoid",
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.grad_accum, bf16=bf16, fp16=not bf16,
        optim=config.optimizer, lr_scheduler_type="constant", warmup_steps=0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None, max_prompt_length=None, max_completion_length=None,
        # Length is audited before construction; NEVER truncate policy/JSON data.
        padding_free=False, use_logits_to_keep=True, reference_free=False,
        precompute_ref_log_probs=config.precompute_reference, precompute_ref_batch_size=1,
        ref_adapter_name=None, sync_ref_model=False, disable_dropout=True,
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_safetensors=True, save_only_model=True, save_total_limit=2,
        logging_steps=1, logging_first_step=True, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False, dataset_num_proc=None,
        seed=config.seed, data_seed=config.seed, generate_during_eval=False)
