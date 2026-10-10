"""Conservative single-L4 QLoRA defaults and configurable data/output locations."""
from src.config.models import MODEL_ID, sft_output_root
from src.utils.paths import configured_path

MAX_STEPS = 8
LEARNING_RATE = 2e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GRAD_ACCUM = 2
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
SEED = 42


def dataset_root():
    return configured_path("OPTUM_SYNTHETIC_DATASET", "data/synthetic")


def output_dir():
    return sft_output_root()
