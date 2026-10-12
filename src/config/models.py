"""Shared Qwen defaults; importing these values never loads model weights."""
import os

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def sft_output_root():
    """Shared, portable location for SFT runs and inference discovery."""
    from src.utils.paths import configured_path
    return configured_path("OPTUM_SFT_OUTPUT_DIR", "models/qwen25_finetune_sf_qlora")


def sft_adapter_pointer():
    """Inference reads model_id and relative adapter_path from this JSON."""
    return sft_output_root() / "latest_successful.json"
