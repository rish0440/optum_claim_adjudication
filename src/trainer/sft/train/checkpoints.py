"""Verify genuine PEFT artifacts and resolve the latest successful run on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

from src.trainer.sft.config import output_dir
from src.utils.paths import project_path

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def verify_adapter(adapter: Path, expected_model: str | None = None) -> dict:
    import torch
    from safetensors import safe_open

    config_path = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    for path in (config_path, weights, adapter / "preprocessor_config.json",
                 adapter / "tokenizer_config.json"):
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing/empty saved artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("peft_type") != "LORA" or set(config.get("target_modules", [])) != set(TARGET_MODULES)
            or config.get("modules_to_save") or config.get("bias") != "none"):
        raise RuntimeError("Expected a language-attention-only PEFT LoRA adapter")
    if expected_model and config.get("base_model_name_or_path") != expected_model:
        raise RuntimeError("Saved adapter base model differs from the training model")
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise RuntimeError("Invalid saved LoRA rank")
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or not all(".language_model." in key and ".lora_" in key for key in keys):
            raise RuntimeError("Adapter tensors must exclusively target language-model LoRA weights")
        for key in keys:
            shape = handle.get_slice(key).get_shape()
            if len(shape) != 2 or min(shape) < 1:
                raise RuntimeError("Invalid LoRA tensor shape")
            if not torch.isfinite(handle.get_tensor(key)).all().item():
                raise RuntimeError("Adapter tensors contain non-finite values")
            if ".lora_A." in key:
                partner = key.replace(".lora_A.", ".lora_B.")
                if shape[0] != rank or partner not in keys or handle.get_slice(partner).get_shape()[1] != rank:
                    raise RuntimeError("LoRA A/B pair or rank is inconsistent")
            elif ".lora_B." not in key:
                raise RuntimeError("Unexpected adapter tensor name")
            elif key.replace(".lora_B.", ".lora_A.") not in keys or shape[1] != rank:
                raise RuntimeError("LoRA B tensor is missing its matching A tensor or has an invalid rank")
        if not any("lora_A" in key for key in keys) or not any("lora_B" in key for key in keys):
            raise RuntimeError("LoRA A/B tensors are missing")
    return {"tensor_count": len(keys), "weights_bytes": weights.stat().st_size,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "artifacts": sorted(p.name for p in adapter.iterdir() if p.is_file())}


def latest_adapter(root: Path) -> Path:
    root = root.resolve()
    pointer = json.loads((root / "latest_successful.json").read_text(encoding="utf-8"))
    adapter = (root / pointer["adapter_path"]).resolve()
    if not adapter.is_relative_to(root / "runs") or adapter.name != "adapter":
        raise RuntimeError("Unsafe adapter path in latest-successful pointer")
    result = json.loads((adapter.parent / "training_result.json").read_text(encoding="utf-8"))
    if (result.get("status") != "success" or result.get("model_id") != pointer["model_id"]
            or result.get("run_id") != pointer["run_id"]
            or result.get("metrics", {}).get("global_steps_completed", 0) < 1):
        raise RuntimeError("The pointed run is not successful")
    for name in ("run_config.json", "training_args.json", "metrics.json", "training_log.json"):
        json.loads((adapter.parent / name).read_text(encoding="utf-8"))
    checked = verify_adapter(adapter, pointer["model_id"])
    if checked["sha256"] != pointer["adapter_sha256"]:
        raise RuntimeError("Saved adapter checksum differs from the successful-run pointer")
    return adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=output_dir())
    args = parser.parse_args()
    root = project_path(args.output_dir)
    adapter = latest_adapter(root)
    print(json.dumps({"adapter": str(adapter), **verify_adapter(adapter)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
