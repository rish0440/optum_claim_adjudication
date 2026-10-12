"""Explicit, opt-in reasoning routes; never change the invoice endpoint."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from src.utils.paths import ROOT

TEXT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class ReasoningRoute:
    backend: str
    endpoint: str
    model: str
    base_model_id: str
    adapter_path: Path | None = None
    adapter_sha256: str | None = None

    def diagnostics(self):
        # No upload paths, invoice contents, credentials or model responses.
        return {"backend": self.backend, "endpoint": self.endpoint,
                "model": self.model, "base_model_id": self.base_model_id,
                "adapter_sha256": self.adapter_sha256}


def successful_reasoning_adapter():
    """Lightweight identity/checksum check, without importing training/GPU libraries.

    The existing train_dpo --verify-latest remains the deep tensor validator.
    """
    root = (ROOT / "models/qwen_25_finetune_dpo_qlora").resolve()
    try:
        pointer = json.loads((root / "latest_successful.json").read_text())
        adapter = (root / pointer["adapter_path"]).resolve()
        if (not adapter.is_relative_to(root / "runs") or adapter.name != "model_weights"
                or adapter.parent.name != pointer["run_id"]):
            raise ValueError("Unsafe reasoning adapter pointer")
        result = json.loads((adapter.parent / "training_result.json").read_text())
        config = json.loads((adapter / "adapter_config.json").read_text())
        if (pointer.get("model_id") != TEXT_MODEL_ID or result.get("model_id") != TEXT_MODEL_ID
                or result.get("status") != "success" or result.get("run_id") != pointer["run_id"]
                or config.get("base_model_name_or_path") != TEXT_MODEL_ID
                or config.get("peft_type") != "LORA" or config.get("r") != 8):
            raise ValueError("Reasoning adapter identity/configuration mismatch")
        digest = hashlib.sha256()
        with (adapter / "adapter_model.safetensors").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        if checksum != pointer.get("adapter_sha256") or checksum != result.get("adapter", {}).get("adapter_sha256"):
            raise ValueError("Reasoning adapter checksum mismatch")
        return adapter, checksum
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Selected DPO reasoning checkpoint is missing or invalid; verify latest_successful.json") from exc


def reasoning_route(extraction_endpoint="http://127.0.0.1:8000/v1"):
    backend = os.environ.get("OPTUM_REASONING_BACKEND", "phase8")
    if backend not in {"phase8", "text_base", "text_dpo"}:
        raise ValueError("OPTUM_REASONING_BACKEND must be phase8, text_base or text_dpo")
    if backend == "phase8":
        return ReasoningRoute(backend, extraction_endpoint, "qwen-base", "Qwen/Qwen2.5-VL-3B-Instruct")
    parsed = urlsplit(extraction_endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port != 8000 or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("Text reasoning requires the unchanged loopback invoice endpoint on port 8000")
    adapter, checksum = successful_reasoning_adapter() if backend == "text_dpo" else (None, None)
    return ReasoningRoute(backend, "http://127.0.0.1:8002/v1",
                          "reasoning-dpo" if backend == "text_dpo" else "reasoning-base",
                          TEXT_MODEL_ID, adapter, checksum)
