"""CPU-only client for our remote vLLM Qwen2.5-VL invoice extraction endpoint.

Phase 5: vLLM runs on a remote Linux GPU; this module runs on the laptop.
It does not import torch, transformers, or model weights. A local SSH tunnel
keeps the backend bound to loopback rather than exposing medical invoices.

Configuration from environment variables (never commit credentials):
  VLM_BASE_URL=http://127.0.0.1:8010/v1
  VLM_MODEL=invoice-extractor
  VLM_API_KEY=<secret>
  VLM_TIMEOUT_SECONDS=180
"""

import base64
import io
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

from src.schemas import InvoiceLine
from src.vlm.evaluate import get_predicted_rows
from src.vlm.training_data import INSTRUCTION


class RemoteVLMError(RuntimeError):
    """Raised when the GPU service or its output is unusable; never silently fabricate CSV."""


@dataclass(frozen=True)
class RemoteVLMConfig:
    """Runtime configuration; HTTPS is mandatory unless server is localhost."""

    base_url: str
    api_key: str
    model: str = "invoice-extractor"
    timeout_seconds: float = 180.0
    max_image_bytes: int = 8 * 1024 * 1024
    max_tokens: int = 512

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("VLM_BASE_URL must not contain credentials or query parameters")
        if parsed.path.rstrip("/") != "/v1":
            raise ValueError("VLM_BASE_URL must end in /v1")
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("VLM_BASE_URL must be an HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote VLM endpoints must use HTTPS; use an SSH tunnel for local HTTP")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("VLM_API_KEY must be set; do not use an unauthenticated model endpoint")
        if not self.model.strip() or self.max_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("Invalid model name, token budget, or timeout")
        if self.max_image_bytes <= 0:
            raise ValueError("Invalid image size limit")

    @classmethod
    def from_env(cls) -> "RemoteVLMConfig":
        """Read settings set in the current shell; .env is optional and not auto-loaded."""
        return cls(
            base_url=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8010/v1").rstrip("/"),
            api_key=os.getenv("VLM_API_KEY", ""),
            model=os.getenv("VLM_MODEL", "invoice-extractor"),
            timeout_seconds=float(os.getenv("VLM_TIMEOUT_SECONDS", "180")),
        )


@dataclass(frozen=True)
class ExtractionResult:
    """Measured real model prediction, before policy retrieval/adjudication."""

    csv_text: str
    rows: list[dict[str, str]]
    model: str
    latency_ms: int
    usage: dict

    def as_dict(self) -> dict:
        return {"csv_text": self.csv_text, "rows": self.rows,
                "model": self.model, "latency_ms": self.latency_ms,
                "usage": self.usage}


class RemoteVLMClient:
    """Send a single invoice image to vLLM's /v1/chat/completions endpoint."""

    def __init__(self, config: RemoteVLMConfig, http_client: httpx.Client | None = None):
        self.config = config
        self._http_client = http_client

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.config.base_url + path
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        client = self._http_client or httpx.Client(
            timeout=httpx.Timeout(self.config.timeout_seconds, connect=10),
            follow_redirects=False,
        )
        try:
            response = client.request(method, url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RemoteVLMError("Model API did not return a JSON object")
            return data
        except httpx.TimeoutException as exc:
            raise RemoteVLMError("GPU inference timed out; check vLLM and try a smaller image") from exc
        except httpx.HTTPStatusError as exc:
            # Don't log response bodies: they may contain sensitive invoice content.
            raise RemoteVLMError(f"GPU API returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RemoteVLMError("Cannot reach GPU server; check SSH tunnel, GPU instance and port") from exc
        except ValueError as exc:
            raise RemoteVLMError("GPU API response is not valid JSON") from exc
        finally:
            if self._http_client is None:
                client.close()

    def available_models(self) -> list[str]:
        """Check API connectivity/authentication without sending an invoice."""
        data = self._request("GET", "/models")
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)]

    def extract(self, image_path: str | Path) -> ExtractionResult:
        """Extract strict image-to-CSV; invalid model outputs raise, not auto-correct."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.config.max_image_bytes:
            raise ValueError("Invoice image exceeds the 8 MB request limit")
        image_bytes = path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                kind = img.format
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Input must be a valid invoice image") from exc
        media_types = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
        if kind not in media_types:
            raise ValueError("Only JPEG, PNG and WebP invoice images are supported")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image_url", "image_url": {"url": f"data:{media_types[kind]};base64,{encoded}"}},
            ]}],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        start = time.perf_counter()
        response = self._request("POST", "/chat/completions", payload)
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        try:
            message = response["choices"][0]["message"]["content"]
            if not isinstance(message, str):
                raise TypeError("Non-text response")
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteVLMError("GPU API response has no assistant text") from exc
        try:
            rows = get_predicted_rows(message)
            # Validate numeric consistency without filling missing rates or guessing data.
            for row in rows:
                InvoiceLine(item=row["Item"], quantity=int(row["Quantity"]),
                            rate=Decimal(row["Rate"]) if row["Rate"] else None,
                            amount=Decimal(row["Amount"]))
        except (ValueError, ArithmeticError) as exc:
            raise RemoteVLMError(f"Model response is not valid invoice CSV: {exc}") from exc
        usage = response.get("usage") or {}
        return ExtractionResult(csv_text=message.strip() + "\n", rows=rows,
                                model=response.get("model", self.config.model),
                                latency_ms=elapsed_ms, usage=usage if isinstance(usage, dict) else {})
