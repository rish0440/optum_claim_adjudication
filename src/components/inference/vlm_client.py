"""Real loopback vLLM image-to-CSV inference; reuse the existing HTTP transport."""
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import re
import time
from uuid import uuid4

from src.components.inference.parsing import CSV_FIELDS, clean_csv_text, get_predicted_rows
from src.components.inference.remote_client import RemoteVLMClient, RemoteVLMConfig, RemoteVLMError, image_data_url
from src.components.prompts.extraction import INSTRUCTION
from src.schema import InvoiceLine
from src.utils.paths import project_path

BASE_SERVED_MODEL = "qwen-base"
ADAPTER_SERVED_MODEL = "invoice-extractor"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def select_model(mode, model=None):
    if mode not in {"base", "adapter"}:
        raise ValueError("mode must be base or adapter")
    selected = model or (BASE_SERVED_MODEL if mode == "base" else ADAPTER_SERVED_MODEL)
    if not selected.strip() or (mode == "adapter" and selected in {BASE_SERVED_MODEL, BASE_MODEL_ID}):
        raise ValueError("Adapter mode requires a distinct served adapter name; base fallback is forbidden")
    if mode == "base" and selected == ADAPTER_SERVED_MODEL:
        raise ValueError("Base mode cannot select the adapter served name")
    return selected


def validate_csv(text):
    """Validate actual generated CSV without filling rates or manufacturing rows."""
    cleaned = clean_csv_text(text)
    try:
        reader = csv.reader(io.StringIO(cleaned), strict=True)
        if next(reader, None) != list(CSV_FIELDS):
            raise ValueError("CSV header must be exactly Item,Quantity,Rate,Amount")
        if any(len(row) != 4 for row in reader):
            raise ValueError("CSV rows must contain exactly four fields")
        rows = get_predicted_rows(cleaned)
        for index, row in enumerate(rows, 1):
            if not re.fullmatch(r"[1-9][0-9]*", row["Quantity"]):
                raise ValueError(f"Row {index}: Quantity must be a positive integer")
            for field in ("Rate", "Amount"):
                value = row[field]
                if field == "Rate" and value == "":
                    continue
                if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or not Decimal(value).is_finite():
                    raise ValueError(f"Row {index}: {field} must be a finite nonnegative decimal")
            try:
                InvoiceLine(item=row["Item"], quantity=int(row["Quantity"]),
                            rate=Decimal(row["Rate"]) if row["Rate"] else None, amount=Decimal(row["Amount"]))
            except ValueError as exc:
                raise ValueError(f"Row {index}: invalid item or Quantity × Rate differs from Amount") from exc
    except (csv.Error, AttributeError, ArithmeticError) as exc:
        raise ValueError("Malformed CSV or invalid numerical value") from exc
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), rows


@dataclass(frozen=True)
class VLMConfig:
    mode: str
    model: str | None = None
    endpoint: str = "http://127.0.0.1:8000/v1"
    timeout: float = 180
    max_tokens: int = 1024
    temperature: float = 0
    top_p: float = 1
    seed: int = 42
    api_key: str = ""
    max_image_bytes: int = 8 * 1024 * 1024
    max_input_pixels: int = 20_000_000

    def __post_init__(self):
        select_model(self.mode, self.model)
        self.transport_config()
        if (not math.isfinite(self.temperature) or self.temperature < 0
                or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1
                or not math.isfinite(self.timeout) or self.max_input_pixels < 1 or self.seed < 0):
            raise ValueError("Invalid generation parameters or image pixel limit")

    def transport_config(self):
        return RemoteVLMConfig(
            base_url=self.endpoint.rstrip("/"), api_key=self.api_key,
            model=select_model(self.mode, self.model), timeout_seconds=self.timeout,
            max_image_bytes=self.max_image_bytes, max_tokens=self.max_tokens, require_api_key=False,
        )


class VLMClient(RemoteVLMClient):
    def __init__(self, config: VLMConfig, http_client=None):
        self.options = config
        super().__init__(config.transport_config(), http_client)

    def build_request(self, image_path):
        path = Path(image_path)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Input must be a local JPG or PNG invoice")
        url = image_data_url(path, self.options.max_image_bytes, formats=("JPEG", "PNG"),
                             max_pixels=self.options.max_input_pixels)
        expected_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        if not url.startswith(f"data:{expected_type};"):
            raise ValueError("Image file extension does not match its actual format")
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": INSTRUCTION},
            ]}],
            "temperature": self.options.temperature, "top_p": self.options.top_p,
            "seed": self.options.seed, "max_tokens": self.options.max_tokens,
        }

    def extract(self, image_path, output_dir=None):
        payload = self.build_request(image_path)
        parent = project_path(output_dir or f"reports/inference/{self.options.mode}")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        artifacts = parent / run_id
        artifacts.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        report = {"status": "FAIL", "mode": self.options.mode, "requested_model": self.config.model}
        try:
            cards = self._request("GET", "/models").get("data", [])
            if not isinstance(cards, list):
                raise RemoteVLMError("Malformed /v1/models response")
            card = next((entry for entry in cards if isinstance(entry, dict)
                         and entry.get("id") == self.config.model), None)
            if card is None:
                raise RemoteVLMError("Requested model is absent from /v1/models; no base fallback permitted")
            if self.options.mode == "adapter" and (not card.get("parent") or not card.get("root")
                                                      or card["root"] == BASE_MODEL_ID):
                raise RemoteVLMError("Requested model is not advertised as a loaded LoRA adapter")
            if self.options.mode == "base" and card.get("parent"):
                raise RemoteVLMError("Base mode unexpectedly selected an adapter")
            base_card = card if self.options.mode == "base" else next(
                (entry for entry in cards if isinstance(entry, dict) and entry.get("id") == card.get("parent")), None)
            if base_card is None or base_card.get("root") != BASE_MODEL_ID:
                raise RemoteVLMError("Served base-model metadata does not match Qwen/Qwen2.5-VL-3B-Instruct")
            report["server_model"] = {key: card.get(key) for key in ("id", "root", "parent")}
            response = self._request("POST", "/chat/completions", payload)
            (artifacts / "response.json").write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
            try:
                choice = response["choices"][0]
                raw = choice["message"]["content"]
                if not isinstance(raw, str):
                    raise TypeError
            except (KeyError, IndexError, TypeError) as exc:
                raise RemoteVLMError("Model response has no assistant text; response.json retained") from exc
            (artifacts / "raw_output.txt").write_text(raw, encoding="utf-8")
            if response.get("model") != self.config.model:
                raise RemoteVLMError("Response model differs from requested model; refusing possible base fallback")
            if choice.get("finish_reason") != "stop":
                raise RemoteVLMError("Generation did not finish normally; increase token budget if truncated")
            csv_text, rows = validate_csv(raw)
            (artifacts / "extracted.csv").write_text(csv_text, encoding="utf-8")
            report.update(status="PASS", returned_model=response["model"], rows=len(rows),
                          latency_ms=round((time.perf_counter()-started)*1000), usage=response.get("usage", {}))
            return {"csv_text": csv_text, "rows": rows, "artifacts": str(artifacts), **report}
        except (RemoteVLMError, ValueError) as exc:
            report["error"] = str(exc)
            raise RemoteVLMError(f"{exc}; diagnostic artifacts: {artifacts}") from exc
        finally:
            (artifacts / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
