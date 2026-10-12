"""Load verified structured inputs or call the real remote VLM, if enabled.

Loading a reference JSON is for component testing only and is never presented
as a prediction from an invoice image. No GPU service means image mode fails.
"""
import csv
import io
import json
from pathlib import Path

from src.schema import InvoiceGroundTruth
from src.components.agentic_workflow.state import ClaimState


def _as_csv(invoice: InvoiceGroundTruth) -> str:
    """Convert trusted JSON rows to canonical CSV without inferring missing rates."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Item", "Quantity", "Rate", "Amount"])
    for line in invoice.lines:
        writer.writerow([line.item, line.quantity,
                         str(line.rate) if line.rate is not None else "",
                         str(line.amount)])
    return output.getvalue()


def extract_invoice(state: ClaimState, *, remote_client=None) -> dict:
    """Select exactly one input source; return a partial graph state update."""
    mode = state.get("input_mode")
    try:
        if mode == "invoice_json":
            path = Path(state["invoice_json_path"])
            invoice = InvoiceGroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
            return {"invoice": invoice, "extracted_csv": _as_csv(invoice),
                    "extraction_source": "verified_structured_json",
                    "trace": [*state.get("trace", []), "invoice_extraction"]}
        if mode == "csv_file":
            text = Path(state["csv_path"]).read_text(encoding="utf-8-sig")
            return {"extracted_csv": text, "extraction_source": "provided_csv",
                    "trace": [*state.get("trace", []), "invoice_extraction"]}
        if mode == "csv_text":
            return {"extracted_csv": state["csv_text"], "extraction_source": "provided_csv",
                    "trace": [*state.get("trace", []), "invoice_extraction"]}
        if mode == "image":
            if not state.get("use_remote_vlm", False):
                raise RuntimeError("Image inference unavailable: vLLM is on hold. Use --remote-vlm only when the GPU API is running.")
            if remote_client is None:
                from src.components.inference.remote_client import RemoteVLMClient, RemoteVLMConfig
                remote_client = RemoteVLMClient(RemoteVLMConfig.from_env())
            # Only errors from remote execution are retry candidates; bad CSV
            # is explicitly validated by the next node.
            from src.components.inference.remote_client import RemoteVLMError
            import time
            for attempt in range(2):
                try:
                    extracted = remote_client.extract(state["image_path"])
                    break
                except RemoteVLMError:
                    if attempt == 1:
                        raise
                    time.sleep(0.3)
            return {"extracted_csv": extracted.csv_text,
                    "extraction_source": "remote_vlm",
                    "metadata": {**state.get("metadata", {}),
                                 "vlm_model": extracted.model,
                                 "vlm_latency_ms": extracted.latency_ms},
                    "trace": [*state.get("trace", []), "invoice_extraction"]}
        raise ValueError("Invalid input mode: supply exactly one of --invoice-json, --csv, --csv-text or --image")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"errors": [*state.get("errors", []), str(exc)],
                "error_stage": "invoice_extraction",
                "trace": [*state.get("trace", []), "invoice_extraction"]}
