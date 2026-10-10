"""Phase 9: local FastAPI HTTP interface to the existing Phase 8 LangGraph.

POST /api/adjudicate accepts CSV, labeled JSON, or a genuine invoice image.
Image mode requires a configured real remote VLM; there is no fake OCR.
Reference-demo mode is clearly identified as ground-truth/component testing.
No uploaded files or results are retained by this service.
"""

import io
import json
import os
import tempfile
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.policy import load_policy, summarize_policy
from src.settings import get_settings

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # same limit as the remote VLM client
EXAMPLES = {
    "invoice_01_standard": "Standard — room cap",
    "invoice_02_exclusions": "Excluded personal-care items",
    "invoice_03_icu_mixed": "ICU and mixed services",
    "invoice_04_diagnostics": "Diagnostics — rate column missing",
    "invoice_05_nonmedical": "Medical vs nonmedical items",
}

app = FastAPI(title="Optum Claim Adjudication — LangGraph Prototype", version="0.9.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def _run_graph(**kwargs) -> dict:
    """Lazy import keeps unrelated status pages usable if LangGraph is missing."""
    from src.agentic_workflow.workflow import run_claim_workflow
    return run_claim_workflow(**kwargs)


def _validate_date(value: str | None) -> str:
    if not value:
        raise HTTPException(status_code=422, detail="Invoice date (YYYY-MM-DD) is required for CSV or image input")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid invoice date; use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise HTTPException(status_code=422, detail="Invoice date must use YYYY-MM-DD")
    return value


def _ensure_image(data: bytes) -> None:
    """Validate file contents, not just the filename or browser MIME type."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPG and PNG invoices are accepted")
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid JPG/PNG image") from exc


def _deliver(result: dict) -> dict:
    """Treat invalid graph output as an HTTP error but retain useful details."""
    if result.get("status") == "error":
        stage = result.get("error_stage")
        code = 502 if stage == "invoice_extraction" and result.get("input_mode") == "image" else 422
        raise HTTPException(status_code=code, detail={
            "message": "Claim workflow failed", "workflow": result,
        })
    return result


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "templates" / "index.html")


@app.get("/api/health")
def health() -> dict:
    """Readiness metadata, without revealing API keys or customer invoice data."""
    policy = load_policy()
    return {"status": "ok", "stage": "phase_9", "policy_id": policy.policy_id,
            "image_inference_configured": bool(os.getenv("VLM_API_KEY") and os.getenv("VLM_BASE_URL"))}


@app.get("/api/dataset-status")
def dataset_status() -> dict:
    base = get_settings().paths.synthetic_dataset
    info = base / "dataset_info.json"
    return {"policy": summarize_policy(load_policy()),
            "synthetic_dataset": json.loads(info.read_text(encoding="utf-8")) if info.exists() else None,
            "vlm_loaded_locally": False,
            "remote_vlm_serving_status": "configured_not_verified" if os.getenv("VLM_API_KEY") and os.getenv("VLM_BASE_URL") else "not_configured",
            "adjudication_connected": True}


@app.get("/api/examples")
def examples() -> dict:
    """Only allow fixed reference labels; do not expose arbitrary local files."""
    folder = get_settings().paths.evaluation_ground_truth
    return {"examples": [{"id": key, "label": label} for key, label in EXAMPLES.items()
                         if (folder / f"{key}.json").is_file()],
            "notice": "Ground-truth references for component demonstration, not image-model predictions"}


class DemoRequest(BaseModel):
    example_id: str
    use_milvus: bool = False


@app.post("/api/adjudicate/demo")
async def adjudicate_demo(request: DemoRequest) -> dict:
    """Run a fixed reference JSON through all nodes; report its provenance."""
    if request.example_id not in EXAMPLES:
        raise HTTPException(status_code=404, detail="Unknown reference invoice")
    path = get_settings().paths.evaluation_ground_truth / f"{request.example_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference invoice file is missing")
    result = await run_in_threadpool(_run_graph, invoice_json=path,
                                     with_milvus=request.use_milvus)
    return _deliver(result)


@app.post("/api/adjudicate")
async def adjudicate(
    file: UploadFile = File(...),
    invoice_date: str | None = Form(None),
    use_milvus: bool = Form(False),
) -> dict:
    """Run a submitted CSV/JSON/image through the real LangGraph workflow."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".json", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=415, detail="Upload a .csv, .json, .jpg, .jpeg or .png file")
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum file size is 8 MB")

    if extension == ".csv":
        when = _validate_date(invoice_date)
        try:
            csv_text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded") from exc
        result = await run_in_threadpool(_run_graph, csv_text=csv_text,
                                         invoice_date=when, invoice_id=Path(filename).stem,
                                         with_milvus=use_milvus)
    elif extension == ".json":
        # Existing graph accepts a local JSON path; use a short-lived sandboxed copy.
        with tempfile.TemporaryDirectory(prefix="optum_claim_") as tmp:
            path = Path(tmp) / "uploaded.json"
            path.write_bytes(data)
            result = await run_in_threadpool(_run_graph, invoice_json=path,
                                             with_milvus=use_milvus)
        # A user-uploaded JSON is structured input, not necessarily verified ground truth.
        # The Phase 8 graph uses the ground-truth label for every JSON path, so
        # correct that provenance explicitly for this public upload endpoint.
        if result.get("status") != "error":
            result["extraction_source"] = "provided_structured_json"
            result["warnings"] = [w for w in result.get("warnings", [])
                                  if not w.startswith("Ground-truth JSON used")]
            result["warnings"].append(
                "User-supplied JSON used as structured input; no VLM extraction was performed"
            )
    else:
        when = _validate_date(invoice_date)
        _ensure_image(data)
        if not os.getenv("VLM_BASE_URL") or not os.getenv("VLM_API_KEY"):
            raise HTTPException(status_code=503, detail=(
                "Image extraction is unavailable: start your real GPU vLLM service "
                "and configure VLM_BASE_URL and VLM_API_KEY. "
                "Use CSV or reference demo mode meanwhile."
            ))
        with tempfile.TemporaryDirectory(prefix="optum_image_") as tmp:
            path = Path(tmp) / ("invoice" + extension)
            path.write_bytes(data)
            result = await run_in_threadpool(_run_graph, image=path,
                                             invoice_date=when,
                                             invoice_id=Path(filename).stem,
                                             with_milvus=use_milvus, remote_vlm=True)
    return _deliver(result)
