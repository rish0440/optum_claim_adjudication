"""Image-only API; bounded, single-worker jobs for the existing claim graph."""
import io
import json
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from src.utils.paths import project_path
from src.inference.remote_client import RemoteVLMError
from app.invoice_date import extract_invoice_date
from app.claim_report import generate_claim_pdf

router = APIRouter(prefix="/api/claims", tags=["Image claims"])
MAX_BYTES, MAX_PIXELS = 8*1024*1024, 20_000_000
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claim")
slots = threading.BoundedSemaphore(4)  # one running, at most three waiting
lock = threading.Lock()
jobs = {}  # bounded to 32 records per demo session; restart clears them
logger = logging.getLogger(__name__)


def validate_upload(data, filename, content_type):
    suffix = Path(filename or "").suffix.lower()
    expected = "PNG" if suffix == ".png" else "JPEG"
    if suffix not in {".jpg", ".jpeg", ".png"} or content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Upload only a JPG, JPEG or PNG invoice image")
    if not data:
        raise HTTPException(422, "Uploaded image is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Maximum file size is 8 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width*image.height > MAX_PIXELS:
                raise HTTPException(413, "Maximum decoded image size is 20 megapixels")
            if image.format != expected or content_type != ("image/png" if expected == "PNG" else "image/jpeg"):
                raise HTTPException(415, "Image extension, MIME type and actual contents must match")
            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(422, "Upload a single-page image")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Invalid or unsafe invoice image") from exc
    return suffix


def _endpoint():
    endpoint = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    if urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Browser claims require a loopback vLLM endpoint")
    return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"


def _graph(**kwargs):
    from src.orchestration.workflow import run_claim_graph
    return run_claim_graph(**kwargs)


def _update(request_id, **values):
    with lock:
        jobs[request_id].update(values)


def _process(request_id, image, directory):
    try:
        _update(request_id, state="processing", message="Resolving invoice date and running the claim workflow")
        metadata = extract_invoice_date(image, endpoint=_endpoint())
        _update(request_id, invoice_date=metadata)
        if metadata["status"] != "FOUND":
            _update(request_id, state="review_required", message="Invoice date is missing or ambiguous. Claim cannot be adjudicated; no date was substituted.")
            return
        report = _graph(image=image, invoice_date=metadata["invoice_date"], request_id=request_id,
                        endpoint=_endpoint(), timeout=120, request_timeout=360, max_tokens=768)
        from src.agents.contracts import WorkflowReport
        report = WorkflowReport.model_validate(report).model_dump(mode="json")
        if report["request_id"] != request_id:
            raise ValueError("Graph request ID mismatch")
        (directory / "final_report.json").write_text(json.dumps({"invoice_date": metadata, "workflow": report}, indent=2), encoding="utf-8")
        (directory / "validated_invoice.csv").write_text(report["extracted_csv"], encoding="utf-8")
        _update(request_id, result=report)
        try:
            generate_claim_pdf(report, metadata, directory / "report.pdf")
            _update(request_id, report_available=True)
        except Exception as exc:
            # Graph results remain available if the optional PDF dependency fails.
            logger.warning("Claim PDF unavailable request=%s category=%s", request_id, type(exc).__name__)
            _update(request_id, report_error="PDF generation unavailable. Check ReportLab installation and server diagnostics.")
        _update(request_id, state="failed" if report["status"] == "FAILED" else "completed",
                message="Claim workflow finished" if report["status"] != "FAILED" else "Claim workflow failed; no approval was issued")
    except RemoteVLMError:
        _update(request_id, state="failed", message="Local GPU service unavailable or returned invalid metadata. Check the vLLM server; no claim was approved.")
    except ValueError:
        _update(request_id, state="review_required", message="Invoice date or workflow output is unsupported. No substitute date or approval was used.")
    except Exception as exc:
        logger.warning("Claim processing failed request=%s category=%s", request_id, type(exc).__name__)
        _update(request_id, state="failed", message="Claim processing failed. Check local policy index and service configuration; no claim was approved.")
    finally:
        try:
            image.unlink(missing_ok=True)  # only this job's private uploaded copy
        except OSError:
            logger.warning("Private upload cleanup failed request=%s", request_id)
        finally:
            slots.release()


@router.post("/analyze", status_code=202)
async def analyze(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_BYTES+1)
        suffix = validate_upload(data, file.filename, file.content_type)
    finally:
        await file.close()
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "Claim queue is full; try again after the current claim finishes")
    request_id = uuid4().hex
    directory = project_path(os.getenv("OPTUM_CLAIM_REPORT_DIR", "reports/claims")) / request_id
    image = None
    try:
        with lock:
            if len(jobs) >= 32:
                raise HTTPException(429, "Demo session limit reached. Retain reports and restart the single-worker app")
            jobs[request_id] = {"request_id": request_id, "state": "queued", "message": "Waiting for claim processing",
                                "invoice_date": None, "result": None, "report_available": False}
        directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        image = directory / ("upload" + suffix)
        image.write_bytes(data)
        image.chmod(0o600)
        executor.submit(_process, request_id, image, directory)
    except Exception:
        slots.release()
        if image is not None:
            image.unlink(missing_ok=True)
        with lock:
            jobs.pop(request_id, None)
        raise
    return {"request_id": request_id, "state": "queued", "status_url": f"/api/claims/{request_id}"}


def _job(request_id):
    with lock:
        if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id) or request_id not in jobs:
            raise HTTPException(404, "Claim not found in this server session")
        return json.loads(json.dumps(jobs[request_id]))


@router.get("/{request_id}")
def status(request_id: str):
    return _job(request_id)


@router.get("/{request_id}/report.pdf")
def download_report(request_id: str):
    job = _job(request_id)
    if not job["report_available"]:
        raise HTTPException(404, "PDF is not available for this claim")
    path = project_path(os.getenv("OPTUM_CLAIM_REPORT_DIR", "reports/claims")) / request_id / "report.pdf"
    if not path.is_file():
        raise HTTPException(404, "PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"claim-{request_id}.pdf")
