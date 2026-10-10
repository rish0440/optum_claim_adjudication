# Browser claim assessment (Phase 8)

Run from the repository root. Keep the existing adapter-mode vLLM process on
127.0.0.1:8000; do not start a second server or use the serving environment for
FastAPI. No API key is required for this loopback path.

```bash
source ../.venv/bin/activate
cat requirements-web.txt
python -m pip show reportlab python-multipart
python -m pip install --dry-run -r requirements-web.txt
# Only if ReportLab is missing, and after reviewing the dry-run:
python -m pip install -r requirements-web.txt
# Only if the existing multipart dependency is missing:
python -m pip install 'python-multipart>=0.0.9,<1.0'

python -m unittest tests.test_claim_web tests.test_phase9 -v
# Optional existing two-agent regression suite (mocked services):
python -m unittest tests.test_claim_graph tests.test_agent1_integration -v

VLM_BASE_URL=http://127.0.0.1:8000/v1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```

Forward port **8001** in VS Code's Ports panel with private visibility and open
http://127.0.0.1:8001 on your local machine (use VS Code's forwarded URL if the
local port differs). Keep port 8000 private. Upload
`data/demos/claim_2024/invoice_2024.jpg`. No source CSV is supplied by this UI.

The upload API returns 202 and a request ID. The browser polls
`GET /api/claims/{request_id}`. Processing includes one bounded qwen-base image
metadata request followed by the actual compiled Phase 7 graph: invoice-extractor,
Milvus, Agent 1, Decimal calculator and Agent 2. The CLI date contract is unchanged.
The UI reports queued/processing/completed/failed, without simulated node progress.
Missing or ambiguous invoice dates produce a cannot-adjudicate review result;
the graph and PDF generator are not invoked in that case.

Date extraction uses vLLM's JSON-schema response format, with strict client-side
validation, calendar validation and agreement with the returned printed date.
It rejects ambiguous numeric day/month dates. Other unsupported date formats fail
safely. A model transcription is **not independently verified OCR** and can be
wrong; inspect the displayed date against the uploaded image. No date is substituted.
The additional multimodal date call and its JSON-schema support must be verified
manually against the current 2048-token server configuration.

For the reference example, verify billed INR 10,500, eligible INR 8,300,
disallowed INR 1,000, pending INR 1,200 and provisional copay INR 830.
Medicines must require evidence; final reimbursement must read **Not finalized**.
Inspect the verified reasoning, real policy citations and trace in the dashboard.
Download the PDF with the **Download PDF Report** button, or
`GET /api/claims/{request_id}/report.pdf`. Check wrapped item names, amounts,
multi-page tables, policy conditions and the review notice in a PDF viewer.

Artifacts are retained in `reports/claims/{request_id}/`: `report.pdf`,
`final_report.json` (workflow plus extracted-date metadata), and
`validated_invoice.csv`. These private directories are excluded from Git and are
not mounted as static files. Uploaded copies are removed after processing.
The existing inference client may also retain its normal diagnostic artifacts;
its behavior is unchanged. Browser JSON/CSV downloads contain actual results.

Local demo limitations: one Uvicorn worker, one running image claim and at most
three queued claims; 32 retained job records per process. Do not run other model
jobs concurrently. No authentication/multi-user access is provided: use private
loopback forwarding only. Request IDs isolate results but are not user identities.
Jobs/polling records do not survive restarts, and interrupted uploads may remain
on disk. Reports are retained until you manually remove them; there is no automatic
retention scheduler. PDF failure leaves the real claim result available with a
visible warning. No tests, model requests, PDF rendering or browser execution have
been performed during code preparation.
