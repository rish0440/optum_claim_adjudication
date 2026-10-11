# Phase 9D: explicit experimental reasoning routing

No deployment files or saved environment defaults are changed. The existing
two vLLM servers stay running with the user's verified 0.40 memory allocation,
2048 context and one sequence each. No additional model/server is launched.

`OPTUM_REASONING_BACKEND` selects one route for **both** real reasoning agents:

| Value | Endpoint | Served model | Backbone |
| --- | --- | --- | --- |
| unset / `phase8` (default) | existing invoice endpoint, normally 8000 | qwen-base | Qwen2.5-VL-3B-Instruct |
| `text_base` | 127.0.0.1:8002/v1 | reasoning-base | Qwen2.5-3B-Instruct |
| `text_dpo` | 127.0.0.1:8002/v1 | reasoning-dpo | same text base + DPO adapter |

The switch is read when a reasoning client is created. Restart FastAPI after
changing it; do not change it during in-flight jobs. Keep `VLM_BASE_URL` on
8000. Text modes reject an invoice endpoint on another port. Extraction and
browser invoice-date metadata extraction remain on 8000, using invoice-extractor
and the existing vision qwen-base respectively. No automatic fallback exists.
The older `/api/adjudicate` API is preserved; the switch applies to the two-agent
claim graph and the Phase 8 `/api/claims/analyze` browser workflow.

## Manual checks (none executed by Codex)

Run from the repository:

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate
OPTUM_REASONING_BACKEND=phase8 python -m unittest tests.test_reasoning_routing tests.test_claim_graph tests.test_claim_web -v
OPTUM_REASONING_BACKEND=phase8 python -m unittest tests.test_agent2_structured_evaluation.StructuredAgent2Tests.test_new_manual_review_fixture_is_deterministic_and_outside_datasets tests.test_agent2_structured_evaluation.StructuredAgent2Tests.test_manual_review_missing_concern_and_unsupported_approval_rejected -v
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8002/health
curl --fail http://127.0.0.1:8002/v1/models
OPTUM_REASONING_BACKEND=text_dpo python -m scripts.check_reasoning_route
OPTUM_REASONING_BACKEND=text_dpo python -m scripts.check_reasoning_route --check-server
```

The first route check is CPU/local-file only. The second checks HTTP model
identity without generation. Neither verifies GPU kernels or model quality.
DPO routing verifies the repository-relative successful pointer, run identity,
base ID, rank 8 and weights checksum. Server model cards must identify that
exact adapter directory and the unadapted reasoning-base parent. The existing
deep tensor verification can also be run manually:

```bash
python -m src.trainer.preference_optimization.train.train_dpo --verify-latest
```

Every agent request checks model cards, exact response model, unchanged JSON
schema, actual chat token budget, completion status and canonical scoped
references. Request/model/endpoint metadata and the graph execution trace
contain routing information without invoice contents. Requested-route trace
entries alone do not prove a successful call; inspect agent_1/agent_2 metadata
and reviewed/status as well. Excess context or truncated output fails safely.

## Genuine Agent 2 medical-review smoke

Reuse the NEW standalone COV-005 Medicines case dated 2024-06-17, outside all
preference splits. Medical relationship is unknown; pending INR 1475.25 remains
pending. The expected references are stored in audit output, never supplied as
generation answers or forced through schema enums. Both model modes use the
same strict FactSelections and canonical/runtime validation.

```bash
python -u -m src.trainer.preference_optimization.eval.structured_agent2 --manual-review-smoke --mode both --check
python -u -m src.trainer.preference_optimization.eval.structured_agent2 --manual-review-smoke --mode both --endpoint http://127.0.0.1:8002 --output-dir reports/dpo/phase9d_manual_review
```

Inspect the printed unique output directory: raw generations, expected scoped
references, semantic failures and runtime-validator acceptance. This exercises
real Agent 2 requests; full graph commands below exercise both agents separately.

## Identical reference CSV comparison and full image pipeline

```bash
OPTUM_REASONING_BACKEND=text_base python -u -m scripts.run_claim_graph --csv data/evaluation/ground_truth/invoice_01_standard.csv --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/phase9d_text_base --verbose
OPTUM_REASONING_BACKEND=text_dpo python -u -m scripts.run_claim_graph --csv data/evaluation/ground_truth/invoice_01_standard.csv --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/phase9d_text_dpo --verbose
OPTUM_REASONING_BACKEND=phase8 python -u -m scripts.run_claim_graph --csv data/evaluation/ground_truth/invoice_01_standard.csv --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/phase9d_phase8 --verbose
OPTUM_REASONING_BACKEND=text_dpo python -u -m scripts.run_claim_graph --image data/demos/claim_2024/invoice_2024.jpg --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/phase9d_image --verbose
```

The image command supplies **only the image**, not its source CSV. Its date is
the actual printed demo date. CLI exit 2 means REVIEW_REQUIRED, not success or
final approval; exit 1 means FAILED. Do not mask these exit codes.

Each unique graph directory contains final_report.json, agent_1.json,
deterministic_claim.json, agent_2.json, policy_evidence.json, trace.json and
validated_invoice.csv when validation succeeded. Check metadata.agent_1.model
and metadata.agent_2.model, the selected rules and warnings. For the verified
reference, expect billed 10500, eligible 8300, disallowed 1000, pending 1200,
copay 830, final reimbursement null, and Medicines needing medical evidence.
These values are inspection expectations, never injected into the live graph.

Compare the exact deterministic artifacts using the directories printed by
the two CSV runs (replace RUN_ID with the actual directory name):

```bash
diff -u reports/claim_graph/phase9d_text_base/RUN_ID/deterministic_claim.json reports/claim_graph/phase9d_text_dpo/RUN_ID/deterministic_claim.json
```

Also inspect both final_report.json files: both must contain a non-null claim,
and any changed status, rule, pending amount, warnings or rejection must be
investigated. Two null results or identical failures are not evidence of
financial agreement. Do not filter out differences or claim DPO improvement.
Only text_base versus text_dpo isolates adapter effects; phase8 is a regression
comparison using a different backbone.

## Browser and PDF

Keep both existing GPU servers running; start **one** application worker:

```bash
OPTUM_REASONING_BACKEND=text_dpo VLM_BASE_URL=http://127.0.0.1:8000/v1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```

Forward port 8001 through VS Code and open http://127.0.0.1:8001. Upload
`data/demos/claim_2024/invoice_2024.jpg`, wait for the genuine result, verify the
model-extracted 2024-01-15 date, item decisions/citations, Medicines concern,
reviewed status and the totals above. Download PDF using the existing button.
It must show NOT FINALIZED rather than INR 0 for null reimbursement, matching
the structured claim result. Compare its item amounts and review findings with
the displayed result; don't infer correctness from a successful download.

Reports remain under `reports/claims/<request_id>/`. Poll and download manually:

```bash
curl --fail http://127.0.0.1:8001/api/claims/REQUEST_ID
curl --fail http://127.0.0.1:8001/api/claims/REQUEST_ID/report.pdf --output /tmp/phase9d-claim.pdf
watch -n 1 nvidia-smi
```

Observe both server terminals for inference/memory errors and the application
terminal for safe failures. A failed Agent 2 review retains verified calculator
results, explicitly unreviewed; it does not replace numbers or approve a claim.

Stop **only FastAPI** with Ctrl-C to switch modes, then start it with
`OPTUM_REASONING_BACKEND=text_base`, or return to the proven default:

```bash
unset OPTUM_REASONING_BACKEND
VLM_BASE_URL=http://127.0.0.1:8000/v1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```

## Limitations

- No new dependencies. No tests, installations, services or inference were run
  during preparation. All live behavior and PDF results await manual checks.
- DPO has not demonstrated higher generated-answer accuracy. The integration
  is explicitly experimental; small synthetic checks do not prove generalization.
- Both servers coexist in the user-verified GPU configuration; context and
  inference peaks still require monitoring. Do not train while serving.
- Historical Phase 9A/9C source-fingerprint checks include reasoning_client.py
  and copilot.py. They will report provenance drift after integration. Do not
  edit datasets/manifests or bypass those guards. Reproduce historical dataset
  evaluations from their original revision; the standalone smoke does not use
  preference splits and remains available here.
- Existing in-process browser jobs/retention limitations remain unchanged;
  requests do not survive application restarts.
