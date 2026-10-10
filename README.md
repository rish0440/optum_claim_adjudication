# Optum Claim Adjudication

A focused FastAPI + LangGraph + fine-tuned VLM + Milvus RAG assignment prototype.

The Phase 3 repository restructuring is complete on Python 3.12. FastAPI
uses the compiled LangGraph workflow for structured invoice adjudication.
All 66 existing CPU tests and eight migration checks pass. Image serving and
GPU training have not been executed during restructuring.

See [the migration report](reports/migration_phase3.md) for the folder tree,
path map, configuration, preserved artifacts, validation and remaining issues.
The numbered phase sections below retain the project's earlier implementation
history; their historical deployment status is superseded by that report.

On this VM, from the Git repository root, reuse the existing parent `.venv`:

```bash
source ../.venv/bin/activate
python -B -m unittest discover -s tests -v
python -B -m scripts.run_workflow --csv data/evaluation/ground_truth/invoice_04_diagnostics.csv --date 2024-04-18
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Canonical modules are `app.main`, `src.trainer.sft`, `src.inference`, and
`src.orchestration`. Old Python import/CLI paths remain compatibility wrappers.
Configuration lives in `src/config/default.yaml`; set `OPTUM_CONFIG_FILE` to
use another YAML file. `OPTUM_PROJECT_ROOT` overrides the root for relative
configured paths. SFT CLI flags continue to accept explicit data/model paths;
`OPTUM_SYNTHETIC_DATASET`, `OPTUM_SFT_OUTPUT_DIR` and `MODEL_ID` provide defaults.
No `.env` file is loaded automatically.

## 1. Quick start (Windows PowerShell)

Use Python 3.11 or 3.12. **No GPU is required for Phase 1–2.**

```powershell
cd optum_claim_adjudication
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.inspect_phase1_phase2
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/**. Check API status at
`http://127.0.0.1:8000/api/dataset-status` and `http://127.0.0.1:8000/api/health`.

macOS/Linux: use `python3 -m venv .venv`, `source .venv/bin/activate` and
`python` thereafter. Run all commands from this repository's root directory.

## 2. Phase 1 — supplied policy and evaluation ground truth

The original `data/policy.json` is validated by `src/adjudication/policy.py` and typed
`src/schemas.py` contracts. `src/config/settings.py` loads settings from
`src/config/default.yaml`. `src/orchestration/state.py` defines a reusable TypedDict
for future LangGraph nodes. No LangGraph execution is needed yet.

The five supplied invoice images are stored **unchanged** under `data/invoices/`.
Corresponding manually transcribed files are under
`data/evaluation/ground_truth/`:

- `invoice_01_standard` — JPG in `data/invoices/`, CSV/JSON in `data/evaluation/ground_truth/`; 4 rows
- `invoice_02_exclusions` — 5 rows
- `invoice_03_icu_mixed` — 5 rows
- `invoice_04_diagnostics` — 6 rows (no visible Rate column; blank Rate cells)
- `invoice_05_nonmedical` — 6 rows
- `manifest.json` and `expected_adjudication.json` are also provided.

**Important:** Total of 26 evaluation line items. The reference adjudication
assumes Surgical Gloves qualify under `COV-019` (Medical Consumables), not an
unambiguous supplied answer key. Invoice dates are within the policy effective
period. The actual JSON policy uses `COV-010` for Blood Tests (`COV-009` is X-Ray).

Recreate the source annotations from the explicitly transcribed values:

```bash
python -m scripts.make_optum_ground_truth
python -m scripts.inspect_phase1_phase2
```

**These five original images and their labels must never be mixed into synthetic
train/validation data, DPO pairs, training prompts, or model-selection runs.**

## 3. Phase 2 — synthetic invoice image/CSV generator

The data generator is **`src/trainer/sft/dataset.py`** and uses Pillow only. It follows:

1. Randomly generate structured line items and prices.
2. Render the same data as fictional hospital invoice JPGs.
3. Save exact matching `Item,Quantity,Rate,Amount` labels as CSV.
4. Write a `target_csv` training record for each image into `annotations.jsonl`.
5. Split without reusing any image across train/validation and verify everything.

Built-in visual variation includes **bordered, semi-bordered, and borderless**
tables, multiple installed font families, different palettes, number of rows,
invoice values, fictitious hospitals, and **missing-rate-column** cases.

### Generate the small, already included smoke-test dataset

The downloaded project already contains 10 reproducibly generated pairs.
To verify without overwriting anything:

```bash
python -m src.trainer.sft.dataset --verify
```

To regenerate those exact examples:

```bash
python -m src.trainer.sft.dataset --count 10 --seed 42 --overwrite
```

Expected: **10 JPGs + 10 CSVs, 8 train / 2 validation, 64 invoice rows**.
Four bordered, three semi-bordered, three borderless; two lack visible rates.
Exact row count applies to seed 42 and may differ for other seeds.

Only after visually inspecting the first ten JPGs and reading sample CSVs,
generate a larger training dataset:

```bash
python -m src.trainer.sft.dataset --count 500 --seed 42 --overwrite
python -m src.trainer.sft.dataset --verify
```

`--overwrite` only replaces the synthetic `train/` and `validation/` outputs.
It **does not touch** `data/invoices/`, `data/evaluation/`, or `data/cord/`.

### Files produced

```text
data/
├── policy.json                        # original, unchanged
├── invoices/                          # five original Optum JPGs, eval ONLY
├── evaluation/ground_truth/           # manually verified CSV/JSON, eval ONLY
├── cord/                              # optional source receipts; not used
└── synthetic/
    ├── dataset_info.json              # seed and split counts
    ├── train/
    │   ├── images/invoice_00001.jpg
    │   ├── labels/invoice_00001.csv
    │   └── annotations.jsonl
    └── validation/
        ├── images/invoice_00004.jpg
        ├── labels/invoice_00004.csv
        └── annotations.jsonl
```

`annotations.jsonl` contains `image`, `csv`, `target_csv`, `layout`,
`rate_column_visible`, `rows`, and `billed_total`. Image/CSV paths are relative
to `data/synthetic/`. This is a simple future VLM fine-tuning input format;
it is **not** a Hugging Face dataset or trained model yet.

## 4. CORD decision — optional and unused

CORD is retail receipt data (many shop/restaurant receipts), unlike an Indian
hospital invoice. If used later, inspect its annotation structure and group
item fields across receipt lines: `menu.nm`, `menu.cnt`, `menu.unitprice`,
`menu.price` can sometimes map to the target CSV columns. Labels are not fully
interchangeable: discounts, grouping, absent prices/rates, and unrelated
receipt categories require conversion and review. For this deadline, leave
`data/cord/` untouched and keep the hospital-image synthetic dataset as the
only VLM supervised training source.

The uploaded .7z project did not contain your downloaded CORD archives; copy
them into `data/cord/` locally only if you want to retain them. They are not
included in the generated project ZIP.

## 5. Tests & evaluation integrity

```bash
python -m unittest discover -s tests -v
python -m scripts.inspect_phase1_phase2
```

The tests check: policy validation, all five held-out labels against source
CSV files, adjudication reference arithmetic, synthetic JPG readability,
CSV/manifest parity, 8/2 split isolation, missing-rate handling and an explicit
SHA256 check that none of the original five JPG bytes appears in synthetic data.

**Do not report extraction accuracy without measured results.** Colab training
succeeded per the user, but no real serving/evaluation output is packaged here. The verified CSV ground truth is a reference
for future extraction evaluation, not evidence of fine-tuning or model results.

## 6. Remaining work (required for complete Optum submission)

**REQUIRED for full submission:** verify remote optimized inference
(vLLM or justified alternative) on a GPU, verify the Phase 6 policy RAG on your laptop,
connect a LangGraph graph, implement accurate deterministic adjudication,
expose a FastAPI upload endpoint and show actual decisions/results in the UI,
run and report held-out evaluation with correct metric denominators, and write
model/run evidence in README.

**OPTIONAL:** CORD receipt conversion, DPO/RLHF preference optimization,
full-scale infrastructure, Kubernetes, cloud databases, elaborate dashboards.

The UI is intentionally not yet connected to a model. Do not confuse saved
reference labels or a mocked API test with end-to-end inference.

## 4. Phase 3 — Qwen2.5-VL-3B QLoRA (Colab Tesla T4)

**Status:** User reports successfully running a 2-step QLoRA smoke test
on a Colab Tesla T4 and saving adapter artifacts in Drive. This repo does not
contain the adapter; actual model quality is not yet established.

### What was implemented

| File | Purpose |
|---|---|
| `requirements-colab.txt` | Separate GPU-only dependencies; do NOT install in Windows local `.venv` |
| `scripts/prepare_colab_dataset.py` | Verify and ZIP synthetic train/validation only, plus training code |
| `src/trainer/sft/training_data.py` | Verify JSONL ↔ JPG ↔ CSV; make VLM prompt/completion samples |
| `src/inference/local.py` | Real Qwen vision inference, optionally with a PEFT adapter |
| `src/trainer/sft/train_lora.py` | 4-bit NF4 + rank-8 LoRA via PEFT and TRL SFTTrainer |
| `src/trainer/sft/evaluate.py` | Base/adapter evaluation with transparent numerator/denominator metrics |
| `notebooks/qwen25_vl_colab.ipynb` | Follow-along Colab notebook, from GPU setup to Drive saving |
| `tests/test_phase3.py` | CPU-only tests for safety, parsing, metrics and bundle contents |

No DPO or CORD training. Phase 5 includes vLLM serving configuration and a
local HTTP client; actual remote GPU inference still needs execution.

### On Windows (keep your existing `.venv`)

Run these commands from the repository root after copying the Phase 3 patch:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.trainer.sft.dataset --verify
python -m unittest discover -s tests -v
python -m scripts.prepare_colab_dataset
```

The last command creates `colab_training_bundle.zip` at the project root.
It contains synthetic data and Phase 3 training files, **not** `data/invoices`,
`data/evaluation`, or `data/cord`.

### On Colab (T4 GPU)

1. Open `notebooks/qwen25_vl_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the notebook cells top to bottom. Upload `colab_training_bundle.zip` in the upload cell.
4. The first model run evaluates the **untrained base** on the *two synthetic validation images*.
5. The training cell attempts **2 optimizer steps** on the 8 synthetic train images.
6. The next cell evaluates the **fine-tuned adapter** on the same two validation images.
7. The final cell saves the adapter and measured reports to Google Drive.

The Colab notebook includes runnable shell commands equivalent to:

```bash
python -m src.trainer.sft.evaluate --dataset-root data/synthetic --split validation \
  --max-samples 2 --output reports/training/baseline.json
python -m src.trainer.sft.train_lora --dataset-root data/synthetic --max-steps 2 \
  --grad-accum 2 --lora-rank 8 --output-dir models/qwen25-vl-invoice-lora
python -m src.trainer.sft.evaluate --dataset-root data/synthetic --split validation \
  --max-samples 2 --adapter models/qwen25-vl-invoice-lora/adapter \
  --output reports/training/finetuned.json
```

Model download and CUDA execution require network and GPU resources.
In case of CUDA OOM reduce `--max-pixels` from `401408` (512×28×28) to
`200704` (256×28×28). Reducing resolution sacrifices small-text legibility.
Do not switch to bf16 on T4. The default training uses fp16 and SDPA.

**After smoke test passes:** regenerate 100–300 synthetic examples locally,
run `--verify`, rebuild the upload ZIP, and retrain for additional optimizer
steps. Larger dataset and longer fine-tuning are not necessary for verifying
the execution pipeline, but only two steps on 8 training images cannot
establish a meaningful performance improvement.

**Model artifacts:** `models/qwen25-vl-invoice-lora/adapter/` and the real
`training_result.json`, `training_log.json`, `reports/training/baseline.json`,
`reports/training/finetuned.json`. No untrained adapter is supplied in the ZIP.

**Scoring:** output contains raw model predictions, parsing errors and strict
numerator/denominator pairs for four-field accuracy by row position, exact-row
precision/recall, valid CSV rate, and fully correct invoices. Results are
always based on real model output, never copied from labels.

**Important:** The Optum five-image dataset remains a final held-out test set
until the system is frozen. It is not used for synthetic generation, training,
model selection, or synthetic validation.


## 5. Phase 5 — Remote GPU model serving (DPO intentionally skipped)

**Code implemented, GPU serving not yet executed in this repository.**

- `deploy/vllm/start_vllm.sh`: runs `vllm serve` privately on Linux GPU, loads
  the trained LoRA adapter as `invoice-extractor`, validates adapter path.
- `deploy/vllm/README.md`: step-by-step GPU setup, SSH tunnel, and security.
- `src/inference/remote_client.py`: local authenticated image-to-CSV client; checks
  image size/type, validates CSV with Pydantic and records round-trip latency.
- `scripts/test_remote_vlm.py`: real `/v1/models` check and synthetic invoice
  extraction from local laptop; no inference is fabricated.
- `tests/test_phase5.py`: HTTP-level mocked unit tests (CPU/no external API).
- `requirements.txt`: adds only `httpx`; no GPU dependencies locally.

### Windows PowerShell: Verify code first (no GPU required)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.test_remote_vlm --help
uvicorn app.main:app --reload
```

### Actual GPU server: mandatory later to close Phase 5

Follow `deploy/vllm/README.md` to copy the real Drive adapter to a Linux GPU,
start vLLM on 127.0.0.1:8001 and create an SSH tunnel from the laptop to
localhost:8010. Then, in PowerShell with tunnel still open:

```powershell
$env:VLM_BASE_URL='http://127.0.0.1:8010/v1'
$env:VLM_MODEL='invoice-extractor'
$env:VLM_API_KEY='your-matching-secret'
python -m scripts.test_remote_vlm --ping
python -m scripts.test_remote_vlm --image data/synthetic/validation/images/invoice_00004.jpg --output reports/generated/vlm_sample.json
```

The second command sends a **synthetic validation** image (not Optum's held-out
images), displays actual CSV/latency and saves the output. If service or adapter
is not available, this step cannot pass yet. This client is not yet wired into
LangGraph or the FastAPI upload endpoint (Phase 8–9).

**Next:** Phase 6 policy RAG, then deterministic adjudication, graph/API wiring,
and final real evaluation.

## Phase 6 — Local Milvus policy RAG (CPU; implemented)

This phase indexes the **45 original rule objects** from `data/policy.json`:
25 coverage and 20 exclusion rules. `policy.json` is never changed. One source
rule = one vector row with exact ID, category, conditions, type, and limit
metadata. `all-MiniLM-L6-v2` produces 384-dimensional embeddings (the index
records and verifies the actual dimensionality). Milvus Lite stores the vectors
in a local database, not in a cloud service.

**Installation on Windows, using your existing `.venv`:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-rag.txt
python -m scripts.build_policy_index
python -m scripts.test_policy_rag --item "Blood Test (CBC)"
python -m scripts.test_policy_rag --item "Personal Care Kit"
python -m scripts.test_policy_rag --evaluate --output reports/generated/policy_rag_metrics.json
python -m unittest discover -s tests -v
```

The first run downloads an embedding model; subsequent runs use its cache.
The first `build_policy_index` run creates `data/policy_index/policy.db` and
`manifest.json`. It can be rerun without recomputing embeddings. When the
policy or embedding model changes, run `python -m scripts.build_policy_index
--rebuild` explicitly. Do not commit the generated `.db` or model cache.

**Which files do what:** `src/rag/indexer.py` creates policy chunks and writes
vectors to Milvus; `src/rag/embeddings.py` embeds text with a CPU model;
`src/rag/retriever.py` performs top-K semantic search;
`src/rag/resolver.py` selects exact categories/approved synonyms and applies
exclusion precedence. When no verified category match exists it returns
`manual_review`; top-K scores are not treated as payout decisions. All selected
rule details are re-read from `policy.json`, not invented from embedding output.
`data/evaluation/policy_rag_cases.json` contains **20 independent text-only
policy queries**. `scripts/test_policy_rag.py --evaluate` reports explicit
numerators and denominators for raw semantic Recall@1, Recall@3, resolved
rule-ID accuracy and policy grounding. Scores must be measured locally:
**do not report arbitrary accuracy**. The five held-out Optum invoice images
are not indexed, read by the RAG indexer, or included in these evaluation cases.

**Important limitations:** A resolver category match is a simple, documented
rule interpretation, not clinical understanding. Surgical Gloves use the
general Medical Consumables rule (`COV-019`) with a review note about the
inpatient-treatment alternative (`COV-024`). Policy dates and claim context
are considered in Phase 7. Phase 6 identifies policy rules **only**; no
reimbursement calculations, agent graph, or live UI connection are implemented.
The optional vLLM server remains on hold.

## Phase 7 — Deterministic Claim Adjudication (implemented)

Phase 7 computes eligibility directly from the source `data/policy.json` and
structured invoice rows, using Python `Decimal` (half-up rounding of INR co-pay).
No LLM, API key, GPU, or vector DB is required to test the calculation engine.
The Phase 6 rule resolver provides policy IDs; optionally retrieve Milvus
candidates with `--with-milvus` (requires the Phase 6 database already built).

### What changed

- `src/adjudication/rules.py`: load extracted `Item,Quantity,Rate,Amount` CSV,
  validate invoice rows, invoke grounded Phase 6 policy rule matching.
- `src/adjudication/calculator.py`: apply exclusions, actual billed amounts,
  per-day/per-visit/per-service/per-session/per-procedure caps (quantity based),
  *shared per-claim* caps (not per row), and 10% co-pay.
- `src/schemas.py`: auditable item/claim outcomes including billed, eligible,
  disallowed, pending-review, reimbursement and source-rule ID.
- `scripts/run_adjudication.py`: offline CLI for JSON, CSV, and five labeled
  reference cases; optional live Milvus candidate retrieval.
- `tests/test_phase7.py`: repeatable unit tests for limits, exclusions, unknown
  categories, dates, missing rates, money rounding and all five sample labels.

### Commands (Windows PowerShell, existing virtual environment)

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v

# One of Optum's manually transcribed JSON records (no image inference):
python -m scripts.run_adjudication --invoice-json data/evaluation/ground_truth/invoice_01_standard.json --output reports/generated/invoice_01_decision.json

# An extracted CSV can be submitted with its date; missing rate stays blank:
python -m scripts.run_adjudication --csv data/evaluation/ground_truth/invoice_04_diagnostics.csv --date 2024-04-18 --output reports/generated/invoice_04_decision.json

# Check all five labeled invoices and actual numerator/denominator metrics:
python -m scripts.run_adjudication --all-reference --output reports/generated/phase7_reference_evaluation.json

# Optional: invoke live Milvus retrieval before the same policy resolution:
python -m scripts.run_adjudication --invoice-json data/evaluation/ground_truth/invoice_01_standard.json --with-milvus
```

The five-invoice CLI checks **oracle CSV/JSON → rule resolution → financial
calculation**. It does **not** represent model extraction quality or an
end-to-end FastAPI/UI run. Those will be evaluated in Phases 8–10.

**Safety:** Missing/unknown policy mappings are `manual_review` and their
billed amounts are kept as *pending*, not denied. An invoice outside the
policy validity period also remains pending. Such claims have
`final_reimbursement: null`; eligible and co-pay totals are provisional.
Surgical Gloves are assigned COV-019 in the supplied reference set; this
mapping carries a review note. Additional medical-necessity verification is
not automated: the assignment's reference annotations assume the billed
covered services satisfy their stated clinical conditions.

**Scope boundary:** This module consumes *validated structured rows*. It does
not yet extract text from an image, execute a LangGraph workflow, or connect
to the browser upload. Phase 5 vLLM remains on hold.


## Phase 8 — LangGraph orchestration (CPU-first)

**What this implements:** one executable LangGraph `StateGraph` connecting
`invoice_extraction → csv_validation → policy_retrieval → rule_resolution →
adjudication → response_builder`. Invalid inputs route directly to the response
builder; they never trigger a financial calculation. Each node contains actual
logic, not a scaffold. This is not a chatbot or a production multi-agent system.

Install only the new workflow dependency in your **existing Windows venv**:

```powershell
python -m pip install -r requirements-workflow.txt
```

### Try the workflow with verified structured inputs

```powershell
python -m scripts.run_workflow --invoice-json data/evaluation/ground_truth/invoice_01_standard.json --output reports/generated/workflow_invoice_01.json
python -m scripts.run_workflow --csv data/evaluation/ground_truth/invoice_04_diagnostics.csv --date 2024-04-18 --output reports/generated/workflow_invoice_04.json
python -m unittest discover -s tests -v
```

Expected from the *reference JSON/CSV* runs: invoice 01 reimbursement INR 8,550,
invoice 04 INR 11,754. These **do not** constitute model-extraction metrics.

### Opt in to live Phase 6 retrieval

If your local Milvus Lite policy database is built and the index is valid:

```powershell
python -m scripts.run_workflow --invoice-json data/evaluation/ground_truth/invoice_01_standard.json --with-milvus
```

The output records `rag_status=milvus_live` only when an actual semantic search
executes successfully. If it fails, it records a warning and falls back to the
exact, conservative `policy.json` resolver by default. To fail rather than
fallback, add `--strict-rag`. Without `--with-milvus`, no vector search is
performed, and the JSON labels `rag_status=policy_only_no_vector_search`.

### Image uploads remain blocked until Phase 5 deployment

`python -m scripts.run_workflow --image path/to/invoice.jpg --date 2024-04-18`
returns a structured error because vLLM is offline; it never pretends to extract
an invoice. After a real authenticated vLLM server is available, add
`--remote-vlm` and set `VLM_BASE_URL`, `VLM_API_KEY`, and `VLM_MODEL` per the
Phase 5 instructions. Inference latency and model name appear as metadata.

### Code organization

- `src/orchestration/state.py`: typed state contract.
- `src/orchestration/graph.py`: actual compiled graph, safe edges.
- `src/orchestration/workflow.py`: one-call facade for FastAPI in Phase 9.
- `src/orchestration/nodes/*/node.py`: separate executable nodes.
- `src/adjudication/{rules,calculator}.py`: reuse and validate Phase 7 rules
  and money; vector similarity never sets financial caps.
- `scripts/run_workflow.py`: local demonstration and JSON export.
- `tests/test_phase8.py`: CPU tests for success, failure, unknown charges, manual
  review, remote inference fakes, and Milvus fallback. Additional actual compiled
  LangGraph tests automatically run if the `langgraph` package is installed.

Do not regenerate Optum ground truth, retrain, or touch your `.venv`/`data/cord`
when applying the Phase 8 patch. The browser `index.html` upload/results integration
is intentionally deferred to Phase 9.

## Phase 9 — FastAPI + browser results UI (CPU-first)

This phase turns the working LangGraph workflow into a web application. The
`index.html` interface is served by FastAPI at `http://127.0.0.1:8000/`.

**Inputs**
- **CSV upload (works locally):** Header must be exactly `Item,Quantity,Rate,Amount`.
  Enter the invoice date shown on the document (YYYY-MM-DD). A blank `Rate`
  is preserved; it is not guessed.
- **JSON upload (works locally):** Existing Phase 1 `InvoiceGroundTruth` JSON
  schema, including its date and line-item values. User-provided JSON is
  explicitly labeled structured input, **not** model extraction or verified truth.
- **Reference demo (works locally):** Select one of the five provided reference
  labels. The response clearly identifies ground-truth/reference testing and
  MUST NOT count toward VLM extraction accuracy.
- **Image upload (requires GPU vLLM):** A real JPEG/PNG is sent to the model
  only when `VLM_BASE_URL` and `VLM_API_KEY` are set and its server is online.
  Otherwise HTTP 503 explains why it cannot proceed. **No fake OCR, no
  substitution of hidden reference labels, no simulated model predictions.**

**Outputs:** Item decisions, source policy rule IDs, notes, partial approvals,
exclusions, pending-review status, co-pay, reimbursement, raw extraction CSV,
model/retrieval provenance, and JSON/CSV downloads. The UI does not write
uploaded files to the project or persist results on the server. Temporary files
for image/JSON modes are deleted after each request. The upload limit is 8 MB.

### Install (reuse the existing Windows .venv)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-workflow.txt
# Only if you want actual Milvus vector retrieval:
# python -m pip install -r requirements-rag.txt
```

### Start the API and browser

```powershell
python -m unittest discover -s tests -v
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Select **Reference demo**, choose
`Standard — room cap`, and click **Run adjudication**. Expected outcome from
*reference structured input*: billed **INR 10,500**, eligible **INR 9,500**,
co-pay **INR 950**, reimbursement **INR 8,550**.

Then test the **CSV / JSON upload** tab with
`data/evaluation/ground_truth/invoice_04_diagnostics.csv`, invoice date
`2024-04-18`. Expected reimbursement: **INR 11,754**. These figures check
policy logic with supplied structured inputs; **they do not verify image
extraction performance**.

### API endpoints

| Endpoint | Input | Purpose |
|---|---|---|
| `GET /` | None | Browser UI |
| `GET /api/health` | None | API/policy and image-service configuration status |
| `GET /api/dataset-status` | None | Policy/dataset stats and integration state |
| `GET /api/examples` | None | Five allowlisted reference samples |
| `POST /api/adjudicate/demo` | JSON `{"example_id":"invoice_01_standard","use_milvus":false}` | Run a reference label through the workflow |
| `POST /api/adjudicate` | Multipart `file`, `invoice_date` (CSV/image), `use_milvus` | Run uploaded data through the workflow |
| `GET /docs` | None | FastAPI Swagger UI |

- `use_milvus=false` means exact policy matching **without vector search**.
- `use_milvus=true` executes Phase 6 retrieval and records `milvus_live`,
  or a visible policy fallback if Milvus cannot run. Both modes use the
  original `policy.json` for actual limits and exclusions.
- Unknown policy matches yield `manual_review` and `final_reimbursement=null`.
- Invalid CSVs return HTTP 422. Image mode without configured serving returns
  HTTP 503. Unavailable model or invalid model response does not fabricate
  line items or reimbursement.

**Files added/modified:** `app/main.py`, `app/templates/index.html`,
`app/static/app.js`, `app/static/styles.css`,
`src/orchestration/nodes/response_builder/node.py`, `requirements.txt`,
`tests/test_phase9.py`, and this README.

**Verification boundary:** Phase 9's API regression tests execute the real
validation/resolution/calculator nodes behind a test-only graph executor,
not a fake payout calculator. The actual compiled LangGraph runtime must
still be tested in your Windows environment with `langgraph` installed; the
current build environment cannot install it. Real remote VLM serving remains
on hold until a Linux GPU endpoint is available.
