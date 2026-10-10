# Phase 3 repository restructuring

Completed on `feature/end-to-end-integration`, based on commit
`201408b7976681882ecc23c8ff58c571bcf62479`. The branch and clean working tree
were verified before edits. Changes remain uncommitted; nothing was pushed.

The Git root is
`/home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication`.
The existing Python 3.12 environment remains at `../.venv` relative to it.
No environment was recreated and no packages were installed in this phase.

## Final organization

Package directories include `__init__.py` files.

```text
app/
  main.py                         FastAPI entry point and existing routes
  schemas.py                      DemoRequest HTTP contract
  templates/index.html
  static/{app.js,styles.css}
src/
  schemas.py                      Shared invoice/policy/claim domain contracts
  trainer/
    sft/{dataset.py,training_data.py,config.py,train_lora.py,evaluate.py}
    preference_optimization/__init__.py  Reserved; no DPO implementation
  inference/{local.py,remote_client.py,parsing.py}
  agents/__init__.py               Reserved; no new agent behavior
  orchestration/
    graph.py
    state.py
    workflow.py
    nodes/
      invoice_extraction/node.py
      csv_validation/node.py
      policy_retrieval/node.py
      rule_resolution/node.py
      adjudication/node.py
      response_builder/node.py
  rag/{embeddings.py,indexer.py,retriever.py,resolver.py}
  adjudication/{policy.py,rules.py,calculator.py}
  prompts/extraction.py
  config/{default.yaml,settings.py,models.py}
  utils/paths.py
  main.py,settings.py,policy.py     Legacy compatibility wrappers
  vlm/                             Legacy compatibility wrappers
  agentic_workflow/                 Legacy compatibility wrappers
scripts/                           Existing CLIs retained
tests/                             Seven original modules plus test_migration.py
data/                              Existing policy, invoices, labels and index manifest
models/                            Existing model directory retained
adapter/                           Existing artifact directory retained in place
deploy/vllm/                       Existing launcher and serving documentation
reports/                           Existing results plus this migration report
notebooks/qwen25_vl_colab.ipynb     Existing notebook with canonical module commands
requirements*.txt                  All four files unchanged
colab_training_bundle.zip          Historical artifact retained unchanged
run_config.json,training_result.json,training_log.json  Retained unchanged
```

## Migration map

Paths below are relative to the Git root. Old executable Python modules remain
as compatibility wrappers; each has one canonical implementation.

| Old path/resource | Canonical path |
|---|---|
| `src/main.py` | `app/main.py` |
| `src/main.py:DemoRequest` | `app/schemas.py` |
| `src/templates/index.html` | `app/templates/index.html` |
| `src/static/app.js` | `app/static/app.js` |
| `src/static/styles.css` | `app/static/styles.css` |
| `src/settings.py` | `src/config/settings.py` |
| `src/config.yaml` | `src/config/default.yaml` |
| Repository-root path utility | `src/utils/paths.py` |
| `src/policy.py` | `src/adjudication/policy.py` |
| `src/vlm/dataset.py` | `src/trainer/sft/dataset.py` |
| `src/vlm/training_data.py` | `src/trainer/sft/training_data.py` |
| `src/vlm/train_lora.py` | `src/trainer/sft/train_lora.py` |
| Existing SFT CLI defaults | `src/trainer/sft/config.py` |
| `src/vlm/evaluate.py` | `src/trainer/sft/evaluate.py` |
| `src/vlm/inference.py` | `src/inference/local.py` |
| `src/vlm/remote_client.py` | `src/inference/remote_client.py` |
| CSV fields/read_rows from training_data; cleaning/parsing from evaluate | `src/inference/parsing.py` |
| Extraction INSTRUCTION from training_data | `src/prompts/extraction.py` |
| Shared Qwen ID and pixel limits | `src/config/models.py` |
| `src/agentic_workflow/graph.py` | `src/orchestration/graph.py` |
| `src/agentic_workflow/state.py` | `src/orchestration/state.py` |
| `src/agentic_workflow/workflow.py` | `src/orchestration/workflow.py` |
| `src/agentic_workflow/nodes/invoice_extraction/node.py` | `src/orchestration/nodes/invoice_extraction/node.py` |
| `src/agentic_workflow/nodes/csv_validation/node.py` | `src/orchestration/nodes/csv_validation/node.py` |
| `src/agentic_workflow/nodes/policy_retrieval/node.py` | `src/orchestration/nodes/policy_retrieval/node.py` |
| `src/agentic_workflow/nodes/rule_resolution/node.py` | `src/orchestration/nodes/rule_resolution/node.py` |
| `src/agentic_workflow/nodes/adjudication/node.py` | `src/orchestration/nodes/adjudication/node.py` |
| `src/agentic_workflow/nodes/response_builder/node.py` | `src/orchestration/nodes/response_builder/node.py` |

Existing graph/node package initializers also moved, with legacy package markers
retained. `scripts/`, `tests/`, `src/rag/`, `src/adjudication/`, `src/schemas.py`,
data, model artifacts and reports keep their responsibilities and locations.
The current phase's target places SFT evaluation under the trainer and processing
nodes under orchestration, superseding those destinations in the earlier audit.

## Compatibility and configuration

- Active imports, test patch targets, scripts, notebook commands and README paths
  use the canonical packages. Legacy modules alias the actual module object,
  preserving shared settings caches, class identity and old monkeypatch targets.
  Legacy `python -m src.vlm.*` commands forward to the canonical `main()`.
- FastAPI endpoints, route definitions and UI remain the same. `DemoRequest` is
  the existing request model moved to `app/schemas.py`; shared domain schemas stay
  in `src/schemas.py`. Assets resolve relative to `app/main.py`.
- HTTP inference imports only shared prompts, parsing and domain validation;
  it no longer imports the training/evaluation implementation. Torch remains a
  lazy import in local inference/training execution.
- Application policy/data/Milvus paths are configurable in
  `src/config/default.yaml`. `OPTUM_CONFIG_FILE` selects another YAML file.
  `OPTUM_PROJECT_ROOT` changes the base of relative configured paths. Explicit
  CLI file arguments retain their normal shell-relative semantics.
- `OPTUM_SYNTHETIC_DATASET` overrides the synthetic location for both application
  settings and SFT CLI defaults. SFT `--dataset-root`, `--output-dir`, `--model`,
  evaluation `--adapter` and local inference `--adapter` remain available.
  `OPTUM_SFT_OUTPUT_DIR` and `MODEL_ID` provide configurable defaults. The existing
  synthetic-only loader still requires a dataset directory named `synthetic`.
- `src/trainer/sft/config.py` extracts the original training defaults without
  changing optimizer, precision, loss masking, rank, seed or training logic.
- The bundle builder now includes canonical modules and their transitive shared
  files plus old command wrappers. It accepts `--dataset-root`, and bundles only
  the selected synthetic pairs at portable `data/synthetic/` paths. Temporary
  test bundles were verified in isolation with Python `-S`. The checked-in ZIP
  was preserved, so it still represents the historical pre-migration code.
- Remote serving environment variables and the existing deployment script are
  untouched. `.env` remains untouched and is not loaded automatically.

## Validation and preservation

| Check | Result |
|---|---|
| Pre-migration suite | 66 passed, 0 failures/errors/skips |
| Post-migration suite | 74 passed, 0 failures/errors/skips |
| Original tests | All 66 preserved and passing |
| Migration regressions | All 8 passing |
| Syntax | 96 Python files compiled in memory |
| Imports | All project source, application and script modules imported |
| FastAPI integration | Five reference demos and reference CSV through the real compiled graph |
| Reference CSV | Invoice 04 reimbursement `11754.00`; missing rates preserved |
| Reference calculations | Invoice totals 5/5; rule IDs, statuses and eligible amounts each 26/26 |
| LangGraph | Compiles and executes; bad input short-circuits as before |
| Financial/training behavior | Core function bodies compared structurally against Git HEAD |
| Protected artifacts | 80 existing tracked files compared byte-for-byte against Git HEAD |
| Browser assets | All three moved assets byte-identical |
| Whitespace | `git diff --check` passed |

Core structural comparisons include `train`, `make_hf_dataset`, `read_records`,
`generate_dataset`, `evaluate_pairs` and `adjudicate_invoice`. Financial/RAG
modules changed only their imports. The Hugging Face representation retains
`images: [path]` cast through `Sequence(Image())` and the original multimodal
prompt/completion layout.

Protected files include existing data, reports, model/adapter artifacts,
training metadata, deployment files, requirements, `.gitignore`, `.env.example`
and the historical ZIP. No sensitive file was moved or rewritten. Existing
tests render temporary images and temporary ZIPs only; repository datasets were
not regenerated. No training, GPU serving, model downloads, agent implementation,
commits or pushes were performed.

## Unresolved issues and phase boundary

No migration regressions or missing-dependency test skips were observed.
The prior environment issue with missing `torchvision` for `qwen_vl_utils`
remains outside this phase; `torchvision` and `vllm` are still absent from this
environment. Imports of project modules succeed because GPU dependencies are
lazy. GPU training and model loading were not executed.

The audit's pre-existing incomplete adapter weights, absent persisted Milvus
database, historical stale training ZIP, limited ignore coverage and difference
between CSV syntax scoring and operational arithmetic validation remain.
Live Milvus search and vLLM inference are not established by CPU/mock tests.
Financial rules and metric definitions were deliberately preserved.

Phase 4 (ten synthetic invoices) is safe to begin after approval. Ten examples
already exist and were preserved; decide explicitly whether to inspect/reuse
them or regenerate. No generation or overwrite was performed in this phase.

## Manual verification

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate
git branch --show-current
git status --short
git diff --check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m unittest discover -s tests -v
python -B -c 'from app.main import app; from src.orchestration.graph import create_graph; create_graph(); print("FastAPI and LangGraph OK")'
python -B -m scripts.run_workflow --csv data/evaluation/ground_truth/invoice_04_diagnostics.csv --date 2024-04-18
python -B -m scripts.run_adjudication --all-reference
python -B -m src.trainer.sft.dataset --verify
python -B -m src.vlm.dataset --verify
python -B -m src.trainer.sft.train_lora --help
python -B -m src.trainer.sft.evaluate --help
python -B -m src.inference.local --help
```

`--verify` checks the existing synthetic files without generating invoices.
`--help` does not load models. To manually serve the existing UI separately:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This starts FastAPI only; it does not start vLLM.
