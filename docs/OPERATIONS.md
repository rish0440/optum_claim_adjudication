# Manual verification and demo operations

None of these commands was executed during cleanup. Run from the repository
root. Do not recreate environments or install from every requirements file.

## CPU imports and dependencies

```bash
source ../.venv/bin/activate
python -m pip check
python -m pip show fastapi uvicorn pydantic Pillow httpx python-multipart reportlab langgraph pymilvus sentence-transformers
python -B -c 'import app.main, src.agents.contracts, src.orchestration.graph, src.rag.retriever, src.adjudication.calculator; print("Application imports OK")'
$HOME/.venv-vllm/bin/python -m pip check
$HOME/.venv-vllm/bin/python -m pip show vllm torch torchvision transformers
```

Review missing/import errors before installing anything. `requirements.txt`,
`requirements-workflow.txt`, `requirements-rag.txt`, `requirements-web.txt` define
runtime dependencies; SFT/DPO pins apply only to `../.venv`. Serving vLLM 0.11.2
has its own Torch/Transformers stack. Never bulk-upgrade either environment.

## CPU regressions (mocked external services)

```bash
OPTUM_REASONING_BACKEND=phase8 python -B -m unittest tests.test_reasoning_routing tests.test_claim_graph tests.test_agent1_integration tests.test_claim_web tests.test_phase7 tests.test_phase9 tests.test_cord_evaluation tests.test_optum_extraction -v
OPTUM_REASONING_BACKEND=phase8 python -B -m unittest discover -s tests -v
```

Some existing suites require local original invoice fixtures. Preserve them on
this VM; an image-free public checkout must restore authorized copies before
running those suites. Historical preference provenance checks may report source
drift after Phase 9D; do not rewrite labels/manifests to conceal it.

## Verify all three successful adapters (CPU; no model load)

```bash
python -u -m src.trainer.sft.train.checkpoints --output-dir models/qwen25_vl_sft
python -u -m src.trainer.preference_optimization.train.train_dpo --verify-latest
python -u -m src.trainer.cord.train --verify-latest
OPTUM_REASONING_BACKEND=text_dpo python -m scripts.check_reasoning_route
sha256sum models/qwen25_vl_sft/runs/20261010T174916Z-0b733db4/adapter/adapter_model.safetensors models/qwen25_3b_reasoning_dpo/runs/20261011T074847Z-60ab777d/adapter/adapter_model.safetensors models/qwen25_vl_cord_sft/runs/20261011T134348Z-f1e33111/adapter/adapter_model.safetensors
bash -n deploy/vllm/start_reasoning.sh deploy/vllm/start_vlm.sh
```

Compare hashes with `models/README.md` and each latest pointer. Shell syntax and
CPU checks do not prove GPU kernels/adapter loading. Do not modify successful
metadata: original/CORD pointer checks require selected `training_log.json`.

## Three terminals and health

Use the three exact launcher commands in the root README. The new reasoning
script is a permanent equivalent of `/tmp/optum-agent2-eval-server-40.sh`, which
is left intact. Check both server terminals and run:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8002/health
curl --fail http://127.0.0.1:8002/v1/models
OPTUM_REASONING_BACKEND=text_dpo python -m scripts.check_reasoning_route --check-server
watch -n 1 nvidia-smi
```

Verify invoice-extractor on 8000, reasoning-base and reasoning-dpo on 8002. The
model-card/checksum route check must identify the selected text adapter. Keep
ports private. Do not run GPU evaluation while either server occupies the GPU.

If the Milvus database is missing in a fresh checkout, acquire/cache the declared
embedding model and build the index separately, before claims:

```bash
python -m scripts.build_policy_index
```

An incompatible existing index needs explicit review before a manual `--rebuild`;
do not overwrite it automatically. Preserve `data/policy.json`.

## End-to-end image and PDF

```bash
OPTUM_REASONING_BACKEND=text_dpo python -u -m scripts.run_claim_graph --image data/demos/claim_2024/invoice_2024.jpg --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/final_demo --verbose
```

Exit 2 is REVIEW_REQUIRED, not a financial approval; exit 1 is failure. Inspect
the printed request directory's `validated_invoice.csv`, `policy_evidence.json`,
`agent_1.json`, `deterministic_claim.json`, `agent_2.json`, `trace.json` and
`final_report.json`. Both agents must report reasoning-dpo and genuine requests.
The synthetic reference is expected to have billed INR 10500, eligible 8300,
disallowed 1000, pending 1200 and provisional copay 830. Medicines evidence is
unverified, so final reimbursement stays null. These are inspection expectations,
never live substitute outputs. Image extraction differences must be investigated.

Open the forwarded browser at http://127.0.0.1:8001, upload the same image, inspect
the model-extracted date and Download PDF Report. Use its actual request ID:

```bash
curl --fail http://127.0.0.1:8001/api/claims/REQUEST_ID
curl --fail http://127.0.0.1:8001/api/claims/REQUEST_ID/report.pdf --output /tmp/claim-assessment.pdf
```

Compare the PDF with the actual JSON/UI: item decisions, real rule IDs, pending
evidence, copay and NOT FINALIZED reimbursement. A downloaded PDF alone is not a
correctness check. Local reports stay under `reports/claims/<request_id>/`.

## Rollback without changing weights or saved pointers

Stop only the relevant foreground process with Ctrl-C; never launch two servers
on the same port. For the original extractor, restart terminal 1 with:

```bash
OPTUM_SFT_OUTPUT_DIR=models/qwen25_vl_sft VLM_PYTHON="$HOME/.venv-vllm/bin/python" bash deploy/vllm/start_vlm.sh adapter --gpu-memory-utilization 0.40
```

Return to the previous reasoning startup by using the untouched temporary script
instead of `deploy/vllm/start_reasoning.sh`. To compare text-base reasoning,
restart FastAPI with `OPTUM_REASONING_BACKEND=text_base`; for the historical
vision-base agent route use `phase8`. Do not change environment variables during
an in-flight claim. The final setup in README remains text_dpo + CORD SFT.
