# Manual verification and demo operations

None of these commands was executed during cleanup. Run from the repository
root. Reuse the existing environments and review any dependency installation.

## CPU imports and dependencies

```bash
source ../.venv/bin/activate
python -m pip check
python -m pip show fastapi uvicorn pydantic Pillow httpx python-multipart reportlab langgraph pymilvus sentence-transformers
python -B -c 'import UI.main, src.components.agentic_workflow.contracts, src.components.agentic_workflow.graph, src.components.rag.retriever, src.utils.calculator; print("Application imports OK")'
$HOME/.venv-vllm/bin/python -m pip check
$HOME/.venv-vllm/bin/python -m pip show vllm torch torchvision transformers
```

Review missing/import errors before installing anything. `requirements.txt`
contains application, workflow, retrieval and training dependencies, including
GPU packages. Preview changes with `python -m pip install --dry-run -r requirements.txt`
when reusing `../.venv`. Serving vLLM 0.11.2 has its own Torch/Transformers stack
and `src/components/deploy/vllm/constraints.txt`. Never bulk-upgrade either environment.

## Essential CPU regressions (mocked external services)

```bash
OPTUM_REASONING_BACKEND=phase8 python -B -m unittest discover -s tests -v
```

The retained suites cover the demo workflow, financial calculations, retrieval,
web reports, inference and training safeguards. See [testing guide](TESTING.md).
Some suites require local original invoice fixtures. Preserve them on
this VM; an image-free public checkout must restore authorized copies before
running those suites. Historical preference provenance checks may report source
drift after Phase 9D; do not rewrite labels/manifests to conceal it.

## Verify both successful adapters (CPU; no model load)

```bash
python -u -m trainer.dpo_qlora_policy_reasoning.train --verify-latest
python -u -m trainer.sft_qlora_invoice_extraction.train --verify-latest
OPTUM_REASONING_BACKEND=text_dpo python -m scripts.check_reasoning_route
sha256sum models/qwen_25_finetune_dpo_qlora/runs/20261011T074847Z-60ab777d/model_weights/adapter_model.safetensors models/qwen25_finetune_sf_qlora/runs/20261011T134348Z-f1e33111/model_weights/adapter_model.safetensors
bash -n src/components/deploy/vllm/start_reasoning.sh src/components/deploy/vllm/start_vlm.sh
```

Compare hashes with `docs/models_README.md` and each latest pointer. Shell syntax and
CPU checks do not prove GPU kernels/adapter loading. Do not modify successful
metadata: extraction pointer checks require selected `training_log.json`.

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
do not overwrite it automatically. Preserve `data/assignment_dataset/policy.json`.

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

## Switch reasoning backends

Stop the relevant foreground process before restarting; keep one server per port.
The extraction server uses the retained SFT model selected in README.

Return to the previous reasoning startup by using the untouched temporary script
instead of `src/components/deploy/vllm/start_reasoning.sh`. To compare text-base reasoning,
restart FastAPI with `OPTUM_REASONING_BACKEND=text_base`; for the historical
vision-base agent route use `phase8`. Do not change environment variables during
an in-flight claim. The final setup in README remains text_dpo + CORD SFT.
