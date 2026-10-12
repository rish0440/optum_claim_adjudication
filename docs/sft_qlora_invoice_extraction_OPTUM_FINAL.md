# Optum invoice extraction comparison

Reuse Phase 10C's evaluator, shared frozen NF4/BF16 Qwen-VL base, verified
adapter pointers/tensors, deterministic decoding, general CORD training prompt,
image limits, Decimal scoring and one-to-one row alignment. No training,
production routing, launchers, weights or labels are changed. Only --dataset
optum changes the input loader/report directory. Both evaluation modes receive identical
conditions. No DPO model is loaded for evaluation.

The originals are data/assignment_dataset/invoices/invoice_01_standard.jpg through the five named
entries in data/assignment_dataset/ground_truth/manifest.json. The existing evaluation
documentation records manually transcribed, human-reviewed labels. Preflight verifies exact
names, image contents, CSV/JSON item equality, quantities, visible rates and
amounts; it records image/label/provenance hashes. It does not perform a new human
visual verification. Invoice 04 has no Rate column: empty rates must remain empty,
and invented rates are penalized. Missing CSV or JSON marks accuracy NOT ASSESSED;
inconsistent existing labels fail preflight instead of being repaired.

Reports reuse the Phase 10C files plus comparison.csv and a conservative demo
recommendation. All five labels are required for a recommendation. Any measured
regression in CSV/document/row or field accuracy retains ORIGINAL SFT; improvements
with no regression recommend NEW CORD SFT; ties or insufficient labels are
INCONCLUSIVE. Latency alone does not choose an adapter. Selection applies only to
this small healthcare demo set, not a generalization claim. No promotion occurs.

```bash
source ../.venv/bin/activate
python -B -m unittest discover -s tests -v
python -u -m trainer.sft_qlora_invoice_extraction.evaluate --dataset optum --preflight --local-files-only

# Manually stop both vLLM servers before the five-image GPU comparison.
CUDA_VISIBLE_DEVICES=0 python -u -m trainer.sft_qlora_invoice_extraction.evaluate --dataset optum --local-files-only --max-new-tokens 1024

# Use the actual printed run directory; original reports are never overwritten.
python -u -m trainer.sft_qlora_invoice_extraction.evaluate --inspect reports/cord/optum_final/RUN_ID
```

## Manual end-to-end check only after reviewing the recommendation

The extraction launcher uses the retained SFT model. Default evaluation now compares pretrained base versus this SFT model; old original-versus-CORD results are historical. In the extraction-server terminal:

```bash
OPTUM_SFT_OUTPUT_DIR=models/qwen25_finetune_sf_qlora VLM_PYTHON="$HOME/.venv-vllm/bin/python" bash src/components/deploy/vllm/start_vlm.sh adapter --gpu-memory-utilization 0.40
```

For the full demo,
restore the previously verified reasoning-dpo server on port 8002 using its
existing launch procedure/memory budgets. Do not start either server during the
Transformers evaluation. In the application terminal:

```bash
source ../.venv/bin/activate
OPTUM_REASONING_BACKEND=text_dpo python -u -m scripts.run_claim_graph --image data/assignment_dataset/invoices/invoice_01_standard.jpg --date 2024-01-15 --endpoint http://127.0.0.1:8000 --output-dir reports/claim_graph/optum_final --verbose

# Optional existing browser app; uses its unchanged production extraction prompt.
OPTUM_REASONING_BACKEND=text_dpo VLM_BASE_URL=http://127.0.0.1:8000/v1 python -m uvicorn UI.main:app --host 127.0.0.1 --port 8001 --workers 1
```

Upload the original image through port 8001 forwarded in VS Code, then inspect
real extracted CSV, agent trace, review requirements and JSON/PDF consistency.
The comparison's general prompt differs from the production prompt: passing
this comparison does not guarantee production-prompt compatibility. GPU kernel
and adapter-serving compatibility remain manual checks. REVIEW_REQUIRED is a
legitimate result; the graph CLI returns exit code 2 for it.
