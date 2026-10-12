# Phase 10B: isolated CORD QLoRA SFT

This experiment starts from **Qwen/Qwen2.5-VL-3B-Instruct**, never the original
invoice adapter or text reasoning model. It reuses the successful SFT HF Image
builder, TRL 0.24.0 vision collator, CheckedInvoiceCollator, AuditedSFTTrainer,
LiveProgress and CPU PEFT adapter verifier. The production extraction instruction is preserved; the active training CLI
is the invoice-extraction trainer below. No new dependencies are introduced.

The dedicated reader validates the Phase 10A manifest and the 161 train / 20
validation records, original annotations, exact Decimal/explicit-rate CSVs,
image pixels and message pairs. It checks global split metadata for duplicate
IDs/content but **never opens any test image, annotation or test.jsonl**. The
19 test receipts are reserved for Phase 10C; five Optum images for Phase 10D.

The general receipt/invoice instruction lives in `trainer/sft_qlora_invoice_extraction/data_prep.py`. Its exact
text, version and SHA256 are saved in training_config.json. An optional
--instruction-file PATH overrides it without modifying labels or production
prompts. Phase 10C must reuse this recorded instruction and assess production
prompt compatibility separately.

Defaults: NF4 with double quantization, BF16, language q/k/v/o attention LoRA
r=8 / alpha=16 / dropout=0.05, batch=1, accumulation=4, LR=1e-4, constant LR,
2 epochs, seed=42, paged_adamw_8bit, gradient checkpointing with non-reentrant
backward. Vision tower/merger, embeddings, LM head and base weights are frozen.
Target names are checked against the actual Transformers model structure.
This gives **82 expected optimizer steps**, including each epoch's final
partial accumulation group, using the installed Transformers step calculation.
The actual completed steps/epochs and losses are recorded, never invented.

The shared builder retains decoded HF images and exact assistant CSVs. Only
the in-memory user instruction changes. TRL uses the matching Qwen processor
and chat template, expands image tokens and passes pixel_values/image_grid_thw
into the model. Prompt, padding and image tokens are ignored in labels. The
audited collator verifies supervised tokens decode to the full CSV; a vision
hook and gradient audit check actual image forward and LoRA backward execution.
max_length=None prevents truncation. The wrapper rejects full processed batches
exceeding --max-sequence-length (default 2048). This model-specific token check
runs after processor loading, before training, on every train/validation item.
CPU preflight does not claim to verify tokenizer budgets or GPU execution.

Validation loss is evaluated at epoch boundaries and once at completion. The
best validation-loss checkpoint is restored before final adapter saving. No test
data influences selection. CSV generation accuracy is not measured in Phase 10B;
lower loss alone does not establish better extraction.

## Manual commands (repository root)

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate

# CPU fixtures; no model loading/downloads
python -B -m unittest tests.test_cord_training tests.test_adapter_checkpoints -v

# CPU manifest/image/HF checks, no processor/model loading or output writes
python -u -m trainer.sft_qlora_invoice_extraction.train --preflight --local-files-only

# Manually stop BOTH vLLM processes on ports 8000 and 8002 first.
# The trainer never kills services. Inspect memory before starting GPU work.
nvidia-smi
```

At least 18 GiB free GPU memory and one visible BF16-capable GPU are required.
--min-free-gib is configurable; availability is not an OOM guarantee. Keep batch
size 1. If OOM occurs, lower --max-pixels to 100352 (128*28*28), keeping
--min-pixels <= --max-pixels; increase accumulation if needed. If an intact
CSV/context exceeds 2048, lower image pixels or deliberately raise
--max-sequence-length with sufficient VRAM. Never truncate rows or image tokens.
No separate FlashAttention installation is required (SDPA is used).

Use Bash pipefail so tee preserves failure status:

```bash
mkdir -p reports/cord/training
set -o pipefail

# Exactly one real optimizer step, full validation, isolated smoke adapter.
CUDA_VISIBLE_DEVICES=0 python -u -m trainer.sft_qlora_invoice_extraction.train \
  --smoke --max-steps 1 --local-files-only \
  2>&1 | tee "reports/cord/training/smoke-$(date -u +%Y%m%dT%H%M%SZ).log"

# Only after inspecting the successful smoke: 2 epochs, validation selection.
CUDA_VISIBLE_DEVICES=0 python -u -m trainer.sft_qlora_invoice_extraction.train \
  --epochs 2 --batch-size 1 --grad-accum 4 --learning-rate 1e-4 \
  --seed 42 --local-files-only \
  2>&1 | tee "reports/cord/training/train-$(date -u +%Y%m%dT%H%M%SZ).log"

# Optional second terminal monitoring; never launched by the trainer.
watch -n 1 nvidia-smi

# CPU checkpoint/config/base-ID/tensor/rank/finite-value/hash checks
python -u -m trainer.sft_qlora_invoice_extraction.train --verify-latest

# For smoke verification use the actual path printed by that run:
# python -u -m trainer.sft_qlora_invoice_extraction.train --verify-adapter models/qwen25_finetune_sf_qlora/smoke/runs/<run-id>/model_weights

cat models/qwen25_finetune_sf_qlora/latest_successful.json
python -c 'import json; from pathlib import Path; root=Path("models/qwen25_finetune_sf_qlora"); pointer=json.loads((root/"latest_successful.json").read_text()); run=(root/pointer["adapter_path"]).parent; print((run/"metrics.json").read_text()); print("Adapter artifacts:"); print("\n".join(sorted(p.name for p in (run/"adapter").iterdir())))'
find models/qwen25_finetune_sf_qlora/runs -maxdepth 3 -type f \
  \( -name metrics.json -o -name training_config.json -o -name manifest.json \
     -o -name adapter_config.json -o -name adapter_model.safetensors \)
```

--local-files-only uses cached model assets and fails clearly if they are missing;
the user may explicitly omit it to permit Hugging Face downloads during their
manual run. Existing Torch/CUDA/Transformers/TRL/PEFT/bitsandbytes must not be
replaced. Preflight reports versions and requires the known TRL 0.24.0 /
Transformers 4.57.1 APIs in ../.venv. It does not initialize CUDA.

## Saved outputs and inspection

```text
models/qwen25_finetune_sf_qlora/
  latest_successful.json                # full runs only; atomic, verified
  runs/<run-id>/
    model_weights/                           # PEFT weights + processor/tokenizer
    checkpoints/                       # up to two epoch checkpoints
    training_config.json                # prompt/hash, hyperparameters, model revision
    run_config.json                     # shared pointer verifier compatibility
    training_args.json
    manifest.json                       # train/validation hashes + source metadata
    metrics.json
    training_log.json
    training_result.json
  smoke/runs/<run-id>/                  # same artifacts, never updates pointer
```

Output overrides must remain inside models/qwen25_finetune_sf_qlora/. Every run uses
a new ID; previous checkpoints are preserved. Failed/interrupted/zero-step,
unchanged-weight or non-finite runs never publish a pointer. Initial/final
trainable tensor digests must differ. Saved safetensors are CPU-checked for
language-only LoRA A/B tensors, rank, finite values, base identity and weights
SHA256. Metadata hashes are also verified. These checks do not prove extraction
improvement or future serving compatibility.

Use the verified command's reported run ID to inspect that run's metrics.json,
training_config.json and training_result.json: actual steps, epochs, training
and validation loss, learning-rate log history, runtime, peak memory, LoRA
parameter count, changed tensor digests and artifact SHA256. No real metrics
exist until you run training. All weights/logs stay Git-ignored.

Stop here: serving, 19 held-out receipt evaluation, adapter comparison and
Optum/end-to-end testing belong to Phase 10C/10D.
