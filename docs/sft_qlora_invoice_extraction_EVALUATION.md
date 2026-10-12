# Invoice extraction evaluation

The default comparison is pretrained Qwen2.5-VL versus the retained invoice SFT model. The original synthetic SFT model has been removed, so older original-versus-CORD measurements are historical. Use `--baseline-root` only when supplying another verified comparison adapter.

The SFT pointer and CPU tensor/checksum verifier identify the trained model. Its metadata must show a completed non-smoke run, finite losses, changed weights and the approved dataset fingerprint. Evaluation does not modify model artifacts.

Only test/test.jsonl, test annotations and its 19 image/CSV pairs are opened.
Original-source annotation fingerprints and exact Decimal labels are rechecked
with the Phase 10A normalization/image utilities. Train/validation separation
and their recorded hashes are checked using manifest metadata; their images,
CSV labels and JSONL samples are not opened. Test files are never written.

One NF4/double-quantized BF16 Qwen2.5-VL base is loaded at the SFT training revision with one processor/chat template. The retained adapter is frozen and its loaded weights are checked. Default baseline generation temporarily disables LoRA; SFT generation enables it. Both modes use the same pinned base, images, prompt and decoding settings. With `--baseline-root`, a second verified adapter is loaded for comparison.

The exact CORD training instruction/hash is used unchanged for both models.
No optional production-prompt diagnostic is included. Defaults: 50176 minimum
pixels, 200704 maximum pixels, 2048 context tokens including up to 1024 new
tokens, greedy do_sample=False, one beam, seed 42, SDPA. Neither labels nor
assistant messages enter generation. Images must produce vision-forward hook
events. Prompt tokens are removed before decoding. Raw generated text is
preserved; no Markdown stripping, output repair, numeric derivation or gold
constraints. Missing EOS is recorded as truncation, with zero extraction credit.
Context-budget violations are recorded as failures without silently cropping
tokens. Pixel resizing is configured identically for both modes.

## Metrics

Names receive whitespace normalization only (case/punctuation remain exact).
Nonnegative plain-decimal rates/amounts and positive integral quantities are
compared with Decimal. Empty rate cells are preserved and invalid for this
explicit-rate CORD ground truth; they are never calculated from other fields.
CSV syntax, exact header, four columns, numeric validity, arithmetic consistency,
strict original-order document match and order-insensitive match are distinct.

One-to-one row alignment considers exact normalized names or agreement on all
three numeric fields, greedily selecting highest four-field agreement first.
Ties use reference/predicted indices. Duplicate predictions cannot earn repeat
credit. Details record alignment, mismatches and missing/extra indices, which
resolve back to original parsed/reference rows. Item detection is an alignment
metric, not proof that every matched field is correct. Complete-row recall and
precision are also reported. Field denominators include all ground-truth rows;
extra rows are penalized by row precision and document match. All 19 documents
remain in denominators even on inference failures. A smoke report explicitly
uses one document and must never be presented as the final 19-image result.

## Manual commands

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate

# CPU fixtures/mocks only
python -B -m unittest discover -s tests -v

# CPU adapter tensors, hashes, training metadata and held-out provenance checks.
# No processor/model loading or CUDA initialization.
python -u -m trainer.sft_qlora_invoice_extraction.evaluate --preflight --local-files-only

# Manually stop BOTH vLLM servers (8000 and 8002) before GPU work.
nvidia-smi
mkdir -p reports/cord/evaluation/logs
set -o pipefail

# One-image inference on both genuine adapters (first manifest test image).
CUDA_VISIBLE_DEVICES=0 python -u -m trainer.sft_qlora_invoice_extraction.evaluate \
  --smoke --local-files-only --max-new-tokens 1024 \
  2>&1 | tee "reports/cord/evaluation/logs/smoke-$(date -u +%Y%m%dT%H%M%SZ).log"

# Full fixed 19-image held-out comparison, after checking smoke results.
CUDA_VISIBLE_DEVICES=0 python -u -m trainer.sft_qlora_invoice_extraction.evaluate \
  --local-files-only --max-new-tokens 1024 \
  2>&1 | tee "reports/cord/evaluation/logs/full-$(date -u +%Y%m%dT%H%M%SZ).log"

# Optional second-terminal monitoring
watch -n 1 nvidia-smi

# Replace RUN_ID with the actual report directory printed by the runner.
python -u -m trainer.sft_qlora_invoice_extraction.evaluate --inspect reports/cord/evaluation/RUN_ID
```

The runner implements --output-dir, --baseline-root, --cord-root, --dataset-root,
--min-pixels, --max-pixels, --max-context-length, --max-new-tokens, --seed and
--local-files-only. Use the same settings for both models; the runner enforces
this internally. Outputs must remain under reports/cord/evaluation/. Each run
has a unique directory. Existing reports remain unchanged. --local-files-only
fails if the pinned base/processor is not cached; no package installation or
download occurs in preflight. No new dependency is required.

At least 18 GiB free GPU memory and one visible BF16 GPU are required. This is
not an OOM guarantee. Do not tune token/pixel budgets on the held-out outcomes.
The CPU preflight cannot verify actual PEFT loading, CUDA kernels or vision
generation; the manual smoke does those checks. A global loading/verification
failure leaves a failed manifest/partial predictions, not fabricated metrics.

Outputs: baseline_predictions.jsonl, cord_sft_predictions.jsonl,
per_image_comparison.csv, per_image_comparison.json, summary.json, summary.md,
evaluation_manifest.json. They include prompt/base/adapter/dataset fingerprints,
raw responses, parsed CSV, errors/truncation, timing, token counts and GPU stats.
--inspect prints the measured comparison and up to four genuine outcome
categories (both succeed, baseline wins, CORD wins, both fail), when present.

This small retail/restaurant test is not a healthcare-invoice benchmark. Do
not interpret training loss as extraction accuracy, tune checkpoints/prompts
on these test outcomes, or declare CORD superior before measurements. Optum
images, application integration and deployment decisions belong to Phase 10D.
