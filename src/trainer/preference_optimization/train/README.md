# Phase 9B: text reasoning DPO + QLoRA

This trains **Qwen/Qwen2.5-3B-Instruct**, a new text-only reasoning backbone.
The existing Qwen2.5-VL invoice extractor and its SFT adapter are untouched.
No serving, agent routing or Phase 8 application integration is included.

Standard TRL 0.24.0 `DPOTrainer` optimizes sigmoid DPO preference loss: relative
chosen/rejected assistant log probabilities against a frozen reference policy.
This is offline DPO, with no PPO, reward model or online RLHF pipeline.
`ref_model=None` plus a new PEFT configuration uses TRL's adapter-disabled,
`no_grad` frozen quantized base as the reference. `reference_free=False` is
mandatory; no second full GPU model is created. Optional
`--precompute-reference` uses TRL reference log-probability precomputation.
Audits check disabled LoRA during reference forwards, finite log probabilities,
backward gradients, changed adapter weights and completed optimizer steps.

The installed Transformers 4.57.1 Qwen2 causal-LM mapping and TRL source were
inspected. The text-model tokenizer/template has NOT been downloaded or loaded
during code preparation. Manual startup verifies the actual template, prompt
boundary, decoded assistant JSON and complete sequence lengths before loading
model weights. TRL masks prompt/padding tokens from completion log probabilities.
No sequence is silently truncated. The stock TRL completion-EOS behavior is
retained and included in the audit.

Phase 9A provenance still names the VL model; it is not rewritten. The same
contracts and labels transfer to text reasoning through the NEW tokenizer.
The loader reuses Phase 9A canonical validation for train (30 Agent 1 + 30 Agent 2)
and validation (5 + 5). It reads manifest metadata for split isolation, but
**never opens test.jsonl**. Source/policy hashes must still match.

Defaults: NF4 double quantization, rank 8/alpha 16, language attention
q/k/v/o projections only, frozen embeddings/LM head/base, learning rate 1e-5,
beta 0.1, batch 1, accumulation 4, 20 steps, seed 42, SDPA, gradient
checkpointing and paged AdamW 8-bit. BF16 is used when supported, otherwise FP16.
TRL disables dropout for stable policy/reference comparisons; configured LoRA
dropout 0.05 therefore has effective training dropout **0** (recorded in metadata).
There is no vision encoder in this new backbone.

## Manual commands

Run from the existing repository, using the original training environment:

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate
python -m unittest tests.test_dpo_training -v
python -u -m src.trainer.preference_optimization.train.train_dpo --check
```

`--check` inspects installed APIs, versions, data and output safety on CPU. It
does not download/load a model or inspect CUDA. requirements-dpo.txt records
the already-inspected stack; no new package is expected and Torch is deliberately
not pinned/replaced. A mismatch fails rather than installing anything.

Optional CPU template check uses only a previously cached tokenizer and fails
clearly if it is absent:

```bash
python -u -m src.trainer.preference_optimization.train.train_dpo --check --check-tokenizer
```

**Stop the existing vLLM server manually before either GPU training command.**
No process is killed by this code. One visible GPU and at least 18 GiB free
memory are required; these checks do not guarantee kernels or memory fit.
The following training commands may download the new text model when you run
them. Add `--local-files-only` to require an existing cache.

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.trainer.preference_optimization.train.train_dpo --check --check-gpu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m src.trainer.preference_optimization.train.train_dpo --max-steps 1 --smoke 2>&1 | tee dpo-smoke.log
CUDA_VISIBLE_DEVICES=0 python -u -m src.trainer.preference_optimization.train.train_dpo --max-steps 20 2>&1 | tee dpo-training.log
```

The smoke run performs real DPO optimization, validation and adapter saving,
isolated under `models/qwen25_3b_reasoning_dpo/smoke/`. It cannot publish a full
training pointer. Run the smoke first and inspect its exit status/logs.
`pipefail` preserves training failure through `tee`.

## Outputs and verification

Each unique run is saved under:

```text
models/qwen25_3b_reasoning_dpo/
  latest_successful.json
  runs/<UTC-time>-<unique-id>/
    adapter/adapter_config.json
    adapter/adapter_model.safetensors
    adapter/tokenizer.json
    adapter/tokenizer_config.json
    adapter/...(other tokenizer files)
    checkpoints/checkpoint-<step>/...(adapter-only checkpoints)
    training_config.json
    training_args.json
    dataset_manifest.json
    metrics.json
    training_log.json
    training_result.json
```

The final adapter uses the lowest validation-loss checkpoint; test data is not
used for selection. Metrics/logs contain DPO loss, standard TRL preference rewards,
accuracy/margins, learning rate, steps, runtime and peak CUDA memory. Preference
metrics do not establish real adjudication improvement; that belongs to Phase 9C.

```bash
python -u -m src.trainer.preference_optimization.train.train_dpo --max-steps 1 --smoke --verify-latest
python -u -m src.trainer.preference_optimization.train.train_dpo --verify-latest
```

Verification checks text-model identity, paired finite language-attention LoRA
tensors, rank, tokenizer artifacts and SHA256. The relative adapter pointer is
atomically updated only after completed, finite, genuinely changed training and
artifact verification. Failed runs preserve the previous pointer. Adapter weights
are ignored by Git; extraction output directories are explicitly forbidden.
Periodic checkpoints are adapter-only, not optimizer-resume checkpoints. After
failure, rerun into a new unique run; no automatic resume is implemented.

If CUDA OOM occurs, first ensure vLLM and other GPU jobs are stopped. Batch size
is already 1; accumulation changes effective batch, not single-pair activation
memory. Keep checkpointing enabled; optional reference precomputation can help
but is not a memory guarantee. A lower `--max-length` is allowed only if EVERY
existing pair fits; the audit refuses truncation of policy evidence or JSON.
Do not replace Torch/CUDA/Transformers to troubleshoot automatically. Actual
GPU fit, FP16/BF16 kernels, tokenizer compatibility and training success remain
unverified until you run these commands. Future text-model serving uses a separate
reasoning endpoint, outside Phase 9B.
