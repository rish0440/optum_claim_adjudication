# Phase 9C: independent held-out baseline vs DPO evaluation

Both modes use **Qwen/Qwen2.5-3B-Instruct**. The existing VL extraction model,
application, agents, graph and training files are untouched. This evaluator loads
one NF4/double-quantized base, loads only the pointer-verified reasoning adapter,
and uses PEFT's adapter-disabled context for baseline and active `dpo` for DPO.
Every pass checks that LoRA layers are unmerged and disabled/enabled as intended.
No second full model, vLLM server, optimizer or training step is involved.

Only `data/dpo/test.jsonl` and manifest metadata are opened. Train/validation
JSONL are never opened. The Phase 9A validator is reused on those 10 test records
against canonical policy and the original synthetic-case registry. This is
annotation replay for validation, not dataset regeneration or model generation.
Unverifiable labels cause preflight failure rather than a fabricated correct
score; human verification of synthetic annotations remains pending.

The successful adapter is selected through the existing
`models/qwen25_3b_reasoning_dpo/latest_successful.json` verifier, never a hardcoded
run. Training provenance must match the unchanged manifest and indicate that
test was not accessed. At startup the base tokenizer vocabulary, special tokens
and chat template must match the saved training tokenizer/template hash.

Teacher-forced scoring reuses Phase 9B serialization/tokenization, including
stock TRL completion EOS. It sums log probabilities of ONLY assistant completion
tokens, never prompt/padding. Baseline/DPO use identical token sequences and a
1e-6 tie tolerance. Ties are not correct. These are raw preference probabilities,
not reference-relative DPO reward accuracy. Unequal completion lengths can affect
summed log probabilities; token lengths are recorded. Scores remain comparable
because the same method and completions are used in both modes.

Generation receives only prompt tokens, with greedy decoding and identical
settings. No chosen/rejected completion enters generation. No JSON repair,
postprocessed agent fallback or ground-truth substitution occurs. Strict raw
CandidateSelections/FactSelections validation checks row identity, scoped
references, status/basis, required concerns and canonical expected behavior.
Malformed/truncated answers remain failures. Explicit citations or invented
amounts are forbidden by the actual raw schema. The report stores sanitized
actual generations, references, discrepancies, denominators and timings.

## Manual commands

Run from the existing repository:

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication
source ../.venv/bin/activate
python -m unittest tests.test_dpo_evaluation -v
python -u -m src.trainer.preference_optimization.eval.evaluate --split test --check
```

CPU preflight validates data, installed versions and adapter tensors/checksums;
it does not load the base model, download anything, inspect CUDA or generate
predictions. No additional dependency is expected beyond Phase 9B's stack.

**Stop the existing vLLM server and other GPU workloads manually.** The evaluator
requires one visible CUDA GPU and at least 18 GiB free. It does not kill anything.
These memory checks do not guarantee kernel compatibility or GPU fit.

Freeze your evaluation settings before running the final test. The default
generation budget is 512 new tokens in a 2048-token context; startup rejects
overlength prompt+budget or scoring sequences, never truncates evidence/JSON.
Do not adjust settings from observed held-out scores.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m src.trainer.preference_optimization.eval.evaluate --split test --max-new-tokens 512 --seed 42 --local-files-only --output-dir reports/dpo/evaluation 2>&1 | tee dpo-evaluation.log
```

`--local-files-only` requires the previously downloaded text backbone/tokenizer.
Omitting it permits downloads when YOU run evaluation. No download/inference
has been performed during code preparation.

Each completed run publishes a unique directory:

```text
reports/dpo/evaluation/<UTC-time>-<unique-id>/
  baseline_predictions.jsonl
  dpo_predictions.jsonl
  pairwise_scores.json
  metrics.json
  comparison.csv
  comparison.md
  metadata.json
  human_review.jsonl
```

On infrastructure/model failure, no completed comparison directory is published;
an uncompleted `.pending` directory may contain `failure.json` and partial files.
Invalid MODEL ANSWERS are scored as failures and still appear in a completed
comparison. No failed inference is silently replaced. Reports/inputs are not
overwritten. After execution, use the exact completed directory printed by CLI:

```bash
cat reports/dpo/evaluation/<printed-run-id>/comparison.md
python -m json.tool reports/dpo/evaluation/<printed-run-id>/metrics.json
python -m json.tool reports/dpo/evaluation/<printed-run-id>/metadata.json
```

Inspect `human_review.jsonl` alongside actual predictions before any deployment
decision. Percentages have explicit denominators; absent categories are N/A.
Generation semantic accuracy, preference ranking and per-agent/scenario metrics
are separate. Reports list improved/worsened/tied for each measure without
automatically declaring DPO superior.

There are only 10 pairs from 5 correlated synthetic cases. Negatives are often
easy, deliberately constructed single-field errors. Labels are not independently
human verified. Agent 2 sees fixed verified decisions, not dynamically generated
baseline/DPO Agent 1 outputs. This is not image extraction or whole-claim accuracy,
nor a measurement of free-text explanation quality. Serving/deployment belongs
to Phase 9D.

Base/adapter identities, cached base weight hashes, resolved revision, adapter
checksum, tokenizer/template hash, dataset/implementation fingerprints, package
versions, precision, settings, seed and memory/runtime are recorded. Phase 9B
did not record base-weight revision/hash, so confirm the local snapshot matches
the training snapshot yourself. The same base object is used for both evaluation
modes regardless. CPU tests, GPU fit and live inference remain unverified until
you execute them. Do not train/tune, choose checkpoints or change agent prompts
from these one-time held-out results.
