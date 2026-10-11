# Phase 9A: offline reasoning preference annotation

Uses the existing Phase 3 preference-optimization namespace rather than adding
a second DPO package. No trainer, inference integration or runtime fallback is
implemented. Existing agents, app, policy, calculator and extraction are unchanged.

From the repository root, manually run:

```bash
source ../.venv/bin/activate
python -m src.trainer.preference_optimization.annotation.generate --output-dir data/dpo --seed 20240901
python -m src.trainer.preference_optimization.annotation.validate --dataset-dir data/dpo
python -m unittest tests.test_dpo_dataset -v
```

Existing app dependencies suffice (Pydantic/PyYAML/Pillow/httpx); no new package
is required. Nothing imports a training/GPU library or loads an embedding model.
Generation refuses to overwrite existing outputs. Use a fresh output directory
to regenerate. No dataset files were generated during code preparation.

Expected target: 40 synthetic cases, one pair per agent per case: **80 pairs**,
60 train (30 per role), 10 validation (5 per role), 10 test (5 per role).
Counts remain pending execution. Context/rule failures flag the whole case and
exclude both roles; actual counts and any shortfall are printed and recorded.

Output:

```
data/dpo/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
  annotation_review.csv
```

Each training record has only `prompt`, `chosen`, `rejected`. `prompt` contains
the actual system instruction and compact user JSON captured through the
unchanged agents. Each completion is one assistant message containing raw JSON:

- Agent 1: `CandidateSelections`, with `item_index`, `candidate_index`,
  `suggested_status`, `basis`. Canonical IDs/conditions are resolved in Python;
  the post-processed `Suggestions` artifact is **not** a completion target.
- Agent 2: `FactSelections`, with `item_index`, `status`, `fact_indices`,
  `concerns`. Facts and concerns have separate, row-scoped integer namespaces;
  all supplied facts/required concerns are selected. Approved/capped items have
  no manual-review concern unless an actual advisory disagreement is supplied.
  The post-processed `ReviewOutput` artifact is **not** a completion target.

Actual Agent 1 inputs do not contain quantities/billed amounts: they contain
item names, date applicability, context requirements and filtered candidates.
These prompts preserve that deployed contract rather than introducing fields.
Agent 2 receives actual calculator decisions and totals. Neither completion can
contain money, totals, free-form prose or generated policy IDs.

The backbone is **Qwen/Qwen2.5-VL-3B-Instruct**, served as `qwen-base`, matching
the code and the user's verified deployment. The server was not queried here.
Do not substitute Qwen2.5-3B-Instruct. In Phase 9B, use the matching Qwen2.5-VL
processor/tokenizer chat template on these conversational records; do not manually
insert role tokens/EOS, concatenate answers into prompts, or use a text-only
backbone's template. No processor/tokenizer was loaded or verified in Phase 9A.
Future TRL version/template/truncation compatibility must be checked before training.

Canonical policy conditions come verbatim from `data/policy.json`. Cases cover
quantity/per-claim caps, actuals, exclusions and explicit excluded subtypes,
missing evidence, unknown similar categories, medical-context uncertainty,
out-of-period dates, unrelated synthetic retrieval hits, advisory disagreement,
wrong-item facts/concerns, missing concerns and contradictory cap status.
Synthetic retrieval fixtures are canonical records, **not live Milvus results**.
Irrelevant records pass through the existing agent filtering; they are not taught
as applicable rules. No patient identities, held-out invoice images, reference
invoice files, validation labels or private records are read. Generated billing
values and dates are fresh seeded cases, not copied reference-claim answers.

Splits are assigned to category/case groups before billing variations. Room
normal/missing-evidence variants remain together; both roles and both rows of
each claim stay in the same split. Each other category group is isolated. Shared
canonical policy language/system instructions are expected, not claim leakage.

Annotation labels are deterministic canonical selections, not model-sampled or
human-verified preferences. The offline capture client is isolated to this package
and cannot replace a production LLM. Existing agents validate chosen answers;
Agent 1's repaired result alone is insufficient, so raw advisory disagreement
is also checked. Agent 2 negatives must trigger the actual documented validator.
All negatives are structurally valid. Wrong-item negatives deliberately select
a **real** fact/concern from another row: scoped validity fails, as documented.
Fabricated/nonexistent references are rejected even in negative examples.

The manifest contains case provenance, raw-error criteria, actual Decimal results,
policy/contract hashes, seed, counts and human-review status. Validation regenerates
case inputs, compares captured prompts/conditions, recalculates monetary facts,
checks identities/indices, duplicates, grouped split isolation and role balance.
Edited/unsupported cases fail rather than acquiring guessed labels. The review
CSV lists preferred/rejected selections, policy context and preference criteria.
Inspect it with the manifest before training; automatic validation is not
independent clinical or human annotation verification.

Length checking uses an intentionally conservative **heuristic**:
UTF-8 serialized chat bytes divided by two, rounded up, plus 128 tokens of
overhead, for prompt plus the longer completion. Values above 2048 are excluded.
This is not an exact token count or guaranteed bound. Verify both completions
with the real tokenizer and chat template in Phase 9B; never silently truncate.

Limits: small synthetic category-focused cases, two rows per claim, limited
clinical context and a narrow set of counterexamples. No claim of measured
generalization or improved live agents is made. Existing deterministic financial
authority must remain in place after any future DPO training.
