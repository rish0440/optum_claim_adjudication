# Independent baseline vs DPO evaluation

Both modes: `Qwen/Qwen2.5-3B-Instruct`. DPO adapter SHA256: `29431d4df8baa1e7690b5ec1d2a1ecabcd1f3dd28b75243a75445e0210ca08bf`.
Checkpoint: `/home/rishabh_mishra0440/optum_claim_adjudication/optum_claim_adjudication/models/qwen25_3b_reasoning_dpo/runs/20261011T074847Z-60ab777d/adapter`.
Split: **test**, 10 pairs ({'agent_1': 5, 'agent_2': 5}), from 5 underlying synthetic cases. Test SHA256: `59810e9d535312175925ff32657f64fe7839fc83f60ef96bde786e1ce998ac66`.
Identical base object, tokenizer, chat template, prompts and greedy settings. Adapter state is audited for every mode.

## Generation and raw preference ranking

Percentages are percentages of explicit denominators in metrics.json; counts are counts. Missing denominators are N/A.

eligible_completion means a nontruncated schema-valid answer with row-owned references, not approval or semantic correctness. Safety rates count affected prompts; reference/approval counts count offending selections.

| Group | Metric | Baseline | DPO | Change | DPO result |
|---|---|---:|---:|---:|---|
| overall | valid_json | 90.000 | 90.000 | 0.000 | tied |
| overall | valid_schema | 60.000 | 60.000 | 0.000 | tied |
| overall | correct_scoped_references | 60.000 | 60.000 | 0.000 | tied |
| overall | eligible_completion | 60.000 | 60.000 | 0.000 | tied |
| overall | semantic_correct | 60.000 | 60.000 | 0.000 | tied |
| overall | unsupported_reference_count | 0.000 | 0.000 | 0.000 | tied |
| overall | wrong_item_reference_count | 0.000 | 0.000 | 0.000 | tied |
| overall | unsupported_approval_count | 0.000 | 0.000 | 0.000 | tied |
| overall | missing_required_concern_count | 0.000 | 0.000 | 0.000 | tied |
| overall | unsupported_review_count | 0.000 | 0.000 | 0.000 | tied |
| overall | classification_correct | 100.000 | 100.000 | 0.000 | tied |
| overall | candidate_correct | 100.000 | 100.000 | 0.000 | tied |
| overall | basis_correct | 100.000 | 100.000 | 0.000 | tied |
| overall | facts_correct | 20.000 | 20.000 | 0.000 | tied |
| overall | concerns_correct | 20.000 | 20.000 | 0.000 | tied |
| overall | status_correct | 20.000 | 20.000 | 0.000 | tied |
| overall | coverage_selection | 100.000 | 100.000 | 0.000 | tied |
| overall | exclusion_identification | 100.000 | 100.000 | 0.000 | tied |
| overall | appropriate_review | N/A | N/A | N/A | not measured |
| overall | unsupported_approval_rate | 0.000 | 0.000 | 0.000 | tied |
| overall | wrong_item_evidence_rate | 0.000 | 0.000 | 0.000 | tied |
| overall | generation_failures | 4.000 | 4.000 | 0.000 | tied |
| overall | average_generation_seconds | 10.152 | 14.618 | 4.465 | worsened |
| overall | raw_preference_accuracy | 100.000 | 100.000 | 0.000 | tied |
| overall | raw_preference_mean_margin | 11.929 | 14.312 | 2.382 | improved |
| agent_1 | valid_json | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | valid_schema | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | correct_scoped_references | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | eligible_completion | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | semantic_correct | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | unsupported_reference_count | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | wrong_item_reference_count | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | unsupported_approval_count | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | missing_required_concern_count | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | unsupported_review_count | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | classification_correct | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | candidate_correct | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | basis_correct | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | facts_correct | N/A | N/A | N/A | not measured |
| agent_1 | concerns_correct | N/A | N/A | N/A | not measured |
| agent_1 | status_correct | N/A | N/A | N/A | not measured |
| agent_1 | coverage_selection | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | exclusion_identification | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | appropriate_review | N/A | N/A | N/A | not measured |
| agent_1 | unsupported_approval_rate | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | wrong_item_evidence_rate | N/A | N/A | N/A | not measured |
| agent_1 | generation_failures | 0.000 | 0.000 | 0.000 | tied |
| agent_1 | average_generation_seconds | 4.118 | 5.832 | 1.714 | worsened |
| agent_1 | raw_preference_accuracy | 100.000 | 100.000 | 0.000 | tied |
| agent_1 | raw_preference_mean_margin | 11.026 | 14.802 | 3.776 | improved |
| agent_2 | valid_json | 80.000 | 80.000 | 0.000 | tied |
| agent_2 | valid_schema | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | correct_scoped_references | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | eligible_completion | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | semantic_correct | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | unsupported_reference_count | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | wrong_item_reference_count | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | unsupported_approval_count | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | missing_required_concern_count | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | unsupported_review_count | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | classification_correct | N/A | N/A | N/A | not measured |
| agent_2 | candidate_correct | N/A | N/A | N/A | not measured |
| agent_2 | basis_correct | N/A | N/A | N/A | not measured |
| agent_2 | facts_correct | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | concerns_correct | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | status_correct | 20.000 | 20.000 | 0.000 | tied |
| agent_2 | coverage_selection | N/A | N/A | N/A | not measured |
| agent_2 | exclusion_identification | N/A | N/A | N/A | not measured |
| agent_2 | appropriate_review | N/A | N/A | N/A | not measured |
| agent_2 | unsupported_approval_rate | N/A | N/A | N/A | not measured |
| agent_2 | wrong_item_evidence_rate | 0.000 | 0.000 | 0.000 | tied |
| agent_2 | generation_failures | 4.000 | 4.000 | 0.000 | tied |
| agent_2 | average_generation_seconds | 16.187 | 23.404 | 7.217 | worsened |
| agent_2 | raw_preference_accuracy | 100.000 | 100.000 | 0.000 | tied |
| agent_2 | raw_preference_mean_margin | 12.833 | 13.821 | 0.988 | improved |

## Preference outcome counts

- baseline: correct=10, incorrect=0, tied=0, scoring failures=0.
- dpo: correct=10, incorrect=0, tied=0, scoring failures=0.

Raw preference accuracy ranks summed chosen/rejected completion log probabilities; it is neither DPO reference-relative reward accuracy nor claim accuracy.
Per-scenario results and their small denominators are in metrics.json.

## Representative actual generation failures

- baseline, synthetic-COV-002-agent_2: invalid_schema_or_forbidden_fields. See sanitized output in predictions JSONL.
- baseline, synthetic-COV-012-agent_2: invalid_schema_or_forbidden_fields. See sanitized output in predictions JSONL.
- baseline, synthetic-COV-014-agent_2: invalid_schema_or_forbidden_fields. See sanitized output in predictions JSONL.
- baseline, synthetic-EXC-012-agent_2: invalid_json. See sanitized output in predictions JSONL.
- dpo, synthetic-COV-002-agent_2: invalid_schema_or_forbidden_fields. See sanitized output in predictions JSONL.
- dpo, synthetic-COV-012-agent_2: invalid_schema_or_forbidden_fields. See sanitized output in predictions JSONL.

## Runtime

Total evaluation time: 283.64s. Peak tracked CUDA allocation: 2.87 GiB.
Per-generation latency and pairwise scoring time are recorded separately.

## Limitations and conclusion

Only 10 synthetic preference pairs (5 correlated case groups) were evaluated. These labels are automatically canonical-validated, not independently human verified.
Negatives are deliberately constructed single-field/scoped-reference errors, often easy; they do not represent the full difficulty of clinical reasoning or real claim review.
Both modes use the same NF4 quantized base; this comparison does not measure full-precision baseline behavior.
Agent 2 receives the same fixed verified decisions/Agent 1 context in both modes; this is independent reasoning evaluation, not a new end-to-end deployment.
No free-text explanation quality is claimed because raw agent contracts contain selections, not prose. Semantics are scored only where canonical annotation replay establishes the expected decision.
Training did not record a base-weight revision/hash; evaluation records its resolved base revision/config hash. Confirm the local base snapshot matches the training snapshot before drawing conclusions.
Metric changes above report improved/worsened/tied independently. No overall superiority or deployment decision is inferred. Do not tune prompts, checkpoints or parameters from these test results.
