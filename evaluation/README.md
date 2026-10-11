# Evaluation tools

Documentation index only; no duplicate evaluation implementation.

| Evaluation | Existing module / instructions |
| --- | --- |
| Held-out CORD extraction (19 receipts) | [CORD evaluation](../src/trainer/cord/EVALUATION.md) |
| Original five Optum invoices | [Final Optum comparison](../src/trainer/cord/OPTUM_FINAL.md) |
| Baseline versus DPO text reasoning | [Preference evaluation](../src/trainer/preference_optimization/eval/README.md) |
| Strict Agent 2 generation and medical-review smoke | [Structured evaluation](../src/trainer/preference_optimization/eval/AGENT2_STRUCTURED.md) |
| Policy retrieval | `python -m scripts.test_policy_rag --evaluate` |

Generated artifacts remain local under `reports/`; they are not source fixtures.
User-reported CORD row F1: 53.33% original SFT versus 84.21% CORD SFT on 19
receipts. Five Optum invoices tied in extraction accuracy. DPO did not establish
better generated-answer accuracy. Repeated evaluation of examined test cases is
not a new independent generalization estimate.

Original images require publication clearance; local reference labels and tests
are preserved. See [cleanup report](../docs/CLEANUP_REPORT.md).
