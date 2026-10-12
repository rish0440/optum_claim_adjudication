# Dataset layout

The `data/` directory contains exactly three folders:

```text
data/
  cord_sft_qlora/                   # 161 train / 20 validation / 19 test receipts
  policy_reasoning_dpo_qlora/       # 60 train / 10 validation / 10 test preferences
  assignment_dataset/
    policy.json                    # authoritative source policy
    invoices/                      # five original assignment images
    ground_truth/                  # original reference labels and expected decisions
    synthetic/                     # generated invoice training fixtures
```

Policy retrieval assets live beside their implementation:

```text
src/components/rag/
  policy_rag_cases.json
  policy_index/                    # existing local Milvus index and compatibility manifest
```

Application paths are defined in `src/config/default.yaml`. Training/evaluation defaults, scripts and the Colab bundle use these same locations. Optional CORD source downloads go under `.cache/cord_source/`.

The five original assignment invoices remain held out from training. All moved images, CSVs, JSONL records, policy content and annotation manifests retain their original bytes. Loaders resolve historical image/label paths without rewriting the original records. Existing preference snapshots and policy citations retain `data/policy.json` as a stable source identifier; the authoritative file is read from `data/assignment_dataset/policy.json` through configuration.
