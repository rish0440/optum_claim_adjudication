# Training tools

Run these modules from the repository root using `../.venv`.

| Folder | Files and purpose |
| --- | --- |
| `trainer/sft_qlora_invoice_extraction/` | `train.py`: training and adapter checks; `data_prep.py`: download, preparation and validation; `evaluate.py`: CORD and Optum evaluation; `runtime.py`: training collator and GPU execution; `__init__.py`: package marker |
| `trainer/dpo_qlora_policy_reasoning/` | `train.py`: training and adapter checks; `data_prep.py`: preference generation, validation and loading; `evaluate.py`: preference evaluation; `evaluate_agent.py`: structured Agent 2 evaluation; `provenance.py`: recorded dataset source verification; `__init__.py`: package marker |

Each retained file supports training, inference adapter resolution, data preparation or evaluation. There are no nested helper packages.

```bash
python -m trainer.sft_qlora_invoice_extraction.data_prep download
python -m trainer.sft_qlora_invoice_extraction.data_prep prepare
python -m trainer.sft_qlora_invoice_extraction.data_prep validate
python -m trainer.sft_qlora_invoice_extraction.train --preflight --local-files-only
python -m trainer.sft_qlora_invoice_extraction.train --verify-latest
python -m trainer.sft_qlora_invoice_extraction.evaluate --preflight --local-files-only
python -m trainer.dpo_qlora_policy_reasoning.data_prep generate
python -m trainer.dpo_qlora_policy_reasoning.data_prep validate
python -m trainer.dpo_qlora_policy_reasoning.train --verify-latest
python -m trainer.dpo_qlora_policy_reasoning.evaluate --help
python -m trainer.dpo_qlora_policy_reasoning.evaluate_agent --help
```

Data preparation commands create or validate datasets; use `--help` to inspect their arguments. Existing datasets must pass validation before training. Stop both GPU servers before starting training or a Transformers evaluation. Extraction and reasoning adapters remain separate.

See [CORD preparation](trainer_sft_qlora_invoice_extraction_README.md), [vision training](sft_qlora_invoice_extraction_TRAINING.md), [vision evaluation](sft_qlora_invoice_extraction_EVALUATION.md), [structured reasoning evaluation](dpo_qlora_policy_reasoning_AGENT2_STRUCTURED.md) and [model artifacts](models_README.md).
