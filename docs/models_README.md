# Trained model artifacts

The repository retains two trained models. Each root has a `latest_successful.json` pointer to its selected run and `model_weights/` folder.

| Model root under `models/` | Selected run | Backbone |
| --- | --- | --- |
| `qwen25_finetune_sf_qlora` | `20261011T134348Z-f1e33111` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen_25_finetune_dpo_qlora` | `20261011T074847Z-60ab777d` | `Qwen/Qwen2.5-3B-Instruct` |

Weights are `ROOT/runs/RUN/model_weights/adapter_model.safetensors`. The weight file names and PEFT configuration format are unchanged. Both selected weight files have exact Git LFS declarations in `.gitattributes`.

```text
Invoice SFT  2c7f8dd988a579effb348a3e8a1530f499fd46bfdaa64e27e8bd807b14bbc969
Reasoning DPO 29431d4df8baa1e7690b5ec1d2a1ecabcd1f3dd28b75243a75445e0210ca08bf
```

Keep the pointer, weights, tokenizer/processor files and run metadata together. Runtime configuration, launchers and future training use these renamed roots and `model_weights/`. The extraction pointer is also the default inference selection.

Saved training configurations and metrics preserve paths recorded at training time. These are historical records; their checksum-protected contents and trained weights were preserved during the rename. The `adapter` keys in pointer/result JSON and PEFT filenames describe the model artifact format and remain unchanged.

The original synthetic SFT model was removed at the owner's request. Existing reported original-versus-CORD results are historical. Current extraction evaluation defaults to pretrained base versus the retained SFT model; `--baseline-root` can select a separately supplied verified comparison model.

Console logs, smoke runs, periodic checkpoints and pretrained backbones remain outside the delivery allowlist. Future successful runs require updating `.gitignore` and `.gitattributes` together. A fresh Git LFS clone needs its weight objects downloaded before startup. See [operations](OPERATIONS.md) and [training tools](trainer_README.md).
