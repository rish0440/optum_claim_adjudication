# Final trained adapters

Each directory has a repository-relative `latest_successful.json`. Keep that
pointer, the selected final `adapter/` and its run metadata together. No weights
were moved, merged or changed during cleanup.

| Adapter root | Selected run | Base model |
| --- | --- | --- |
| `qwen25_vl_sft` | `20261010T174916Z-0b733db4` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen25_3b_reasoning_dpo` | `20261011T074847Z-60ab777d` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen25_vl_cord_sft` | `20261011T134348Z-f1e33111` | `Qwen/Qwen2.5-VL-3B-Instruct` |

Weights are `ROOT/runs/RUN/adapter/adapter_model.safetensors`. Each is about
15 MB; all three are narrowly declared Git LFS objects in `.gitattributes`.
Git LFS is missing on this VM: attributes alone do not store weights in LFS.
Complete the manual delivery checks before staging or pushing them.

Pointer-recorded SHA256 values (verification commands in
[operations](../docs/OPERATIONS.md)):

```text
Original SFT  24ba883c5ae7ee687e965b9602abc3701da97778bf5482727923a47c7b62844a
Reasoning DPO 29431d4df8baa1e7690b5ec1d2a1ecabcd1f3dd28b75243a75445e0210ca08bf
CORD SFT      2c7f8dd988a579effb348a3e8a1530f499fd46bfdaa64e27e8bd807b14bbc969
```

Retain adapter configuration, saved chat template, tokenizer and image processor
files, metrics, configuration, dataset provenance and successful-run records.
The small selected `training_log.json` files are deliberately retained as
checkpoint metadata; existing verification/training contracts consume them.
Console logs, smoke/failed runs, periodic checkpoints and full pretrained base
weights remain excluded. Acquire/cache backbones separately; never add them here.

Future successful runs require reviewing both the `.gitignore` allowlist and
the exact `.gitattributes` weight paths. An updated pointer alone is insufficient
for a new deliverable. A fresh LFS clone needs `git lfs pull` before startup.

Final demo selects CORD SFT through process-scoped `OPTUM_SFT_OUTPUT_DIR`; the
original remains available for rollback. Reasoning stays independently selected
with `OPTUM_REASONING_BACKEND=text_dpo`. See [README](../README.md).
