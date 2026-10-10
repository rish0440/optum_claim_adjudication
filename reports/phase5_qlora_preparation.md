# Phase 5: manual Qwen2.5-VL QLoRA training

Code preparation only. No installation, dataset dry run, training, GPU queries/jobs,
model downloads, checkpoint loading, or inference were executed in this phase.
Static AST parsing passed for 11 changed/new Python files; whitespace checks passed.
Six new CPU regression tests are provided but have not been executed.

## Implementation and preserved behavior

Canonical entry point: `src/trainer/sft/train/train_qlora.py`.
The existing `src.trainer.sft.train_lora` and `src.vlm.train_lora` paths remain
compatible, including their `train(dataset_root, output_dir, **kwargs)` API.
The bundle packaging file list includes the relocated implementation and its
validation dependencies; no ZIP was regenerated.

The existing `make_hf_dataset` implementation is unchanged: `images` is
`Sequence(Image())`, containing one decoded invoice image per record; the user
prompt contains the image marker and existing extraction instruction; the assistant
completion contains the exact CSV. `read_records` gains an optional manifest path
that must stay in its own synthetic split. CSV/JSONL parity, empty Rate fields,
arithmetic, image inventories, duplicate content/IDs, and split separation are
checked by the existing Phase 4 validator before training.

Actual manifests remain `data/synthetic/train/annotations.jsonl` (8 images) and
`data/synthetic/validation/annotations.jsonl` (2 images). No data was regenerated.

TRL 0.24.0's existing vision collator applies Qwen's processor/chat template,
passes `pixel_values` and `image_grid_thw`, and masks prompt/padding tokens using
completion-only loss. The wrapper additionally checks image tokens are ignored,
supervised tokens decode to the target CSV, and the batch contains real image
tensors. It rejects empty/text-only/invalid batches. No truncation or packing is
applied, because truncation can discard image tokens or CSV targets. At training
time a vision forward hook and a post-backward gradient audit confirm real
multimodal computation. These runtime checks have not yet been exercised.

## Initial configuration

| Parameter | Default | Purpose |
|---|---|---|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` | Existing model and extraction task |
| Quantization | 4-bit NF4, double quantization | Reduce frozen base-weight memory |
| Compute | BF16 when supported, otherwise FP16 | Automatic device-supported mixed precision |
| LoRA | rank 8, alpha 16, dropout 0.05, bias none | Small conventional PEFT adapter |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Language decoder attention only |
| Batch / accumulation | 1 / 2 | Effective batch 2, small activation footprint |
| Steps / learning rate | 8 / `2e-4` | Short initial functional run |
| Scheduler / warmup | constant / 0 | Simple short-run behavior |
| Pixel bounds | 50,176–200,704 (64–256 × 28²) | Conservative image-token budget |
| Attention / checkpointing | SDPA / enabled, non-reentrant | Avoid flash-attn dependency; save activation memory |
| Optimizer | `paged_adamw_8bit` | bitsandbytes optimizer; `adamw_torch` is an explicit alternative |
| Logging / saves | every 1 / 4 optimizer steps | Live progress, retained intermediate checkpoints |
| Seed | 42 | Seed Python/NumPy/Torch through Transformers, and dataset sampling |
| Evaluation | final validation loss, batch 1 | Separate validation; no generation or retained logits |

Rank, alpha, dropout, pixel bounds, batch, accumulation, steps, optimizer, dtype,
seed, logging, save frequency, manifests and output directory are CLI configurable.
An 8-step run with 8 images and effective batch 2 is about two epochs. This tiny
dataset/run is a functional experiment, not evidence of generalization or reliable
production invoice extraction. Low image resolution also trades OCR detail for
memory safety; later increase it only after measuring usage and accuracy.

The trainer inspects every matched module before attachment: all suffix matches
must be language-layer attention projections. After attachment, every trainable
parameter must be a language LoRA weight. Vision encoder, merger, embeddings,
LM head and all base parameters remain frozen. No `all-linear`, visual targets,
embedding resizing, `modules_to_save`, DoRA or RS-LoRA is used.

## Dependencies and compatibility

Read-only metadata inspection found Python 3.12 and these packages:
Torch `2.14.1` (installed build file: `2.14.1+cu130`), Transformers `4.57.1`,
TRL `0.24.0`, PEFT `0.17.1`, bitsandbytes `0.48.1`, Accelerate `1.12.0`,
Datasets `4.3.0`, Pillow `12.3.0`, safetensors `0.8.0`. `torchvision` is missing.
bitsandbytes has a `libbitsandbytes_cuda130.so` file; its presence does not prove
that its CUDA kernels load or work with the VM driver/PyTorch combination.
The Qwen processor in Transformers 4.57 also initializes its video-processing
component, so torchvision is needed even for this image-only training path.

`requirements-sft.txt` pins the existing Hugging Face stack and matching
`torchvision==0.29.1`. `requirements-sft-constraints.txt` constrains Torch to 2.14.1
and avoids replacing it through dependency resolution. This pin accepts the
existing `+cu130` local build. No Torch reinstall or uncontrolled upgrade is
recommended. No qwen-vl-utils, flash-attn, vLLM or OpenAI dependency is added.
Existing project CPU dependencies (including Pydantic and PyYAML) must remain
installed for dataset validation; they are pinned at the observed installed
versions, Pydantic 2.14.0 and PyYAML 6.0.3. The legacy Colab requirements also
include these two dependencies so the preserved packaging entry point remains
self-contained.

The matching TorchVision release is documented by
[PyTorch's 2.14.1 announcement](https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455).
Installed source signatures were inspected for `dtype`, PEFT non-reentrant
checkpointing, `SFTConfig`, and TRL's completion-only vision collator. No runtime
compatibility, CUDA availability, driver support, memory headroom, or successful
adapter saving has been verified. Stop if CUDA is unavailable or library imports
fail; share the diagnostics before changing the working environment.

The language-only targets are the safest conservative choice for future vLLM.
[vLLM's supported-model list](https://docs.vllm.ai/en/latest/models/supported_models/)
lists Qwen2.5-VL LoRA and describes language-backbone support; visual LoRA support
is version/model dependent and experimental. Future serving must use a version
with the Qwen2.5-VL LoRA implementation and the mapper for Transformers 4.57's
`model.language_model.*` keys. It must enable LoRA with maximum rank >= the saved
rank and use the same base model. Saved QLoRA adapters can be applied to the
ordinary base model; the NF4 training base is not exported or merged. vLLM load
compatibility must be tested in the later inference phase; it is not guaranteed
by passing training. Do not add visual/merger/embedding/head adapters without
checking that serving version first.

## Commands for you to execute, in order

### A. Activate the existing environment and enter the actual Git root

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication
source .venv/bin/activate
cd optum_claim_adjudication
python --version
python -m pip --version
```

The `.venv` is one level above this Git root. Do not create another environment.

### B. Inspect versions, NVIDIA and PyTorch CUDA

```bash
nvidia-smi
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
for name in ('torch', 'torchvision', 'transformers', 'trl', 'peft', 'bitsandbytes',
             'accelerate', 'datasets', 'Pillow', 'safetensors'):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, 'MISSING')
import torch
print('Torch build:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('STOP: CUDA must work before training')
print('GPU:', torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info(0)
print('Free/total GiB:', free / 2**30, total / 2**30)
print('BF16 supported:', torch.cuda.is_bf16_supported())
PY
```

These commands inspect CUDA and may create a small CUDA context; they do not
load a model. Ensure the NVIDIA driver supports the installed CUDA 13.0 runtime.

### C. Install only the missing matching TorchVision package, if still missing

```bash
python -m pip freeze > /tmp/optum-sft-packages-before.txt
python -m pip install --dry-run --no-deps torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

Review the plan; it should add TorchVision without replacing Torch. Then:

```bash
python -m pip install --no-deps torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip check
python - <<'PY'
import torch, torchvision, transformers, trl, peft, bitsandbytes
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
print('Training imports OK:', torch.__version__, torchvision.__version__)
PY
```

If additional training packages are missing, preview the pinned set first:

```bash
python -m pip install --dry-run -r requirements-sft.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

After reviewing, install only the missing/required pinned packages. With the
already-installed transitive dependencies, the controlled full-set command is:

```bash
python -m pip install --no-deps -r requirements-sft.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip check
```

Do not blindly repair unrelated `pip check` conflicts or replace Torch. If a
matching CUDA wheel is unavailable, stop and report that rather than mixing
CPU/CUDA wheel channels. Package imports alone do not verify bitsandbytes kernels.

### D. CPU-only dataset dry run and optional CPU regression tests

```bash
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u -m src.trainer.sft.train.train_qlora \
  --train-manifest data/synthetic/train/annotations.jsonl \
  --validation-manifest data/synthetic/validation/annotations.jsonl \
  --dry-run
```

Expected dataset summary: 10 images, train 8 / validation 2, 64 CSV rows,
train 57 / validation 7, valid image decoding and HF prompt/completion records,
no image overlap. Dry-run makes no model/processor requests or checkpoints and
does not import the training runtime. It cannot validate model-tokenization,
forward/backward computation or GPU memory usage.

Optional tests (none were executed during preparation):

```bash
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest discover -s tests -p 'test_sft_training.py' -v
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest discover -s tests -v
```

### E. Start the short run manually with live output and a log

This is the first command that loads/downloads the base model and trains on GPU.
It uses the existing Hugging Face access configuration; no OpenAI key is needed.
If you exported offline flags in your shell earlier, unset them before this run.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -u -m src.trainer.sft.train.train_qlora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --train-manifest data/synthetic/train/annotations.jsonl \
  --validation-manifest data/synthetic/validation/annotations.jsonl \
  --output-dir models/qwen25_vl_sft \
  --max-steps 8 --batch-size 1 --grad-accum 2 \
  --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --min-pixels 50176 --max-pixels 200704 \
  --logging-steps 1 --save-steps 4 --seed 42 --log-gpu-utilization \
  2>&1 | tee "reports/qlora-$(date -u +%Y%m%dT%H%M%SZ).log"
```

`set -o pipefail` preserves training failures through `tee`. Run the pipeline as
shown and inspect its exit status immediately (`echo "$?"`) if needed. No monitor
is started automatically. The optional utilization flag queries nvidia-smi only
at log events with a one-second timeout and reports unavailable if it fails.

### F. Monitor from a separate SSH/VS Code terminal

```bash
watch -n 1 nvidia-smi
```

Or a compact stream (stop with Ctrl-C):

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 1
```

### G. Verify the latest successful adapter on CPU after training

```bash
CUDA_VISIBLE_DEVICES=-1 python -u -m src.trainer.sft.train.checkpoints \
  --output-dir models/qwen25_vl_sft
python - <<'PY'
import json
from pathlib import Path
from src.config.models import sft_adapter_pointer
pointer_path = sft_adapter_pointer()
pointer = json.loads(pointer_path.read_text())
run = (pointer_path.parent / pointer['adapter_path']).parent
for name in ('run_config.json', 'metrics.json', 'training_result.json'):
    print(name)
    print(json.dumps(json.loads((run / name).read_text()), indent=2))
PY
git status --short --ignored models/qwen25_vl_sft
```

The verifier checks that required files are nonempty, safetensors contains
language-model LoRA A/B tensors with valid rank/shapes, the base model matches,
the run completed successfully, metadata JSON exists, and the SHA-256 matches
the pointer. It reads tensors on CPU and rejects non-finite adapter values;
it performs no inference or adapter loading onto a GPU. Confirm
`global_steps_completed` is 8, losses are finite, and status is `success`.

## Output contract and future discovery

Illustrative terminal output (timings, memory and losses below are placeholders,
not observed training results):

```text
Library versions: {"python": "3.12.x", "torch": "2.14.1", ...}
Dataset: {"status": "PASS", "total_images": 10, "train": 8, "validation": 2, ...}
[HH:MM:SS] CUDA available=True runtime=13.0 GPU=NVIDIA L4 total=<GiB> free=<GiB> dtype=torch.bfloat16
[HH:MM:SS] Loading Qwen processor/chat template and NF4 base model...
[HH:MM:SS] Processor/base-model loading completed in <seconds>s
[HH:MM:SS] Total parameters=<count>; trainable LoRA=<count>; vision/merger frozen
[HH:MM:SS] Multimodal collator preflight PASS: train, 8 images; exact CSV loss targets
[HH:MM:SS] First training batch PASS: vision forward executed and LoRA backward gradients present
[HH:MM:SS] step=1/8 epoch=0.250 loss=<value> lr=0.0002 elapsed=<seconds>s VRAM allocated=<GiB> reserved=<GiB>
[HH:MM:SS] Periodic checkpoint saved: .../checkpoints/checkpoint-4
[HH:MM:SS] Training complete: {"training_loss": <value>, "evaluation": {"eval_loss": <value>}, ...}
[HH:MM:SS] Verified adapter saved: .../runs/<run-id>/adapter
[HH:MM:SS] Saved artifact files: adapter_config.json, adapter_model.safetensors, ...
```

```text
models/qwen25_vl_sft/
├── latest_successful.json
└── runs/
    └── <UTC timestamp>-<unique suffix>/
        ├── run_config.json
        ├── training_args.json
        ├── metrics.json
        ├── training_log.json
        ├── training_result.json
        ├── checkpoints/
        │   ├── checkpoint-4/   # Trainer adapter/processor/state/optimizer files
        │   └── checkpoint-8/
        └── adapter/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── preprocessor_config.json
            ├── tokenizer_config.json
            └── <processor/tokenizer files emitted by save_pretrained>
```

Expected supporting processor files include `chat_template.jinja`,
`tokenizer.json`, `vocab.json`, `merges.txt`, special-token files and
`video_preprocessor_config.json`; exact emitted names are printed live and stored
in `training_result.json` after saving. No adapter files were created in this
phase. The directory is Git-ignored, including optimizer states and metadata.
Existing successful runs/checkpoints are retained; no overwrite/resume/delete is
performed. A failed or interrupted run does not replace the successful pointer.

`latest_successful.json` contains `model_id`, `run_id`, `adapter_path` relative to
the output root, and `adapter_sha256`. `src.config.models.sft_adapter_pointer()`
locates it through `OPTUM_SFT_OUTPUT_DIR` (default `models/qwen25_vl_sft`). When
using a custom CLI output directory, use the same `OPTUM_SFT_OUTPUT_DIR` for later
discovery. Future inference can read that config pointer without importing the
trainer or using a VM-specific absolute path. Load the saved Qwen processor for
Hugging Face inference; in future vLLM serving mirror its pixel limits and chat
template. No vLLM integration or inference behavior was changed here.

## OOM and runtime troubleshooting

Ensure other GPU processes are stopped and inspect free VRAM. Keep batch size 1;
for larger effective batches increase accumulation, which does not shrink the
activation memory of one invoice. First reduce `--max-pixels` to `100352`
(128 × 28²), keeping `--min-pixels 50176`; this may reduce OCR accuracy. Keep
checkpointing enabled and sequence truncation disabled. Rank 4 / alpha 8 reduces
adapter/optimizer memory somewhat, but image activations are usually the larger
lever. Evaluation already uses batch 1 and loss-only output; `--no-evaluation`
explicitly skips it if necessary and records that fact.

Optional `--gpu-memory-fraction 0.85` constrains PyTorch's caching allocator.
It is not a guaranteed cap on total VRAM: CUDA contexts, external allocations,
other processes, fragmentation and kernel workspaces can consume additional
memory. It can also make PyTorch raise OOM earlier. The default leaves it unset.

If the paged optimizer is incompatible with the installed runtime, explicitly
choose `--optimizer adamw_torch`; NF4 still requires working bitsandbytes. If the
model-specific module or collator audits fail, retain the error and stop; do not
disable those checks or silently switch to text-only training. Fixed seeds do
not guarantee bitwise reproducibility across CUDA/hardware/library versions.

## Files prepared

Created: `src/trainer/sft/train/{__init__.py,train_qlora.py,runtime.py,checkpoints.py}`,
`requirements-sft.txt`, `requirements-sft-constraints.txt`,
`tests/test_sft_training.py`, and this report.

Modified: `src/trainer/sft/train_lora.py` (compatibility wrapper),
`src/trainer/sft/config.py`, `src/trainer/sft/training_data.py`,
`src/trainer/sft/annotation/validation.py`, `src/config/models.py`,
`scripts/prepare_colab_dataset.py` (packaging references), and `.gitignore`.
`requirements-colab.txt` adds the shared validator's two CPU dependencies.

FastAPI/UI, reimbursement rules, Milvus policies, LangGraph workflow, original
invoices, synthetic images/labels/manifests, existing adapters and existing tests
were not edited in this phase. No commit or push was performed. The prior Phase
3/4 uncommitted changes were retained. Wait for the user's manual run results
before any subsequent phase.
