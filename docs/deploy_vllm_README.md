# Phase 6: manual vLLM image-to-CSV serving

Code preparation only. No project code, tests, installations, model downloads,
services, GPU checks or inference were executed. Source/configuration/metadata
were read; prepared code has not been runtime verified.

## Environment blocker: preserve the existing training stack

Read-only package metadata shows Torch 2.14.1 (Phase 5 build: +cu130),
TorchVision 0.29.1+cu130, Transformers 4.57.1, TRL 0.24.0, PEFT 0.17.1 and
httpx 0.28.1. No vLLM distribution was found. The successful Phase 5 pointer
exists and its adapter configuration declares the correct base ID, rank 8 and
q/k/v/o language-attention targets. No weights were loaded or verified here.

This launcher uses the documented **vLLM 0.11.2** CLI/model implementation as
its reference. That implementation supports Qwen2.5-VL LoRA and explicitly maps
`model.language_model.*` checkpoint names to `language_model.model.*`. Its
Transformers requirement includes 4.57.1. However, its published CUDA requirements
pin **Torch 2.9.0 and TorchVision 0.24.0**, conflicting with this VM. Consequently,
an ordinary wheel installation is **not established as compatible** with the
single existing environment. Do not force `--no-deps`, downgrade Torch, or bulk
upgrade the training environment. A later release is not automatically safer:
vLLM 0.31.0 source requirements also conflict (Torch 2.13.0 and Transformers >=5.10.4).

Official references:
- [0.11.2 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.2/requirements/cuda.txt)
- [0.11.2 common requirements](https://github.com/vllm-project/vllm/blob/v0.11.2/requirements/common.txt)
- [0.11.2 Qwen implementation and mapper](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/models/qwen2_5_vl.py)
- [0.11.2 supported server arguments](https://docs.vllm.ai/en/v0.11.2/cli/serve/)
- [Installation with existing PyTorch](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

A source build against existing Torch is an upstream option, **not a verified
solution for this exact Torch/Transformers/CUDA combination**. It would require
manual build-toolkit/ABI/kernel validation and compatible transitive dependencies.
No build commands or dependency replacements are performed automatically. A
verified custom build or separately authorized dependency reconciliation is
needed before the server can be certified to run. Other vLLM versions are
rejected until their arguments, multimodal LoRA and mapping are reviewed.

## Manual commands

Run these yourself. Use the existing `.venv`; stop training before serving.

```bash
cd /home/rishabh_mishra0440/optum_claim_adjudication
source .venv/bin/activate
cd optum_claim_adjudication
python -m pip show vllm torch torchvision transformers trl peft
python -m pip check
python -u -m src.components.inference.vllm_server --check
```

The check inspects installed metadata/dependencies and the Qwen source mapper;
it does not start vLLM, import its CUDA extensions, or load a model. It currently
reports BLOCKED because vLLM is missing. Even a metadata PASS would not certify
CUDA kernels, memory usage or LoRA loading.

### Installation planning only

```bash
python -m pip install --dry-run \
  -c src/components/deploy/vllm/constraints.txt vllm==0.11.2
```

This is a resolver preview, **not an installation**. It is expected to report
the Torch/TorchVision conflict. Do not remove the constraints to force success.
There is no certified drop-in installation command for the observed environment.
Share the resolver output or details of a verified custom build before changing
packages. No separate FlashAttention install is prescribed. vLLM may pull its
own serving dependencies, including the OpenAI Python SDK; no OpenAI API key or
external OpenAI service is used by this client.

### 1. Base-model server, foreground terminal

After resolving compatibility:

```bash
bash src/components/deploy/vllm/start_vlm.sh base
```

Default server arguments: model `Qwen/Qwen2.5-VL-3B-Instruct`, loopback
`127.0.0.1:8000`, served name `qwen-base`, tensor parallelism 1, BF16, GPU memory
utilization 0.75, context 2048, one concurrent sequence, one image/no videos,
pixel bounds 50176–200704, eager execution. The model is loaded normally for
serving; the NF4 training base is not merged, exported or automatically reused.
If BF16 is unsupported, explicitly use `--dtype half`; never silently change it.

Options can be passed directly, for example:

```bash
bash src/components/deploy/vllm/start_vlm.sh base \
  --gpu-memory-utilization 0.75 --max-model-len 2048 \
  --max-pixels 200704 --image-limit 1
```

`VLLM_PORT`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS` and
`VLLM_GPU_MEMORY_UTILIZATION` also configure defaults. The host cannot be changed
from loopback through this launcher. It uses `../.venv/bin/python` relative to
the actual Git root; `VLM_PYTHON` can select that same existing interpreter if
the checkout is relocated. The old `start_vllm.sh` delegates to this launcher.

### 2. Health/models and real base inference, second terminal

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
python -u -m scripts.infer_invoice \
  --mode base --model qwen-base \
  --image data/assignment_dataset/synthetic/validation/images/invoice_00004.jpg \
  --output-dir reports/inference/base
```

The client prints validated CSV and a validation/artifact summary. It sends
the original image as a base64 JPEG/PNG data URL, uses the exact training
instruction, temperature 0/top-p 1/seed 42, and saves the actual response. It
does not read labels or substitute expected CSV. Source images are limited to
8 MiB/20 million pixels; server preprocessing uses the smaller configured pixel
budget. `--endpoint`, `--timeout`, `--max-tokens`, generation settings, image limits
and output directory are configurable. `VLM_BASE_URL` and `VLM_MODEL` are optional.

### 3. Stop base server, then launch the adapter

Press **Ctrl-C in the foreground server terminal**, wait for shutdown, and inspect
`nvidia-smi` before starting the next server. No background monitor or kill script
is launched automatically.

```bash
bash src/components/deploy/vllm/start_vlm.sh adapter
```

Adapter mode resolves `src.config.models.sft_adapter_pointer()` and reuses the
Phase 5 checkpoint verifier: successful run, expected base model, real tensor
files/shapes/checksum, conventional rank-8 LoRA. It uses the saved chat template
and training pixel limits unless explicitly overridden. It passes documented
`--enable-lora --max-loras 1 --max-lora-rank 8` and a JSON `--lora-modules` entry.
The adapter is served as `invoice-extractor`. Any pointer/configuration, dependency
or adapter-loading failure stops startup; there is no retry without the adapter.
The shared `OPTUM_SFT_OUTPUT_DIR` override is honored. Weights are not altered.

### 4. Mandatory adapter inference on the same validation image

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
python -u -m scripts.infer_invoice \
  --mode adapter --model invoice-extractor \
  --image data/assignment_dataset/synthetic/validation/images/invoice_00004.jpg \
  --output-dir reports/inference/adapter
```

The client verifies the requested model is listed, its adapter/base relationship
matches Qwen, and the response model matches the requested name. It rejects
missing adapters, base aliases disguised as adapters, truncated generations,
malformed CSV, invalid numbers and inconsistent arithmetic. Empty Rate values
remain empty. Base-only success is insufficient for final verification.

### 5. Revalidate saved CSVs and inspect failures

```bash
for csv_path in reports/inference/base/*/extracted.csv reports/inference/adapter/*/extracted.csv; do
  [[ -f "$csv_path" ]] || continue
  python -u -m scripts.infer_invoice --validate-csv "$csv_path" || break
done
```

Validation prints CSV and PASS/FAIL. A failed inference returns nonzero and prints
the error plus diagnostic directory. Inspect that directory's `validation.json`,
`response.json` and `raw_output.txt`; `extracted.csv` exists only on success.
Structural validation does not prove OCR correctness or preservation of every
visible row; manually compare the smoke image with both outputs. No accuracy
claim is made before real inference/evaluation.

### GPU monitoring, separate terminal

```bash
watch -n 1 nvidia-smi
```

Keep training stopped. For OOM, reduce `--max-model-len` and `--max-pixels`
(for example 100352); concurrency is already 1 and eager execution avoids CUDA
graph allocations. GPU memory utilization is an engine allocation target, not
an absolute cap on all VRAM use. Do not expose port 8000 publicly.

## Saved inference artifacts

```text
reports/inference/
├── base/<UTC timestamp>-<unique suffix>/
└── adapter/<UTC timestamp>-<unique suffix>/
    ├── response.json       # Actual HTTP JSON response
    ├── raw_output.txt      # Exact assistant text, including malformed output
    ├── extracted.csv       # Only when validation succeeds
    └── validation.json     # PASS/FAIL, model metadata, rows/timing or error
```

Unique directories preserve previous outputs. Connection/model-list failures
produce `validation.json` without response/text/CSV files. Missing assistant text
retains `response.json`. Markdown fences can be stripped using the existing
parser; no missing fields, rates or invoice items are invented.

The optional `VLLM_API_KEY` is a service-local token, not an OpenAI credential.
When set on the server, set the same variable in the client terminal and add
`-H "Authorization: Bearer $VLLM_API_KEY"` to the model-list curl command.
The existing authenticated remote client retains its default key requirement;
only the new local mode permits missing authentication on loopback.

## Written tests, not executed

`tests/test_vllm_inference.py` covers verified-pointer reuse/failure, real-image
request construction and limits, CSV escaping, empty Rate, malformed/numerically
invalid CSV, HTTP failure, response artifacts, base/adapter selection, advertised
adapter metadata, truncated output, and loopback launcher modes. HTTP is mocked;
no service, model, GPU or invoice dataset is used.

If you choose to run them later:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m unittest discover -s tests \
  -p 'test_vllm_inference.py' -v
```

No QLoRA training code, FastAPI/UI, Milvus/policy logic, LangGraph workflow,
data, or weights were modified. No commit/push was performed. Serving, adapter
loading, dependency imports, output accuracy, and memory usage remain unverified.
