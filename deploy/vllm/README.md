# Phase 5: Remote vLLM serving (not DPO or LangGraph)

Use a **Linux NVIDIA GPU instance**, preferably >=24GB VRAM for comfortable
Qwen2.5-VL-3B serving. The Colab T4 might be memory-constrained for vLLM;
training in 4-bit with bitsandbytes does **not** mean the served model is
automatically 4-bit. Do not install vLLM in your Windows `.venv`.

Official references:
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- https://docs.vllm.ai/en/latest/cli/serve/
- https://docs.vllm.ai/en/stable/features/lora/
- https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/

## 1. Transfer your actual adapter from Google Drive

In Colab, the saved training output is typically:

```
models/qwen25-vl-invoice-lora/adapter/
  adapter_config.json
  adapter_model.safetensors
  ...processor files...
```

Copy the **adapter directory** to the GPU server's
`~/models/qwen25-vl-invoice-lora/adapter/` (e.g. download from Drive, then
upload via `scp -r`). Do **not** overwrite with untrained placeholder files.
Verify `adapter_config.json` contains the intended base model and LoRA rank 8.

Copy only the serving script (not the five original invoice images) from your
laptop to the GPU machine, for example from Windows PowerShell:

```powershell
scp deploy/vllm/start_vllm.sh YOUR_GPU_USER@YOUR_GPU_HOST:~/start_vllm.sh
```

## 2. GPU machine installation (Linux shell)

Use a new Python virtual environment on the remote server with a compatible
CUDA GPU driver and PyTorch/vLLM runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm
nvidia-smi
```

Follow the official vLLM install matrix if preinstalled CUDA/PyTorch conflicts.

## 3. Start the private model server (Linux GPU shell)

```bash
export VLLM_API_KEY='choose-a-long-random-secret'
export ADAPTER_DIR="$HOME/models/qwen25-vl-invoice-lora/adapter"
bash ~/start_vllm.sh
```

The script binds to **127.0.0.1:8001** and serves the adapter under the model
name **`invoice-extractor`**. It does not run on Windows. If your adapter
rank differs from 8, change `--max-lora-rank` appropriately. Stop/close the
GPU instance when not in use.

If model initialization fails due to VRAM: choose a 24GB GPU (e.g. L4/A10),
reduce `--max-model-len`, or verify the vLLM version. Don't claim successful
GPU serving without an actual `/v1/models` and extraction response. The latest
vLLM version may differ from the versions used in Qwen's published examples.

## 4. Private SSH tunnel (local Windows PowerShell, separate terminal)

```powershell
ssh -N -L 8010:127.0.0.1:8001 YOUR_GPU_USER@YOUR_GPU_HOST
```

If the GPU provider requires a nonstandard SSH port, add `-p PORT`.
Keep this terminal open. Use the *same secret* as the GPU environment:

```powershell
$env:VLM_BASE_URL = 'http://127.0.0.1:8010/v1'
$env:VLM_MODEL = 'invoice-extractor'
$env:VLM_API_KEY = 'choose-a-long-random-secret'
python -m scripts.test_remote_vlm --ping
python -m scripts.test_remote_vlm --image data/synthetic/validation/images/invoice_00004.jpg --output reports/generated/sample_vlm.json
```

**Security:** `VLLM_API_KEY` alone is not perimeter security. Some vLLM
endpoints do not enforce this key. Use SSH tunneling, VPN or a properly
restricted authenticated HTTPS reverse proxy; **never open the bare vLLM
port to the public Internet**. Do not transmit real patient invoices outside
an approved environment. The supplied synthetic examples are preferred for
this demo. Keep API keys in environment variables rather than Git.

## 5. Interpretation

- `--ping` confirms the server can be reached and lists served models.
- `--image` validates real image-to-CSV inference and prints latency (ms).
- The 2-step fine-tuned LoRA is a pipeline smoke test, not evidence of
  extraction quality. Before the final demo, run real evaluation and compare
  against the pretrained model on held-out examples.
- The Phase 5 client is not yet connected to FastAPI's upload button or
  LangGraph: that is the separate Phase 8/9 integration work.
