# Model artifacts

Phase 3 trains on Colab and saves a PEFT/QLoRA adapter and processor under
`models/qwen25-vl-invoice-lora/adapter/` inside Colab. A successful run also
creates `run_config.json`, `training_result.json`, and `training_log.json`.

The base checkpoint is `Qwen/Qwen2.5-VL-3B-Instruct` (downloaded separately).
The adapter is NOT bundled in this repository because it has not yet been
trained and does not exist. Persist it to Google Drive after training.

Phase 4 will load the adapter into a GPU inference endpoint, while FastAPI,
LangGraph and policy handling run on the local CPU-only laptop.
