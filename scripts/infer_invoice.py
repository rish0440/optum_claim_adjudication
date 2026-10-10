"""Manual Phase 6 inference or CSV validation; never loads a local model."""
import argparse
import json
import os
from pathlib import Path

from src.inference.vlm_client import VLMClient, VLMConfig, validate_csv
from src.inference.remote_client import RemoteVLMError
from src.utils.paths import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--validate-csv", type=Path)
    parser.add_argument("--mode", choices=("base", "adapter"))
    parser.add_argument("--endpoint", default=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("VLM_MODEL"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-bytes", type=int, default=8*1024*1024)
    parser.add_argument("--max-input-pixels", type=int, default=20_000_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.image and not args.mode:
        parser.error("--mode base or adapter is required for inference")
    try:
        if args.validate_csv:
            text, rows = validate_csv(project_path(args.validate_csv).read_text(encoding="utf-8"))
            print(text, end="", flush=True)
            print(f"CSV validation PASS: {len(rows)} rows", flush=True)
        else:
            config = VLMConfig(mode=args.mode, model=args.model, endpoint=args.endpoint,
                               timeout=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature,
                               top_p=args.top_p, seed=args.seed, api_key=os.getenv("VLLM_API_KEY", ""),
                               max_image_bytes=args.max_image_bytes, max_input_pixels=args.max_input_pixels)
            result = VLMClient(config).extract(project_path(args.image), args.output_dir)
            print(result["csv_text"], end="", flush=True)
            print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "csv_text"}}, indent=2), flush=True)
    except (RemoteVLMError, ValueError, OSError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
