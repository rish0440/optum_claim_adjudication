"""Phase 5: test GPU model API from local Windows/Linux without running LangGraph.

From project root (after setting VLM_* environment variables):
  python -m scripts.test_remote_vlm --ping
  python -m scripts.test_remote_vlm --image data/synthetic/validation/images/invoice_00004.jpg
  python -m scripts.test_remote_vlm --image sample.jpg --output reports/generated/vlm_sample.json

No ground-truth CSV is sent to the GPU. Results are real model predictions.
"""

import argparse
import json
from pathlib import Path

from src.vlm.remote_client import RemoteVLMClient, RemoteVLMConfig, RemoteVLMError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ping", action="store_true", help="Check auth and model list; no image sent")
    mode.add_argument("--image", type=Path, help="JPEG/PNG/WebP image to extract")
    parser.add_argument("--output", type=Path, help="Save the prediction as JSON")
    args = parser.parse_args()
    try:
        client = RemoteVLMClient(RemoteVLMConfig.from_env())
        if args.ping:
            models = client.available_models()
            print(json.dumps({"connected": True, "models": models}, indent=2))
            if client.config.model not in models:
                print(f"WARNING: served model name {client.config.model!r} not listed; check adapter setup")
            return 0
        result = client.extract(args.image).as_dict()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved actual prediction: {args.output}")
        return 0
    except (RemoteVLMError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Phase 5 check failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
