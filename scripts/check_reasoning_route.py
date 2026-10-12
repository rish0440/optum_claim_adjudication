"""Manual Phase 9D route check; network access only with --check-server."""
import argparse
import json
import time

from src.components.inference.reasoning_client import TextReasoningClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-server", action="store_true")
    args = parser.parse_args()
    client = TextReasoningClient()
    report = client.route.diagnostics()
    report["server_identity_checked"] = False
    if args.check_server:
        client.preflight(deadline=time.monotonic() + 30)
        report["server_identity_checked"] = True
    report["gpu_execution_verified"] = False
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
