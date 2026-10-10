"""Build/reuse the local Milvus policy rule collection (CPU only).

Run: python -m scripts.build_policy_index
     python -m scripts.build_policy_index --rebuild
"""

import argparse
import json

from src.rag.indexer import build_policy_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Replace existing index")
    args = parser.parse_args()
    print(json.dumps(build_policy_index(rebuild=args.rebuild), indent=2))


if __name__ == "__main__":
    main()
