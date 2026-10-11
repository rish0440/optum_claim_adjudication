"""Manually generate offline canonical DPO pairs; never invoke model inference."""
import argparse
import csv
import hashlib
import json

from src.adjudication.policy import load_policy
from src.inference.vlm_client import BASE_MODEL_ID
from src.utils.paths import ROOT, project_path
from .cases import build_pairs, compact, source_hashes, SEED, SPLITS
from .validate import validate_record, validate_records


def generate_dataset(directory="data/dpo", seed=SEED):
    policy = load_policy(ROOT/"data/policy.json")
    pairs, flagged = build_pairs(policy, seed)
    for record, annotation in pairs:
        annotation["generation_seed"] = seed
        annotation["validation_status"] = validate_record(record, annotation, policy)
    counts = validate_records(pairs, policy)
    directory = project_path(directory)
    names = [f"{split}.jsonl" for split in SPLITS] + ["manifest.json", "annotation_review.csv"]
    if any((directory/name).exists() for name in names):
        raise FileExistsError("Dataset already exists; choose a new --output-dir (existing files are never overwritten)")
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (directory/f"{split}.jsonl").write_text("".join(compact(record)+"\n" for record, a in pairs if a["data_split"] == split), encoding="utf-8")
    manifest = {"schema_version": 1, "base_model_id": BASE_MODEL_ID, "served_model": "qwen-base",
        "model_verification": "existing serving configuration and user-verified deployment; no server queried",
        "chat_template": "Qwen2.5-VL processor/tokenizer chat template; system/user prompt, assistant-only chosen/rejected; do not pre-render or append EOS manually",
        "generation_seed": seed, "target_pairs": 80, "target_splits": {"train": 60, "validation": 10, "test": 10},
        "pair_count": len(pairs), "counts_by_split_role": counts, "policy_source": "data/policy.json",
        "policy_sha256": hashlib.sha256((ROOT/"data/policy.json").read_bytes()).hexdigest(),
        "source_sha256": source_hashes(), "human_verified": False,
        "retrieval_provenance": "synthetic canonical retrieval fixtures; no Milvus query or embedding",
        "flagged_cases": flagged, "annotations": [a for _, a in pairs]}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    fields = ("pair_id", "role", "scenario", "preferred_response_summary", "rejected_response_summary",
              "why_chosen_is_preferred", "applicable_canonical_policy", "validation_status", "human_verified")
    with (directory/"annotation_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record, a in pairs:
            writer.writerow(dict(pair_id=a["pair_id"], role=a["agent_role"], scenario=a["scenario_family"]+" / "+a["scenario"],
                preferred_response_summary=record["chosen"][0]["content"], rejected_response_summary=record["rejected"][0]["content"],
                why_chosen_is_preferred=a["chosen_preference_reason"]+" Rejected error: "+a["rejected_error_type"],
                applicable_canonical_policy=", ".join(sorted({row["rule_id"] for row in a["deterministic_claim"]["decisions"] if row["rule_id"]}
                    | ({a["case"]["rule_id"]} if a["case"]["rule_id"] and a["case"]["mode"] not in {"missing", "date_outside"} else set()))),
                validation_status=a["validation_status"], human_verified=False))
    return {"pairs": len(pairs), "counts_by_split_role": counts, "flagged_cases": flagged,
            "human_verified": False, "output_dir": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/dpo")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
