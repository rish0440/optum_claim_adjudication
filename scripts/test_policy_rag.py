"""Inspect Milvus rule hits and measure retrieval on independent evaluation cases.

Run: python -m scripts.test_policy_rag --item "Blood Test (CBC)"
     python -m scripts.test_policy_rag --evaluate
"""

import argparse
import json
from pathlib import Path

from src.adjudication.policy import load_policy
from src.rag.resolver import PolicyRuleResolver
from src.rag.retriever import PolicyRetriever
from src.config.settings import ROOT, get_settings


def evaluate(retriever, resolver, cases: list[dict], top_k: int = 3) -> dict:
    """Honest Recall@1/3 + resolution accuracy; denominators always explicit."""
    policy = load_policy()
    known = {rule.rule_id for rule in policy.coverage_rules + policy.exclusion_rules}
    if not cases:
        raise ValueError("No evaluation cases supplied")
    counters = {"recall_at_1": 0, "recall_at_3": 0,
                "resolved_correct": 0, "grounded": 0}
    results = []
    for case in cases:
        expected = case["expected_rule_id"]
        if expected not in known:
            raise ValueError(f"Evaluation label {expected} is absent from policy.json")
        hits = retriever.search(case["item"], top_k=max(top_k, 3))
        ids = [hit["rule_id"] for hit in hits]
        resolved = resolver.resolve(case["item"], hits)
        counters["recall_at_1"] += int(expected in ids[:1])
        counters["recall_at_3"] += int(expected in ids[:3])
        counters["resolved_correct"] += int(resolved["rule_id"] == expected)
        grounded = (resolved["rule_id"] is not None and
                    resolved["rule_id"] in known and
                    resolved["conditions"] == resolver.by_id[resolved["rule_id"]].conditions)
        counters["grounded"] += int(grounded)
        results.append({"item": case["item"], "expected_rule_id": expected,
                        "semantic_top_3": ids[:3], "resolved_rule_id": resolved["rule_id"],
                        "status": resolved["status"], "grounded": grounded})
    count = len(cases)
    metrics = {name: {"correct": value, "total": count,
                      "fraction": round(value / count, 4)} for name, value in counters.items()}
    return {"dataset": "independent_policy_queries_not_optum_invoice_images",
            "cases": count, "metrics": metrics, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--item", help="One invoice line item to search")
    group.add_argument("--evaluate", action="store_true", help="Run curated retrieval queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    retriever = PolicyRetriever()
    resolver = PolicyRuleResolver()
    if args.evaluate:
        path = ROOT / "data" / "evaluation" / "policy_rag_cases.json"
        cases = json.loads(path.read_text(encoding="utf-8"))
        report = evaluate(retriever, resolver, cases, top_k=args.top_k)
        for name, metric in report["metrics"].items():
            print(f"{name}: {metric['correct']}/{metric['total']} ({metric['fraction']:.1%})")
    else:
        hits = retriever.search(args.item, top_k=args.top_k)
        report = resolver.resolve(args.item, hits)
        print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
