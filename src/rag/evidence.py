"""Bounded Milvus evidence grounded in the unchanged canonical policy JSON."""
import math
import time

from src.agents.contracts import AgentSafetyError, reject_instructions
from src.rag.indexer import source_documents
from src.rag.resolver import PolicyRuleResolver


def retrieve_evidence(invoice, policy, retriever, *, top_k=5, min_score=None, deadline=None, timeout=10):
    documents = {doc["rule_id"]: doc for doc in source_documents(policy)}
    resolver = PolicyRuleResolver(policy)
    evidence, resolutions = {}, {}
    for index, line in enumerate(invoice.lines):
        reject_instructions(line.item)
        hits = {}
        for kind in ("coverage", "exclusion"):
            options = {}
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Policy retrieval request deadline exceeded")
                options["timeout"] = min(timeout, remaining)
            for hit in retriever.search(line.item, top_k=top_k, kind=kind, **options):
                rule_id = hit.get("rule_id")
                doc = documents.get(rule_id)
                if doc is None:
                    raise AgentSafetyError("Retrieved policy ID is not canonical")
                if doc["kind"] != kind:
                    raise AgentSafetyError("Retrieved policy kind does not match the filtered search")
                for field in ("kind", "category", "conditions", "limit_type", "unit", "source", "text"):
                    if field in hit and hit[field] not in (None, "") and hit[field] != doc[field]:
                        raise AgentSafetyError("Index policy metadata differs from policy.json; rebuild the index manually")
                cap = hit.get("maximum_reimbursement")
                if cap is not None and str(cap) != doc["max_amount"]:
                    from decimal import Decimal
                    if Decimal(str(cap)) != Decimal(doc["max_amount"]):
                        raise AgentSafetyError("Index policy cap differs from policy.json")
                score = float(hit.get("score", 0))
                if not math.isfinite(score):
                    raise AgentSafetyError("Nonfinite policy similarity score")
                if min_score is not None and score < min_score:
                    continue
                reject_instructions(doc["text"])
                if rule_id not in hits or score > hits[rule_id]["score"]:
                    hits[rule_id] = {**doc, "score": score, "origin": "milvus"}
        resolution = resolver.resolve(line.item, list(hits.values()))
        rule_id = resolution.get("rule_id")
        # Canonical exact resolution is allowed only after a successful search
        # with some actual evidence. Empty search results stay explicitly pending.
        selected = []
        if hits and rule_id:
            selected.append(hits.get(rule_id) or {**documents[rule_id], "score": None,
                                                 "origin": "canonical_exact_resolution"})
        for kind in ("exclusion", "coverage"):
            ranked = sorted((hit for hit in hits.values() if hit["kind"] == kind),
                            key=lambda hit: (-hit["score"], hit["rule_id"]))
            for hit in ranked:
                if hit["rule_id"] not in {entry["rule_id"] for entry in selected}:
                    selected.append(hit)
                    break
        evidence[index] = selected[:3]
        resolutions[index] = resolution if selected else {
            "item": line.item, "status": "manual_review", "rule_id": None,
            "review_reason": "Missing verified policy evidence"}
    return evidence, resolutions


def compact_evidence(evidence):
    """Deduplicate only relevant facts for the small Qwen context window."""
    return {entry["rule_id"]: {key: entry[key] for key in ("kind", "category", "conditions")}
            for entries in evidence.values() for entry in entries}
