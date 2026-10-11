"""CPU scoring of raw selections against replay-validated canonical labels."""
import math
import re
from collections import Counter
from statistics import mean

from src.agents.contracts import CandidateSelections, FactSelections
from src.trainer.preference_optimization.annotation.validate import strict_json


def sanitize_output(text):
    # Evaluator accepts only registered synthetic cases, never private invoices.
    text = str(text)[:16384]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b(?:hf_|sk-|AIza)[A-Za-z0-9_-]+", "[TOKEN]", text)
    return re.sub(r"(?i)(?:bearer\s+|(?:api[_ -]?key|password|secret)\s*[:=]\s*)[^\s,}\"]+", "[REDACTED]", text)


def preference_score(chosen, rejected, tolerance=1e-6):
    if not all(math.isfinite(value) for value in (chosen, rejected)):
        raise ValueError("Nonfinite conditional log probability")
    margin = chosen - rejected
    return {"chosen_logp": chosen, "rejected_logp": rejected, "margin": margin,
            "outcome": "tied" if abs(margin) <= tolerance else "correct" if margin > 0 else "incorrect"}


def assess_generation(raw, record, annotation, *, truncated=False):
    """No repair and no postprocessed agent fallback. Expected labels were validated.

    Field/set comparisons follow the actual CandidateSelections/FactSelections
    contracts. Facts and conditions are supplied by canonical Python records,
    never inferred from a generated amount or invented citation.
    """
    role = annotation["agent_role"]
    schema = CandidateSelections if role == "agent_1" else FactSelections
    expected = schema.model_validate(strict_json(record["chosen"][0]["content"]))
    context = strict_json(record["prompt"][1]["content"])
    result = {"valid_json": False, "valid_schema": False, "correct_scoped_references": False,
              "eligible_completion": False, "semantic_correct": False, "truncated": truncated,
              "unsupported_reference_count": 0, "wrong_item_reference_count": 0,
              "unsupported_approval_count": 0, "missing_required_concern_count": 0,
              "unsupported_review_count": 0, "items": [], "failures": [],
              "correctness_status": "INCORRECT"}
    try:
        value = strict_json(raw)
        result["valid_json"] = True
    except (ValueError, TypeError):
        result["failures"].append("invalid_json")
        return result
    # Explicit citations/amounts are forbidden in these raw selection schemas.
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        for row in value["items"]:
            if isinstance(row, dict):
                citations = row.get("policy_ids", [])
                result["unsupported_reference_count"] += (len(citations) if isinstance(citations, list) else 1) if citations else 0
                if role == "agent_1" and row.get("suggested_status") == "COVERED":
                    index = row.get("item_index")
                    truth = next((r for r in expected.items if type(index) is int and r.item_index == index), None)
                    if truth is None or truth.suggested_status != "COVERED" or row.get("candidate_index") != truth.candidate_index or row.get("basis") != truth.basis:
                        result["unsupported_approval_count"] += 1
    try:
        output = schema.model_validate(value)
        result["valid_schema"] = True
    except ValueError:
        result["failures"].append("invalid_schema_or_forbidden_fields")
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            for row in value["items"]:
                if isinstance(row, dict) and row.get("candidate_index") is not None and type(row.get("candidate_index")) is not int:
                    result["unsupported_reference_count"] += 1
        return result
    if [r.item_index for r in output.items] != [r.item_index for r in expected.items]:
        result["failures"].append("row_identity_or_order")
        return result
    all_facts = {f["index"] for row in context["items"] for f in row.get("facts", [])}
    all_concerns = {f["index"] for row in context["items"] for f in row.get("concern_facts", [])}
    references_ok = True
    for selected, truth, row in zip(output.items, expected.items, context["items"]):
        details = {"item_index": truth.item_index}
        if role == "agent_1":
            allowed = {c["index"] for c in row["candidates"]}
            scope = selected.candidate_index is None or selected.candidate_index in allowed
            if not scope:
                result["unsupported_reference_count"] += 1
            details.update(classification_correct=selected.suggested_status == truth.suggested_status,
                candidate_correct=selected.candidate_index == truth.candidate_index,
                basis_correct=selected.basis == truth.basis,
                expected_status=truth.suggested_status,
                scoped_reference_correct=scope)
            details["correct"] = all(details[k] for k in ("classification_correct", "candidate_correct", "basis_correct")) and scope
        else:
            facts = {f["index"] for f in row["facts"]}
            concerns = {f["index"] for f in row["concern_facts"]}
            wrong_facts = set(selected.fact_indices)-facts
            wrong_concerns = set(selected.concerns)-concerns
            scope = not wrong_facts and not wrong_concerns
            result["unsupported_reference_count"] += len(wrong_facts)+len(wrong_concerns)
            result["wrong_item_reference_count"] += len(wrong_facts & all_facts)+len(wrong_concerns & all_concerns)
            result["missing_required_concern_count"] += len(set(truth.concerns)-set(selected.concerns))
            facts_correct = len(selected.fact_indices) == len(set(selected.fact_indices)) and set(selected.fact_indices) == set(truth.fact_indices)
            concerns_correct = len(selected.concerns) == len(set(selected.concerns)) and set(selected.concerns) == set(truth.concerns)
            details.update(facts_correct=facts_correct, concerns_correct=concerns_correct,
                status_correct=selected.status == truth.status, expected_status=truth.status,
                scoped_reference_correct=scope)
            details["correct"] = facts_correct and concerns_correct and details["status_correct"] and scope
            result["unsupported_review_count"] += int(not details["correct"])
        references_ok &= scope
        result["items"].append(details)
        if not details["correct"]:
            result["failures"].append(f"unsupported_selection:item_{truth.item_index}")
    result["correct_scoped_references"] = references_ok
    result["eligible_completion"] = references_ok and not truncated
    result["semantic_correct"] = all(row["correct"] for row in result["items"]) and not truncated
    if truncated:
        result["failures"].append("generation_did_not_reach_eos")
    result["correctness_status"] = "CORRECT" if result["semantic_correct"] else "INCORRECT"
    return result


def rate(successes, total):
    return {"successes": successes, "denominator": total,
            "percent": 100*successes/total if total else None}


def summarize(predictions, scores):
    """Missing/invalid answers remain in denominators; no invented zero-sample rate."""
    result = {"samples": len(predictions)}
    for field in ("valid_json", "valid_schema", "correct_scoped_references", "eligible_completion", "semantic_correct"):
        result[field] = rate(sum(p["assessment"][field] for p in predictions), len(predictions))
    for field in ("unsupported_reference_count", "wrong_item_reference_count", "unsupported_approval_count",
                  "missing_required_concern_count", "unsupported_review_count"):
        result[field] = sum(p["assessment"][field] for p in predictions)
    for key in ("classification_correct", "candidate_correct", "basis_correct", "facts_correct", "concerns_correct", "status_correct"):
        eligible = [(p, e) for p in predictions for e in p["expected_items"]
                    if (p["agent_role"] == "agent_1") == (key in {"classification_correct", "candidate_correct", "basis_correct"})]
        successes = sum(next((row.get(key, False) for row in p["assessment"]["items"] if row["item_index"] == e["item_index"]), False) for p, e in eligible)
        result[key] = rate(successes, len(eligible))
    for name, status in (("coverage_selection", "COVERED"), ("exclusion_identification", "EXCLUDED"), ("appropriate_review", "REVIEW_REQUIRED")):
        eligible = [(p, e) for p in predictions if p["agent_role"] == "agent_1" for e in p["expected_items"] if e["suggested_status"] == status]
        successes = sum(next((r["correct"] for r in p["assessment"]["items"] if r["item_index"] == e["item_index"]), False) for p, e in eligible)
        result[name] = rate(successes, len(eligible))
    agent1 = [p for p in predictions if p["agent_role"] == "agent_1"]
    result["unsupported_approval_rate"] = rate(sum(p["assessment"]["unsupported_approval_count"] > 0 for p in agent1), len(agent1))
    result["wrong_item_evidence_rate"] = rate(sum(p["assessment"]["wrong_item_reference_count"] > 0 for p in predictions if p["agent_role"] == "agent_2"), sum(p["agent_role"] == "agent_2" for p in predictions))
    outcomes = Counter(s["outcome"] for s in scores)
    result["preference"] = {"correct": outcomes["correct"], "incorrect": outcomes["incorrect"],
        "tied": outcomes["tied"], "accuracy": rate(outcomes["correct"], len(scores)),
        "mean_margin": mean(s["margin"] for s in scores) if scores else None,
        "failures": len(predictions)-len(scores), "definition": "raw summed assistant-completion logp; ties are not correct; NOT DPO reward accuracy"}
    result["generation_failures"] = sum(not p["assessment"]["semantic_correct"] for p in predictions)
    result["average_generation_seconds"] = mean(p["generation_seconds"] for p in predictions) if predictions else None
    return result


def aggregate(predictions, pairwise):
    metrics = {}
    for model in ("baseline", "dpo"):
        rows = predictions[model]
        scores = [s[model] for s in pairwise]
        metrics[model] = {"overall": summarize(rows, scores), "by_role": {}, "by_scenario": {}}
        for group, field in (("by_role", "agent_role"), ("by_scenario", "scenario_family")):
            for value in sorted({r[field] for r in rows}):
                subset = [r for r in rows if r[field] == value]
                ids = {r["pair_id"] for r in subset}
                metrics[model][group][value] = summarize(subset, [s[model] for s in pairwise if s["pair_id"] in ids])
    return metrics
