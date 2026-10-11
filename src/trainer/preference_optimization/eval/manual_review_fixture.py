"""One new deterministic synthetic smoke case, never a preference dataset row."""
import copy
from decimal import Decimal

from src.adjudication.policy import load_policy
from src.agents.contracts import FactSelections
from src.agents.reasoning_reviewer import ReasoningReviewAgent
from src.trainer.preference_optimization.annotation.cases import (
    AnnotationClient, agent1, calculate, compact, fingerprint, prepare_case,
)
from src.utils.paths import ROOT

CASE_ID = "agent2-new-manual-review-smoke-20240617"


def load_manual_review_fixture():
    """Build prompt/labels through existing policy, Decimal and review code.

    No treatment record, prescription or clinical link is supplied. COV-005
    is conditional coverage, not an exclusion or proof of medical necessity.
    The production handoff must therefore leave the medicine amount pending.
    """
    policy = load_policy(ROOT/"data/policy.json")
    spec = {"case_id": CASE_ID, "family": "new-smoke:medical-relationship-unverified",
        "rule_id": "COV-005", "mode": "canonical", "prior_disagreement": False,
        "split": "standalone_smoke", "date": "2024-06-17", "item": "Medicines",
        "lines": [{"item": "Medicines", "quantity": 3, "rate": None, "amount": "1475.25"}]}
    invoice, evidence, resolutions = prepare_case(spec, policy)
    _, suggestions, metadata = agent1(invoice, evidence, resolutions, policy)
    claim = calculate(invoice, resolutions, suggestions, metadata, policy)
    rule = next((r for r in policy.coverage_rules if r.rule_id == "COV-005"), None)
    if (rule is None or resolutions[0].get("rule_id") != rule.rule_id or
            not policy.effective_from <= invoice.invoice_date <= policy.effective_to or
            suggestions.items[0].policy_ids != [rule.rule_id] or not suggestions.items[0].needs_review or
            claim.decisions[0].status != "manual_review" or
            claim.pending_review_total != Decimal("1475.25") or claim.final_reimbursement is not None):
        raise ValueError("Canonical rules no longer establish the intended medical-evidence review fixture")
    before = claim.model_dump(mode="json")
    capture = AnnotationClient()
    ReasoningReviewAgent(capture).review(invoice, evidence, suggestions, claim,
        deadline=float("inf"), policy=policy, agent_metadata=metadata)
    expected = FactSelections.model_validate(capture.output.model_dump())
    if (len(expected.items) != 1 or expected.items[0].status != "manual_review" or
            set(expected.items[0].fact_indices) != {0, 1} or expected.items[0].concerns != [0] or
            capture.data["items"][0]["concern_facts"] != [{"index": 0, "kind": "POLICY_CONTEXT_MISSING"}] or
            claim.model_dump(mode="json") != before):
        raise ValueError("Unexpected canonical fact/concern mapping or changed calculator result")
    # The expected completion is kept ONLY for scoring, never in model messages.
    record = {"prompt": copy.deepcopy(capture.messages),
              "chosen": [{"role": "assistant", "content": compact(expected.model_dump())}]}
    annotation = {"pair_id": CASE_ID+"-agent_2", "agent_role": "agent_2",
        "scenario_family": spec["family"], "scenario": "missing_medical_support",
        "source_policy_ids": [rule.rule_id], "case": spec, "deterministic_claim": before}
    audit = {"split": "standalone_smoke", "pairs": 1, "case_id": CASE_ID,
        "case": spec, "canonical_rule_id": rule.rule_id, "canonical_conditions": rule.conditions,
        "expected_references": expected.model_dump(), "deterministic_claim": before,
        "fixture_sha256": fingerprint({"spec": spec, "prompt": record["prompt"], "expected": expected.model_dump()}),
        "evaluation_kind": "new_non_dataset_manual_review_smoke",
        "independent_generalization_evidence": False, "human_verified": False}
    return [(record, annotation)], audit
