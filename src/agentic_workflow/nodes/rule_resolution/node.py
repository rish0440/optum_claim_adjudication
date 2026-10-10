"""Resolve the selected policy rule for each row against the actual policy JSON."""
from src.adjudication.rules import resolve_lines
from src.agentic_workflow.state import ClaimState
from src.policy import load_policy
from src.rag.resolver import PolicyRuleResolver


def resolve_policy_rules(state: ClaimState, *, policy=None, resolver=None) -> dict:
    """Do not allow vector similarity alone to approve or exclude claims."""
    trace = [*state.get("trace", []), "rule_resolution"]
    try:
        policy = policy or load_policy()
        resolver = resolver or PolicyRuleResolver(policy)
        pairs = resolve_lines(state["invoice"], policy=policy, resolver=resolver,
                              hits_by_index=state.get("candidate_rules", {}))
        return {"resolved_pairs": pairs, "trace": trace}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        return {"errors": [*state.get("errors", []), str(exc)],
                "error_stage": "rule_resolution", "trace": trace}
