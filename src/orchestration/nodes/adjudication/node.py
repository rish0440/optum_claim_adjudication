"""Apply Phase 7's Decimal-based calculator to the already resolved line items."""
from src.adjudication.calculator import adjudicate_invoice
from src.orchestration.state import ClaimState
from src.adjudication.policy import load_policy


def calculate_claim(state: ClaimState, *, policy=None) -> dict:
    """Financial calculations use policy.json, never embedding scores or LLM output."""
    trace = [*state.get("trace", []), "adjudication"]
    try:
        result = adjudicate_invoice(state["invoice"], policy=policy or load_policy(),
                                    resolved_pairs=state["resolved_pairs"])
        return {"claim_result": result, "trace": trace}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        return {"errors": [*state.get("errors", []), str(exc)],
                "error_stage": "adjudication", "trace": trace}
