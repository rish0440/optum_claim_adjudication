"""Deterministic supervisor routes with a request-wide bounded retry budget."""


def route_input(state):
    if state.get("workflow_status") == "FAILED":
        return "final_response"
    return "invoice_extraction" if state["input_mode"] == "image" else "csv_validation"


def after_step(success):
    def route(state):
        if state.get("retry_node"):
            return state["retry_node"]
        return "final_response" if state.get("workflow_status") == "FAILED" else success
    return route


def after_retrieval(state):
    if state.get("workflow_status") == "FAILED":
        return "final_response"
    if any(not entries for entries in state.get("candidate_rules", {}).values()):
        return "deterministic_adjudication"
    return "policy_adjudication_agent"


def after_calculation(state):
    if state.get("workflow_status") == "FAILED" or not state.get("agent_suggestions"):
        return "final_response"
    return "reasoning_review_agent"


def after_review(state):
    return state.get("retry_node") or "final_response"
