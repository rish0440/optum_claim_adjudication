"""Phase 8 LangGraph: sequential claim processing with safe failure branches.

One-shot workflow only; no conversational agent, LLM router or memory store.
Errors bypass downstream processing and produce a structured error response.
"""
from langgraph.graph import END, START, StateGraph

from src.agentic_workflow.nodes.adjudication.node import calculate_claim
from src.agentic_workflow.nodes.csv_validation.node import validate_csv
from src.agentic_workflow.nodes.invoice_extraction.node import extract_invoice
from src.agentic_workflow.nodes.policy_retrieval.node import retrieve_policy
from src.agentic_workflow.nodes.response_builder.node import build_response
from src.agentic_workflow.nodes.rule_resolution.node import resolve_policy_rules
from src.agentic_workflow.state import ClaimState


def _route_after(success: str):
    """Create a deterministic router, keeping failure edges explicit."""
    def choose_next(state: ClaimState) -> str:
        return "response_builder" if state.get("errors") else success
    return choose_next


def create_graph(*, remote_client=None, retriever=None, policy=None, resolver=None):
    """Compile real LangGraph nodes with injectable integrations for tests."""
    from functools import partial

    builder = StateGraph(ClaimState)
    builder.add_node("invoice_extraction", partial(extract_invoice, remote_client=remote_client))
    builder.add_node("csv_validation", validate_csv)
    builder.add_node("policy_retrieval", partial(retrieve_policy, retriever=retriever))
    builder.add_node("rule_resolution", partial(resolve_policy_rules, policy=policy, resolver=resolver))
    builder.add_node("adjudication", partial(calculate_claim, policy=policy))
    builder.add_node("response_builder", build_response)
    builder.add_edge(START, "invoice_extraction")
    for name, success in (
        ("invoice_extraction", "csv_validation"),
        ("csv_validation", "policy_retrieval"),
        ("policy_retrieval", "rule_resolution"),
        ("rule_resolution", "adjudication"),
    ):
        builder.add_conditional_edges(
            name, _route_after(success),
            {success: success, "response_builder": "response_builder"},
        )
    builder.add_edge("adjudication", "response_builder")
    builder.add_edge("response_builder", END)
    return builder.compile()
