"""Compiled graphs for the preserved pipeline and Phase 7 two-agent workflow."""
from langgraph.graph import END, START, StateGraph

from src.components.agentic_workflow.nodes.adjudication.node import calculate_claim
from src.components.agentic_workflow.nodes.csv_validation.node import validate_csv
from src.components.agentic_workflow.nodes.invoice_extraction.node import extract_invoice
from src.components.agentic_workflow.nodes.policy_retrieval.node import retrieve_policy
from src.components.agentic_workflow.nodes.response_builder.node import build_response
from src.components.agentic_workflow.nodes.rule_resolution.node import resolve_policy_rules
from src.components.agentic_workflow.state import ClaimState


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


def create_claim_graph(*, endpoint="http://127.0.0.1:8000/v1", policy=None,
                       retriever=None, text_client=None, extraction_client=None,
                       timeout=120, max_tokens=768, checkpoint=False, verbose=False):
    """Compile the two-agent workflow. The existing create_graph API is preserved."""
    from src.utils.policy import load_policy
    from src.components.agentic_workflow.nodes.copilot import ClaimNodes
    from src.components.agentic_workflow.routers import route_input, after_step, after_retrieval, after_calculation, after_review
    nodes = ClaimNodes(policy=policy or load_policy(), endpoint=endpoint, retriever=retriever,
        text_client=text_client, extraction_client=extraction_client, timeout=timeout,
        max_tokens=max_tokens, verbose=verbose)
    builder = StateGraph(ClaimState)
    for name in ("orchestrator", "invoice_extraction", "csv_validation", "policy_retrieval",
                 "policy_adjudication_agent", "deterministic_adjudication", "reasoning_review_agent", "final_response"):
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_input, {
        name: name for name in ("invoice_extraction", "csv_validation", "final_response")})
    builder.add_conditional_edges("invoice_extraction", after_step("csv_validation"), {
        name: name for name in ("invoice_extraction", "csv_validation", "final_response")})
    builder.add_conditional_edges("csv_validation", after_step("policy_retrieval"), {
        name: name for name in ("invoice_extraction", "policy_retrieval", "final_response")})
    builder.add_conditional_edges("policy_retrieval", after_retrieval, {
        name: name for name in ("policy_adjudication_agent", "deterministic_adjudication", "final_response")})
    builder.add_conditional_edges("policy_adjudication_agent", after_step("deterministic_adjudication"), {
        name: name for name in ("policy_adjudication_agent", "deterministic_adjudication", "final_response")})
    builder.add_conditional_edges("deterministic_adjudication", after_calculation, {
        name: name for name in ("reasoning_review_agent", "final_response")})
    builder.add_conditional_edges("reasoning_review_agent", after_review, {
        name: name for name in ("reasoning_review_agent", "final_response")})
    builder.add_edge("final_response", END)
    checkpointer = None
    if checkpoint:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
