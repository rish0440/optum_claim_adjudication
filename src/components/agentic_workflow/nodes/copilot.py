"""Request-scoped tools and two-agent nodes; the legacy node package is retained."""
from pathlib import Path
import time

from src.utils.calculator import adjudicate_invoice
from src.components.agentic_workflow.contracts import AgentFormatError, AgentSafetyError, WorkflowReport, reject_instructions
from src.components.agentic_workflow.policy_adjudicator import PolicyAdjudicationAgent
from src.components.agentic_workflow.reasoning_reviewer import ReasoningReviewAgent
from src.config.settings import get_settings
from src.components.inference.reasoning_client import TextReasoningClient
from src.components.inference.remote_client import RemoteVLMError
from src.components.inference.vlm_client import VLMClient, VLMConfig, validate_csv
from src.components.agentic_workflow.nodes.csv_validation.node import validate_csv as validate_invoice
from src.components.rag.evidence import retrieve_evidence


class ClaimNodes:
    def __init__(self, *, policy, endpoint, retriever=None, text_client=None,
                 extraction_client=None, timeout=120, max_tokens=768, verbose=False):
        self.policy, self.endpoint, self.retriever = policy, endpoint, retriever
        self.timeout, self.verbose = timeout, verbose
        self.extraction_client = extraction_client
        client = text_client or TextReasoningClient(endpoint, timeout=timeout, max_tokens=max_tokens)
        self.reasoning_route = client.route.diagnostics() if isinstance(client, TextReasoningClient) else None
        self.adjudicator = PolicyAdjudicationAgent(client)
        self.reviewer = ReasoningReviewAgent(client)

    def execute(self, name, state, action):
        started = time.monotonic()
        transport_failure = False
        try:
            if started >= state["deadline_monotonic"]:
                raise TimeoutError("Claim deadline exceeded")
            update = action(state)
            update.setdefault("retry_node", None)
            outcome = "ok" if update.get("workflow_status") != "REVIEW_REQUIRED" else "review"
        except Exception as exc:
            transport_failure = isinstance(exc, RemoteVLMError)
            recoverable = isinstance(exc, AgentFormatError)
            counters = dict(state["retry_counters"])
            if recoverable and sum(counters.values()) < state["max_retries"]:
                counters[name] = counters.get(name, 0) + 1
                update = {"retry_counters": counters, "retry_node": name,
                          "warnings": [*state["warnings"], f"Retrying {name} once after invalid JSON/schema"]}
                outcome = "retry"
            else:
                safe_detail = str(exc) if isinstance(exc, (AgentSafetyError, AgentFormatError)) else type(exc).__name__
                update = {"retry_node": None, "error_stage": name,
                          "errors": [*state["errors"], f"{name}: {safe_detail}"],
                          "workflow_status": "REVIEW_REQUIRED" if name == "reasoning_review_agent" else "FAILED"}
                outcome = "rejected"
            if name in ("policy_adjudication_agent", "reasoning_review_agent") and hasattr(exc, "diagnostics"):
                key = "agent_1_rejection" if name == "policy_adjudication_agent" else "agent_2_rejection"
                update["metadata"] = {**state["metadata"], key: exc.diagnostics}
            elif name == "reasoning_review_agent":
                update["metadata"] = {**state["metadata"], "agent_2_rejection": {
                    "category": type(exc).__name__, "field": "transport_or_runtime",
                    "item_index": None, "sanitized_response": None}}
        trace = {"node": name, "request_id": state["request_id"], "outcome": outcome,
                 "elapsed_ms": round((time.monotonic()-started)*1000)}
        if self.reasoning_route and name in ("policy_adjudication_agent", "reasoning_review_agent"):
            # This is the requested route; only successful response metadata verifies service.
            trace["requested_reasoning_route"] = self.reasoning_route
            key = "agent_1" if name == "policy_adjudication_agent" else "agent_2"
            served = update.get("metadata", {}).get(key, {}).get("model")
            if served:
                trace["served_model"] = served
            if outcome == "rejected" and transport_failure:
                update["warnings"] = [*update.get("warnings", state["warnings"]),
                                      "Selected reasoning service failed; no alternate model was used"]
        if self.reasoning_route and name == "orchestrator":
            update["model_ids"] = {**state["model_ids"], "agent_1": self.reasoning_route["model"],
                                   "agent_2": self.reasoning_route["model"]}
            update["metadata"] = {**state["metadata"], "reasoning_route": self.reasoning_route}
        update["execution_trace"] = [*state["execution_trace"], trace]
        update["trace"] = [*state["trace"], name]
        if self.verbose:
            print(f"{state['request_id']} {name}: {outcome} ({trace['elapsed_ms']} ms)", flush=True)
            if "requested_reasoning_route" in trace:
                print(f"  reasoning route: {self.reasoning_route['model']} at {self.reasoning_route['endpoint']}", flush=True)
                if "served_model" in trace:
                    print(f"  verified response model: {trace['served_model']}", flush=True)
        return update

    def orchestrator(self, state):
        def action(current):
            if current["input_mode"] == "csv_file":
                path = Path(current["csv_path"])
                if path.stat().st_size > 64 * 1024:
                    raise ValueError("Invoice CSV exceeds input size limit")
                return {"extracted_csv": path.read_text(encoding="utf-8-sig"), "extraction_source": "provided_csv"}
            if current["input_mode"] == "csv_text":
                if len(current["csv_text"].encode("utf-8")) > 64 * 1024:
                    raise ValueError("Invoice CSV exceeds input size limit")
                return {"extracted_csv": current["csv_text"], "extraction_source": "provided_csv"}
            return {}
        return self.execute("orchestrator", state, action)

    def invoice_extraction(self, state):
        def action(current):
            endpoint = self.endpoint.rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            client = self.extraction_client or VLMClient(VLMConfig(
                mode="adapter", endpoint=endpoint,
                timeout=min(self.timeout, current["deadline_monotonic"]-time.monotonic())))
            try:
                extracted = client.extract(current["image_path"])
            except RemoteVLMError as exc:
                if isinstance(exc.__cause__, ValueError):
                    raise AgentFormatError("Image extraction returned invalid CSV") from exc
                raise
            if extracted.get("requested_model") != "invoice-extractor" or extracted.get("returned_model") != "invoice-extractor":
                raise AgentSafetyError("Extraction did not use the verified invoice-extractor adapter")
            return {"extracted_csv": extracted["csv_text"], "extraction_source": "invoice-extractor",
                    "metadata": {**current["metadata"], "extraction": {
                        key: extracted.get(key) for key in ("requested_model", "returned_model", "latency_ms", "usage", "artifacts")}}}
        return self.execute("invoice_extraction", state, action)

    def csv_validation(self, state):
        def action(current):
            try:
                text, rows = validate_csv(current["extracted_csv"])
                if not rows or len(rows) > 50:
                    raise ValueError("Invoice needs between 1 and 50 rows")
            except ValueError as exc:
                if current["input_mode"] == "image":
                    # The retry route returns to extraction, never repairs invoice values.
                    raise AgentFormatError("Invalid extracted invoice CSV") from exc
                raise
            for row in rows:
                reject_instructions(row["Item"])
            validated = validate_invoice({**current, "extracted_csv": text})
            if validated.get("errors"):
                raise ValueError("Invalid invoice data/date")
            return {"invoice": validated["invoice"], "extracted_lines": validated["extracted_lines"],
                    "extracted_csv": text, "csv_validated": True}
        update = self.execute("csv_validation", state, action)
        if update.get("retry_node") == "csv_validation":
            update["retry_node"] = "invoice_extraction"
        return update

    def policy_retrieval(self, state):
        def action(current):
            try:
                if self.retriever is None:
                    from src.components.rag.retriever import PolicyRetriever
                    self.retriever = PolicyRetriever()
                settings = get_settings().rag
                evidence, resolutions = retrieve_evidence(current["invoice"], self.policy, self.retriever,
                    top_k=settings.top_k, min_score=settings.min_score,
                    deadline=current["deadline_monotonic"], timeout=settings.query_timeout)
                missing = any(not entries for entries in evidence.values())
                return {"candidate_rules": evidence, "policy_resolutions": resolutions,
                        "rag_status": "milvus_missing_evidence" if missing else "milvus_live",
                        **({"workflow_status": "REVIEW_REQUIRED", "warnings": [*current["warnings"],
                            "Missing policy evidence; affected rows require review"]} if missing else {})}
            except Exception as exc:
                # No exact-policy approval fallback when the actual RAG service is unavailable.
                return {"candidate_rules": {i: [] for i in range(len(current["invoice"].lines))},
                        "policy_resolutions": {}, "rag_status": "unavailable_or_invalid",
                        "workflow_status": "REVIEW_REQUIRED",
                        "errors": [*current["errors"], f"Policy retrieval failed ({type(exc).__name__})"],
                        "warnings": [*current["warnings"], "Verify the Milvus index; if missing/stale, run manually: "
                                     "python -m scripts.build_policy_index (add --rebuild only for a stale index)"]}
        return self.execute("policy_retrieval", state, action)

    def policy_adjudication_agent(self, state):
        def action(current):
            output, metadata = self.adjudicator.suggest(current["invoice"], current["candidate_rules"],
                current["policy_resolutions"], deadline=current["deadline_monotonic"], policy=self.policy)
            warnings = [*current["warnings"], *[
                f"Agent 1 advisory disagreement on row {index}; independently verified policy retained"
                for index in metadata.get("disagreement_items", [])]]
            return {"agent_suggestions": output, "metadata": {**current["metadata"], "agent_1": metadata},
                    "warnings": warnings}
        return self.execute("policy_adjudication_agent", state, action)

    def deterministic_adjudication(self, state):
        def action(current):
            by_id = {rule.rule_id: rule for rule in self.policy.coverage_rules + self.policy.exclusion_rules}
            pairs = []
            for index, line in enumerate(current["invoice"].lines):
                resolution = current["policy_resolutions"].get(index, {
                    "status": "manual_review", "rule_id": None,
                    "review_reason": "Policy evidence unavailable; no eligibility assumption"})
                # Preserve the calculator's existing Decimal/pending-review path.
                # A matching conditional rule cannot approve an uncertain row.
                suggestion = current["agent_suggestions"].items[index] if current.get("agent_suggestions") else None
                canonical_review = index in current["metadata"].get("agent_1", {}).get("canonical_review_items", [])
                if canonical_review and resolution.get("status") == "covered":
                    resolution = {**resolution, "status": "manual_review", "rule_id": None,
                                  "review_reason": suggestion.reason}
                pairs.append({"line": line, "resolution": resolution, "rule": by_id.get(resolution.get("rule_id"))})
            claim = adjudicate_invoice(current["invoice"], policy=self.policy, resolved_pairs=pairs)
            warnings = list(current["warnings"])
            return {"claim_result": claim, "resolved_pairs": pairs, "warnings": warnings,
                    **({"workflow_status": "REVIEW_REQUIRED"} if claim.claim_status == "manual_review" else {})}
        return self.execute("deterministic_adjudication", state, action)

    def reasoning_review_agent(self, state):
        def action(current):
            output, metadata = self.reviewer.review(current["invoice"], current["candidate_rules"],
                current["agent_suggestions"], current["claim_result"], deadline=current["deadline_monotonic"],
                policy=self.policy, agent_metadata=current["metadata"].get("agent_1", {}))
            return {"agent_review": output, "metadata": {**current["metadata"], "agent_2": metadata},
                    **({"workflow_status": "REVIEW_REQUIRED"} if any(item.needs_review for item in output.items) else {})}
        return self.execute("reasoning_review_agent", state, action)

    def final_response(self, state):
        # Response assembly always runs, including after the request deadline.
        claim = state.get("claim_result")
        reviewed = bool(state.get("agent_review"))
        status = ("FAILED" if state["workflow_status"] == "FAILED" or claim is None else
                  "REVIEW_REQUIRED" if state["workflow_status"] == "REVIEW_REQUIRED" or not reviewed else "COMPLETED")
        trace = [*state["execution_trace"], {"node": "final_response", "request_id": state["request_id"],
                                            "outcome": status, "elapsed_ms": 0}]
        explanations = [{"item_index": index, "reviewed": reviewed,
                         "text": state["agent_review"].items[index].explanation if reviewed else decision.explanation}
                        for index, decision in enumerate(claim.decisions)] if claim else []
        warnings = [*state["warnings"], *([] if reviewed or not claim else [
            "LLM review unavailable/rejected; explanations and amounts are deterministic and unreviewed"])]
        report = WorkflowReport(request_id=state["request_id"], thread_id=state.get("thread_id"),
            status=status, reviewed=reviewed, claim=claim, extracted_csv=state.get("extracted_csv", "") if state.get("csv_validated") else "",
            evidence=state.get("candidate_rules", {}), agent_1=state.get("agent_suggestions"), agent_2=state.get("agent_review"),
            explanations=explanations, errors=state["errors"], warnings=warnings, retries=state["retry_counters"], trace=trace,
            metadata={**state["metadata"], "model_ids": state["model_ids"],
                      "duration_ms": round((time.monotonic()-state["started_monotonic"])*1000)})
        return {"final_report": report, "result": report.model_dump(mode="json"), "workflow_status": status,
                "execution_trace": trace, "trace": [*state["trace"], "final_response"]}
