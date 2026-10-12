"""Agent 2 selects verified facts; Python renders exact calculator explanations."""
from src.utils.policy import load_policy
from src.components.rag.resolver import PolicyRuleResolver
from src.components.agentic_workflow.contracts import (AgentFormatError, AgentSafetyError, FactSelections,
                                 ItemReview, ReviewedTotals, ReviewOutput)
from src.utils.paths import ROOT


class ReasoningReviewAgent:
    def __init__(self, client):
        self.client = client
        self.prompt = (ROOT / "src/components/prompts/reasoning.txt").read_text(encoding="utf-8")

    def review(self, invoice, evidence, suggestions, claim, *, deadline, policy=None, agent_metadata=None):
        policy = policy or load_policy()
        canonical = {rule.rule_id: rule for rule in policy.coverage_rules + policy.exclusion_rules}
        resolver = PolicyRuleResolver(policy)
        agent_metadata = agent_metadata or {}
        advice = {row["item_index"]: row for row in agent_metadata.get("model_selections", [])}
        disagreements = set(agent_metadata.get("disagreement_items", []))
        rows, facts_by_item, concerns_by_item = [], {}, {}

        def reject(category, index=None, selection=None, field="fact_indices"):
            error = AgentSafetyError(f"Agent 2 rejected: {category}; item_index={index}")
            error.diagnostics = {"category": category, "item_index": index, "field": field,
                                 "sanitized_response": selection.model_dump() if selection else None}
            if selection is not None:
                error.diagnostics.update(allowed_fact_indices=sorted(facts_by_item.get(index, set())),
                    selected_fact_indices=selection.fact_indices,
                    allowed_concern_indices=sorted(concerns_by_item.get(index, {})),
                    selected_concern_indices=selection.concerns,
                    expected_status=claim.decisions[index].status)
            raise error

        if len(invoice.lines) != len(claim.decisions) or len(suggestions.items) != len(invoice.lines):
            reject("input_row_count", field="items")
        for index, (line, decision) in enumerate(zip(invoice.lines, claim.decisions)):
            values = decision.model_dump(mode="json")
            # Compact verified facts, without duplicating long calculator prose.
            # Disjoint row-owned IDs prevent another item's otherwise valid fact
            # from being accepted. Facts and concerns have separate namespaces.
            facts = [{"index": 2 * index, "kind": "decision", **{key: values[key] for key in
                ("status", "rule_id", "billed_amount", "approved_amount", "disallowed_amount", "pending_review_amount")}}]
            context_id = decision.rule_id or (suggestions.items[index].policy_ids[0]
                if suggestions.items[index].needs_review and len(suggestions.items[index].policy_ids) == 1 else None)
            if context_id:
                rule = canonical.get(context_id)
                entry = next((doc for doc in evidence.get(index, []) if doc.get("rule_id") == context_id), None)
                if rule is None or not entry or not rule.conditions.strip() or (
                        entry.get("conditions") != rule.conditions or entry.get("category") != rule.category
                        or resolver.resolve(line.item).get("rule_id") != context_id):
                    reject("unverified_policy_fact", index, field="policy_record")
                facts.append({"index": 2 * index + 1, "kind": "policy", "rule_id": rule.rule_id, "conditions": rule.conditions})
            concern_map = ({2 * index: "POLICY_CONTEXT_MISSING"} if decision.status == "manual_review" else {})
            if index in disagreements:
                concern_map[2 * index + 1] = "AGENT_DISAGREEMENT"
            concerns_by_item[index] = concern_map
            decision_type = ("MANUAL_REVIEW" if decision.status == "manual_review" else
                             "POLICY_CAP" if decision.status == "partial" else
                             "EXCLUDED" if decision.rule_id and decision.rule_id.startswith("EXC-") else
                             "APPROVED" if decision.status == "approved" else "NOT_PAYABLE")
            raw_advice = advice.get(index)
            rows.append({"item_index": index, "item": line.item, "facts": facts,
                "agent_1": {key: raw_advice[key] for key in ("suggested_status", "basis")} if raw_advice else
                           {"suggested_status": suggestions.items[index].suggested_status},
                "decision_type": decision_type, "review_required": decision.status == "manual_review",
                "concern_facts": [{"index": slot, "kind": kind} for slot, kind in concern_map.items()],
                "required_concerns": list(concern_map)})
            facts_by_item[index] = {fact["index"] for fact in facts}
        totals = {key: claim.model_dump(mode="json")[key] for key in
                  ("eligible_total", "copay_amount", "final_reimbursement")}
        try:
            output, metadata = self.client.generate(self.prompt, {"items": rows, "totals": totals},
                                                     FactSelections, deadline=deadline)
        except AgentFormatError as exc:
            if not hasattr(exc, "diagnostics"):
                exc.diagnostics = {"category": "invalid_json_or_schema", "item_index": None,
                                   "field": "response_schema", "sanitized_response": None}
            raise
        if [item.item_index for item in output.items] != list(range(len(invoice.lines))):
            reject("row_identity_or_order", field="item_index")
        reviewed = []
        for item, decision in zip(output.items, claim.decisions):
            available = facts_by_item[item.item_index]
            if len(item.fact_indices) != len(set(item.fact_indices)) or set(item.fact_indices) != available:
                reject("unknown_or_missing_fact", item.item_index, item)
            if item.status != decision.status:
                reject("contradictory_status", item.item_index, item, "status")
            if len(item.concerns) != len(set(item.concerns)) or set(item.concerns) != set(concerns_by_item[item.item_index]):
                reject("unsupported_or_missing_concern", item.item_index, item, "concerns")
            explanation = decision.explanation
            policy_fact = next((fact for fact in rows[item.item_index]["facts"] if fact["kind"] == "policy"), None)
            if decision.rule_id is None and policy_fact:
                explanation += f" Policy context {policy_fact['rule_id']}: {policy_fact['conditions']}"
            if any(concerns_by_item[item.item_index][slot] == "AGENT_DISAGREEMENT" for slot in item.concerns):
                explanation += " Agent 1 advice contradicted verified evidence; the canonical decision is retained."
            reviewed.append(ItemReview(item_index=item.item_index, status=decision.status,
                policy_ids=[decision.rule_id] if decision.rule_id else [],
                approved_amount=decision.approved_amount, disallowed_amount=decision.disallowed_amount,
                pending_review_amount=decision.pending_review_amount, explanation=explanation,
                needs_review=decision.status == "manual_review"))
        return ReviewOutput(items=reviewed, totals=ReviewedTotals(**totals)), {**metadata,
            "fact_selections": output.model_dump()["items"], "fact_map": {
                row["item_index"]: {fact["index"]: "calculator_decision" if fact["kind"] == "decision" else fact["rule_id"]
                                    for fact in row["facts"]} for row in rows},
            "concern_map": concerns_by_item,
            "explanation_source": "selected_verified_facts"}
