"""Agent 1 selects scoped candidates; Python verifies and supplies policy facts."""
from src.adjudication.policy import load_policy
from src.agents.contracts import (AgentFormatError, AgentSafetyError, CandidateSelections,
                                 PolicySuggestion, Suggestions, reject_instructions)
from src.rag.resolver import PolicyRuleResolver
from src.utils.paths import ROOT


def _diagnostic(category, index=None, selection=None, *, field="candidate_index"):
    return {"category": category, "item_index": index, "field": field,
            "candidate_index": selection.candidate_index if selection else None,
            "sanitized_response": selection.model_dump() if selection else None}


class PolicyAdjudicationAgent:
    def __init__(self, client):
        self.client = client
        self.prompt = (ROOT / "src/prompts/adjudicator.txt").read_text(encoding="utf-8")

    def suggest(self, invoice, evidence, resolutions, *, deadline, policy=None):
        policy = policy or load_policy()
        resolver = PolicyRuleResolver(policy)
        applicable = policy.effective_from <= invoice.invoice_date <= policy.effective_to
        rows, maps, diagnostics = [], {}, []
        for index, line in enumerate(invoice.lines):
            reject_instructions(line.item)
            resolution = resolver.resolve(line.item)
            # Independent category/exclusion verification, not the LLM or score.
            rule_id = resolution.get("rule_id")
            if rule_id != resolutions.get(index, {}).get("rule_id"):
                rule_id = None
                diagnostics.append(_diagnostic("resolution_conflict", index, field="rule_id"))
            candidates = {}
            for slot, doc in enumerate(evidence.get(index, [])):
                canonical = resolver.by_id.get(doc.get("rule_id"))
                if canonical is None:
                    error = AgentSafetyError("Agent 1 evidence contains a noncanonical policy reference")
                    error.diagnostics = _diagnostic("noncanonical_evidence", index, field="rule_id")
                    raise error
                kind = "exclusion" if canonical.rule_id.startswith("EXC-") else "coverage"
                if not isinstance(doc.get("conditions"), str) or not doc["conditions"].strip():
                    diagnostics.append(_diagnostic("missing_canonical_condition", index, field="conditions"))
                    continue
                if any(doc.get(key) != value for key, value in
                       (("category", canonical.category), ("conditions", canonical.conditions), ("kind", kind))):
                    error = AgentSafetyError("Agent 1 evidence differs from canonical policy")
                    error.diagnostics = _diagnostic("canonical_metadata_mismatch", index, field="policy_record")
                    raise error
                # Keep explicit excluded subtypes; omit unrelated semantic hits.
                if applicable and canonical.rule_id == rule_id:
                    reject_instructions(canonical.conditions)
                    candidates[slot] = doc
            maps[index] = candidates
            context_required = bool(resolution.get("review_reason")) or any(
                phrase in (resolver.by_id[rule_id].conditions.lower() if rule_id else "")
                for phrase in ("when medically related", "when directly related", "medically necessary"))
            rows.append({"item_index": index, "item": line.item,
                "context_required": context_required,
                "candidates": [{"index": slot, "kind": doc["kind"], "category": doc["category"],
                                "conditions": doc["conditions"]} for slot, doc in candidates.items()]})
        data = {"date_applicable": applicable, "items": rows}
        try:
            result, metadata = self.client.generate(self.prompt, data, CandidateSelections, deadline=deadline)
        except AgentFormatError as exc:
            if not hasattr(exc, "diagnostics"):
                exc.diagnostics = _diagnostic("invalid_json_or_schema", field="response_schema")
            raise
        if [item.item_index for item in result.items] != list(range(len(invoice.lines))):
            error = AgentSafetyError("Agent 1 changed row identities or order")
            error.diagnostics = _diagnostic("row_identity_or_order", field="item_index")
            raise error
        verified, canonical_review_items, disagreement_items = [], [], []
        for item in result.items:
            doc = maps[item.item_index].get(item.candidate_index)
            expected = "EXCLUDED" if doc and doc["kind"] == "exclusion" else "COVERED"
            failure = None
            if item.candidate_index is not None and doc is None:
                failure = "unknown_or_inapplicable_candidate"
            elif not applicable:
                failure = "policy_date_outside"
            elif item.suggested_status != "REVIEW_REQUIRED" and (
                    doc is None or item.suggested_status != expected or item.basis != "RULE_MATCH"):
                failure = "unsupported_classification"
            elif item.suggested_status == "REVIEW_REQUIRED" and item.basis == "RULE_MATCH":
                failure = "contradictory_review_basis"
            elif item.basis == "DATE_OUTSIDE" or (item.basis == "NO_EVIDENCE" and doc is not None):
                failure = "unsupported_reason_basis"
            elif doc and doc["kind"] == "coverage" and rows[item.item_index]["context_required"]:
                failure = "clinical_conditions_unverified"
            # Advisory errors are not evidence of eligibility uncertainty. Resolve
            # the independently validated row, even if the model selected badly.
            canonical_doc = next(iter(maps[item.item_index].values()), None)
            needs_review = not applicable or canonical_doc is None or (
                canonical_doc["kind"] == "coverage" and rows[item.item_index]["context_required"])
            canonical_status = ("REVIEW_REQUIRED" if needs_review else
                                "EXCLUDED" if canonical_doc["kind"] == "exclusion" else "COVERED")
            if needs_review:
                canonical_review_items.append(item.item_index)
            if failure not in (None, "clinical_conditions_unverified", "policy_date_outside") or (
                    item.suggested_status != canonical_status):
                disagreement_items.append(item.item_index)
            if failure:
                field = {"unsupported_classification": "suggested_status",
                         "contradictory_review_basis": "basis", "unsupported_reason_basis": "basis",
                         "clinical_conditions_unverified": "conditions",
                         "policy_date_outside": "date_applicable"}.get(failure, "candidate_index")
                diagnostic = _diagnostic(failure, item.item_index, item, field=field)
                diagnostic["allowed_candidate_indices"] = sorted(maps[item.item_index])
                diagnostic["canonical_policy_id"] = doc["rule_id"] if doc else None
                diagnostics.append(diagnostic)
            reason = ("Invoice date outside policy effective period." if not applicable else
                      "Treatment context requires manual review." if canonical_doc else
                      "Insufficient verified policy evidence.") if needs_review else canonical_doc["conditions"]
            verified.append(PolicySuggestion(item_index=item.item_index,
                suggested_status=canonical_status,
                policy_ids=[canonical_doc["rule_id"]] if canonical_doc else [], reason=reason, needs_review=needs_review,
                supported_conditions=canonical_doc["conditions"] if canonical_doc else None))
        # Preserve actual model judgments as typed, privacy-safe reasoning codes;
        # canonical IDs/conditions in downstream state are supplied only by Python.
        return Suggestions(items=verified), {**metadata, "model_selections": result.model_dump()["items"],
            "candidate_map": {index: {slot: doc["rule_id"] for slot, doc in candidates.items()}
                              for index, candidates in maps.items()},
            "candidate_diagnostics": diagnostics, "canonical_review_items": canonical_review_items,
            "disagreement_items": disagreement_items, "reason_source": "independently_verified_policy",
            "suggestions_role": "validated canonical result; raw LLM advice retained in model_selections"}
