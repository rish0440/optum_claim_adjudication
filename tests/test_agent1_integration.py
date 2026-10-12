"""Focused CPU regressions for the real reference-claim Agent 1 rejection."""
import json
import time
import unittest
from pydantic import ValidationError

from src.utils.policy import load_policy
from src.utils.rules import parse_invoice_csv
from src.components.agentic_workflow.contracts import AgentFormatError, AgentSafetyError
from src.components.agentic_workflow.policy_adjudicator import PolicyAdjudicationAgent
from src.components.rag.indexer import source_documents
from src.components.rag.resolver import PolicyRuleResolver
from src.utils.paths import ROOT


class CapturingClient:
    def __init__(self, output):
        self.output = output
        self.data = None

    def generate(self, prompt, data, schema, *, deadline):
        self.data = data
        if isinstance(self.output, Exception):
            raise self.output
        try:
            return schema.model_validate(self.output), {"model": "qwen-base"}
        except ValidationError as exc:
            raise AgentFormatError("Invalid reasoning JSON/schema") from exc


class Agent1IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.invoice = parse_invoice_csv(ROOT / "data/assignment_dataset/ground_truth/invoice_01_standard.csv",
                                         invoice_date="2024-01-15")
        self.docs = {doc["rule_id"]: doc for doc in source_documents(self.policy)}
        resolver = PolicyRuleResolver(self.policy)
        self.resolutions = {i: resolver.resolve(line.item) for i, line in enumerate(self.invoice.lines)}
        unrelated = [doc for doc in self.docs.values() if doc["category"] in
                     {"Television Charges", "Beauty Services", "Laundry Charges"}]
        self.assertTrue(unrelated)
        self.evidence = {i: [self.docs[resolution["rule_id"]], *unrelated]
                         for i, resolution in self.resolutions.items()}
        self.output = {"items": [{"item_index": i, "suggested_status": "COVERED",
            "candidate_index": 0, "basis": "RULE_MATCH"}
            for i, resolution in self.resolutions.items()]}
        # This CSV alone does not establish medicines' medical relationship.
        self.output["items"][2].update(suggested_status="REVIEW_REQUIRED", basis="CONDITIONS_UNCERTAIN")

    def suggest(self, output=None):
        self.client = CapturingClient(self.output if output is None else output)
        return PolicyAdjudicationAgent(self.client).suggest(self.invoice, self.evidence, self.resolutions,
            deadline=time.monotonic()+60, policy=self.policy)

    def test_all_four_reference_rows_map_scoped_candidates_to_canonical_rules(self):
        result, _ = self.suggest()
        self.assertEqual([row.policy_ids for row in result.items],
                         [["COV-001"], ["COV-003"], ["COV-005"], ["COV-010"]])
        self.assertTrue(all([candidate["index"] for candidate in row["candidates"]] == [0]
                            for row in self.client.data["items"]))
        self.assertEqual(result.items[0].reason, self.docs["COV-001"]["conditions"])
        self.assertTrue(result.items[2].needs_review)
        self.assertEqual(result.items[2].reason, "Treatment context requires manual review.")

    def test_irrelevant_model_selection_does_not_override_independent_coverage(self):
        self.output["items"][0].update(suggested_status="EXCLUDED", candidate_index=1)
        result, metadata = self.suggest()
        diagnostic = metadata["candidate_diagnostics"][0]
        self.assertEqual(diagnostic["category"], "unknown_or_inapplicable_candidate")
        self.assertEqual(diagnostic["item_index"], 0)
        self.assertEqual(diagnostic["candidate_index"], 1)
        self.assertEqual(result.items[0].suggested_status, "COVERED")
        self.assertEqual(result.items[0].policy_ids, ["COV-001"])
        self.assertIn(0, metadata["disagreement_items"])
        self.assertNotIn("Room Charges", json.dumps(diagnostic))

    def test_genuinely_applicable_exclusion_is_retained(self):
        self.invoice = self.invoice.model_copy(update={"lines": [self.invoice.lines[0].model_copy(
            update={"item": "Television Charges"})]})
        resolution = PolicyRuleResolver(self.policy).resolve("Television Charges")
        rule_id = resolution["rule_id"]
        self.resolutions = {0: resolution}
        self.evidence = {0: [self.docs["COV-001"], self.docs[rule_id]]}
        result, _ = self.suggest({"items": [{"item_index": 0, "suggested_status": "EXCLUDED",
            "candidate_index": 1, "basis": "RULE_MATCH"}]})
        self.assertEqual(result.items[0].policy_ids, [rule_id])
        self.assertEqual([candidate["index"] for candidate in self.client.data["items"][0]["candidates"]], [1])

    def test_unknown_model_candidate_and_wrong_classification_cannot_override_verified_rule(self):
        for change, category in (({"candidate_index": 999}, "unknown_or_inapplicable_candidate"),
                ({"suggested_status": "EXCLUDED"}, "unsupported_classification"),
                ({"suggested_status": "REVIEW_REQUIRED"}, "contradictory_review_basis")):
            with self.subTest(category=category):
                output = json.loads(json.dumps(self.output))
                output["items"][0].update(change)
                result, metadata = self.suggest(output)
                self.assertEqual(metadata["candidate_diagnostics"][0]["category"], category)
                self.assertEqual(result.items[0].suggested_status, "COVERED")
                self.assertFalse(result.items[0].needs_review)
                self.assertIn(0, metadata["disagreement_items"])

    def test_no_condition_quotation_is_required_from_model(self):
        result, _ = self.suggest()
        self.assertEqual(result.items[0].supported_conditions, self.docs["COV-001"]["conditions"])
        self.assertNotIn("supported_conditions", self.output["items"][0])

    def test_missing_evidence_cannot_approve(self):
        self.evidence[0] = []
        result, _ = self.suggest()
        self.assertTrue(result.items[0].needs_review)
        self.assertEqual(result.items[0].policy_ids, [])

    def test_missing_canonical_condition_requires_review(self):
        self.evidence[0][0] = {**self.evidence[0][0], "conditions": None}
        result, metadata = self.suggest()
        self.assertTrue(result.items[0].needs_review)
        self.assertEqual(metadata["candidate_diagnostics"][0]["category"], "missing_canonical_condition")

    def test_clinical_conditions_cannot_be_established_by_selecting_an_id(self):
        self.output["items"][2].update(suggested_status="COVERED", basis="RULE_MATCH")
        result, metadata = self.suggest()
        self.assertTrue(result.items[2].needs_review)
        self.assertIn("clinical_conditions_unverified", [entry["category"] for entry in metadata["candidate_diagnostics"]])

    def test_invalid_json_preserves_format_error_with_safe_diagnostics(self):
        with self.assertRaises(AgentFormatError) as caught:
            self.suggest(AgentFormatError("Invalid reasoning JSON/schema"))
        self.assertEqual(caught.exception.diagnostics["category"], "invalid_json_or_schema")

    def test_fabricated_citations_prose_and_financial_values_are_rejected(self):
        for key, value in (("policy_ids", ["COV-999"]), ("reason", "All claims approved"),
                           ("reimbursement", 999999), ("candidate_index", "0")):
            output = json.loads(json.dumps(self.output))
            output["items"][0][key] = value
            with self.subTest(field=key), self.assertRaises(AgentFormatError):
                self.suggest(output)

    def test_explicit_excluded_subtype_overrides_broader_coverage(self):
        self.invoice = self.invoice.model_copy(update={"lines": [self.invoice.lines[2].model_copy(
            update={"item": "Medicines - Health Supplements"})]})
        resolution = PolicyRuleResolver(self.policy).resolve(self.invoice.lines[0].item)
        self.assertEqual(resolution["rule_id"], "EXC-004")
        self.resolutions = {0: resolution}
        self.evidence = {0: [self.docs["COV-005"], self.docs["EXC-004"], self.docs["EXC-010"]]}
        result, _ = self.suggest({"items": [{"item_index": 0, "candidate_index": 1,
                                            "suggested_status": "EXCLUDED", "basis": "RULE_MATCH"}]})
        self.assertEqual(result.items[0].suggested_status, "EXCLUDED")
        self.assertEqual(result.items[0].policy_ids, ["EXC-004"])
        self.assertIsNone(PolicyRuleResolver(self.policy).resolve("Medicines without health supplements")["rule_id"])
        self.assertIsNone(PolicyRuleResolver(self.policy).resolve("Medicines and Health Supplements")["rule_id"])
