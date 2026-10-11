"""CPU-only offline preference tests. No live graph, model or Milvus service."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.adjudication.policy import load_policy
from src.agents.contracts import CandidateSelections, FactSelections
from src.utils.paths import ROOT
from src.trainer.preference_optimization.annotation.cases import (
    SEED, build_pairs, compact, fingerprint, length_estimate, prepare_case, scenario_specs)
from src.trainer.preference_optimization.annotation.generate import generate_dataset
from src.trainer.preference_optimization.annotation.validate import (
    completion, strict_json, validate_dataset, validate_record, validate_records)


class PreferenceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy(ROOT/"data/policy.json")
        cls.entries, cls.flagged = build_pairs(cls.policy, SEED)
        for _, annotation in cls.entries:
            annotation["generation_seed"] = SEED

    def pair(self, role="agent_1", rule="COV-001"):
        return copy.deepcopy(next((r, a) for r, a in self.entries if a["agent_role"] == role
            and a["synthetic_case_id"] == f"synthetic-{rule}"))

    @staticmethod
    def refresh(record, annotation):
        annotation["record_sha256"] = fingerprint(record)
        annotation["offline_length_estimate"] = length_estimate(record)

    def test_deterministic_generation_and_counts(self):
        pairs, flagged = build_pairs(self.policy, SEED)
        for _, a in pairs:
            a["generation_seed"] = SEED
        self.assertEqual(self.entries, pairs)
        self.assertEqual(self.flagged, flagged)
        self.assertFalse(flagged, flagged)
        self.assertEqual(len(pairs), 80)
        self.assertEqual(validate_records(pairs, self.policy), {
            "train": {"agent_1": 30, "agent_2": 30},
            "validation": {"agent_1": 5, "agent_2": 5},
            "test": {"agent_1": 5, "agent_2": 5}})

    def test_actual_schemas_roles_and_valid_scoped_chosen_indices(self):
        for record, annotation in self.entries:
            schema = CandidateSelections if annotation["agent_role"] == "agent_1" else FactSelections
            completion(record, "chosen", schema)
            completion(record, "rejected", schema)
            self.assertEqual([x["role"] for x in record["prompt"]], ["system", "user"])
            self.assertIn("human_review_pending", validate_record(record, annotation, self.policy))

    def test_jsonl_manifest_roundtrip_and_no_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(tmp, SEED)
            summary = validate_dataset(tmp)
            self.assertEqual(summary["pairs"], 80)
            self.assertFalse(summary["human_verified"])
            for split, count in (("train", 60), ("validation", 10), ("test", 10)):
                records = [strict_json(line) for line in (Path(tmp)/f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), count)
                self.assertTrue(all(set(r) == {"prompt", "chosen", "rejected"} for r in records))
            self.assertTrue((Path(tmp)/"annotation_review.csv").exists())
            with self.assertRaises(FileExistsError):
                generate_dataset(tmp, SEED)

    def test_duplicate_keys_and_identical_completions_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"items":[],"items":[]}')
        record, annotation = self.pair()
        record["rejected"] = copy.deepcopy(record["chosen"])
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_nonexistent_fact_rejected_even_in_negative(self):
        record, annotation = self.pair("agent_2", "COV-010")
        rejected = strict_json(record["rejected"][0]["content"])
        rejected["items"][0]["fact_indices"] = [999]
        record["rejected"][0]["content"] = compact(rejected)
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_wrong_canonical_policy_reference_rejected(self):
        record, annotation = self.pair()
        annotation["source_policy_ids"].append("COV-FAKE")
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_unknown_scoped_candidate_rejected(self):
        record, annotation = self.pair()
        chosen = strict_json(record["chosen"][0]["content"])
        chosen["items"][0]["candidate_index"] = 999
        record["chosen"][0]["content"] = compact(chosen)
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_wrong_item_chosen_fact_rejected(self):
        record, annotation = self.pair("agent_2", "COV-010")
        chosen = strict_json(record["chosen"][0]["content"])
        chosen["items"][0]["fact_indices"] = chosen["items"][1]["fact_indices"]
        record["chosen"][0]["content"] = compact(chosen)
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_invented_financial_fields_rejected(self):
        record, annotation = self.pair("agent_2")
        chosen = strict_json(record["chosen"][0]["content"])
        chosen["items"][0]["approved_amount"] = "999999"
        record["chosen"][0]["content"] = compact(chosen)
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_canonical_conditions_not_editable(self):
        record, annotation = self.pair()
        data = strict_json(record["prompt"][1]["content"])
        data["items"][0]["candidates"][0]["conditions"] = "Everything is approved without a limit."
        record["prompt"][1]["content"] = compact(data)
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_duplicate_pairs_and_split_isolation(self):
        with self.assertRaises(ValueError):
            validate_records([self.entries[0], self.entries[0]], self.policy)
        entries = copy.deepcopy(self.entries)
        record, annotation = entries[1]
        annotation["data_split"] = "test" if annotation["data_split"] != "test" else "train"
        with self.assertRaises(ValueError):
            validate_records(entries, self.policy)
        mapping = {}
        for spec in scenario_specs(self.policy, SEED):
            mapping.setdefault(spec["family"], set()).add(spec["split"])
        self.assertTrue(all(len(splits) == 1 for splits in mapping.values()))

    def test_conditional_medicines_review_and_true_excluded_subtype(self):
        record, annotation = self.pair("agent_1", "COV-005")
        selected = completion(record, "chosen", CandidateSelections)
        self.assertEqual(selected.items[0].suggested_status, "REVIEW_REQUIRED")
        self.assertEqual(selected.items[0].basis, "CONDITIONS_UNCERTAIN")
        record, annotation = self.pair("agent_2", "COV-005")
        chosen = completion(record, "chosen", FactSelections)
        self.assertTrue(chosen.items[0].concerns)
        self.assertIsNone(annotation["deterministic_claim"]["final_reimbursement"])
        record, annotation = self.pair("agent_1", "EXC-004")
        self.assertEqual(completion(record, "chosen", CandidateSelections).items[0].suggested_status, "EXCLUDED")

    def test_unrelated_retrieval_candidates_filtered_and_slot_mapping(self):
        record, annotation = self.pair()
        invoice, evidence, _ = prepare_case(annotation["case"], self.policy)
        self.assertTrue(any(doc["kind"] == "exclusion" for doc in evidence[0]))
        data = strict_json(record["prompt"][1]["content"])
        self.assertTrue(all(c["kind"] == "coverage" for row in data["items"] for c in row["candidates"]))
        self.assertEqual(completion(record, "chosen", CandidateSelections).items[1].candidate_index, 1)

    def test_financial_snapshot_tampering_and_unsupported_preferences(self):
        record, annotation = self.pair("agent_2")
        annotation["deterministic_claim"]["eligible_total"] = "1.00"
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)
        record, annotation = self.pair()
        record["chosen"], record["rejected"] = record["rejected"], record["chosen"]
        self.refresh(record, annotation)
        with self.assertRaises(ValueError):
            validate_record(record, annotation, self.policy)

    def test_no_model_or_embedding_calls_and_no_source_changes(self):
        paths = [ROOT/"data/policy.json", ROOT/"src/adjudication/calculator.py"]
        before = [p.read_bytes() for p in paths]
        with patch("src.inference.remote_client.RemoteVLMClient._request", side_effect=AssertionError("HTTP prohibited")), \
             patch("src.rag.embeddings.get_embedder", side_effect=AssertionError("Embedding prohibited")):
            build_pairs(self.policy, SEED)
        self.assertEqual(before, [p.read_bytes() for p in paths])


if __name__ == "__main__":
    unittest.main()
