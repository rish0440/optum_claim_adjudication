"""Phase 6 regression tests use a fake Milvus/embedding model (CPU/offline).

They test index contracts and resolver safety, NOT true Milvus integration,
MiniLM semantic quality, or extraction accuracy. Real Milvus CLI test required.
"""

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from src.policy import load_policy
from src.rag.indexer import build_policy_index, source_documents
from src.rag.resolver import PolicyRuleResolver
from src.rag.retriever import PolicyRetriever
from scripts.test_policy_rag import evaluate
from src.settings import ROOT


class FakeEmbeddings:
    def encode(self, texts, **kwargs):
        output = []
        for text in texts:
            values = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = hashlib.sha256(word.encode()).digest()[0] % 64
                values[bucket] += 1.0
            magnitude = math.sqrt(sum(value * value for value in values)) or 1
            output.append([value / magnitude for value in values])
        return output


class FakeMilvus:
    """Mimic only the MilvusClient calls our code uses, including COSINE search."""

    def __init__(self):
        self.collections = {}
        self.loaded = set()

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, dimension, **kwargs):
        self.collections[collection_name] = []
        self.loaded.discard(collection_name)
        self.dimension = dimension

    def load_collection(self, collection_name):
        if collection_name not in self.collections:
            raise ValueError("Collection does not exist")
        self.loaded.add(collection_name)

    def drop_collection(self, collection_name):
        del self.collections[collection_name]
        self.loaded.discard(collection_name)

    def insert(self, collection_name, data):
        self.collections[collection_name].extend(data)
        return {"insert_count": len(data)}

    def search(self, collection_name, data, limit, output_fields, filter=""):
        if collection_name not in self.loaded:
            raise RuntimeError("Collection is released; call load_collection first")
        rows = list(self.collections[collection_name])
        if filter:
            kind = filter.split('"')[1]
            rows = [row for row in rows if row["kind"] == kind]
        found = sorted(rows,
                       key=lambda row: sum(a * b for a, b in zip(data[0], row["vector"])),
                       reverse=True)[:limit]
        return [[{"distance": sum(a * b for a, b in zip(data[0], row["vector"])),
                  "entity": {key: row[key] for key in output_fields}} for row in found]]


class Phase6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.manifest = Path(self.tmp.name) / "manifest.json"
        self.embedder = FakeEmbeddings()
        self.client = FakeMilvus()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **opts):
        return build_policy_index(db_path=self.db, manifest_path=self.manifest,
                                  client=self.client, embedder=self.embedder, **opts)

    def retriever(self):
        return PolicyRetriever(db_path=self.db, manifest_path=self.manifest,
                               client=self.client, embedder=self.embedder)

    def test_index_exact_45_source_rules_with_metadata(self):
        docs = source_documents(self.policy)
        ids = [doc["rule_id"] for doc in docs]
        self.assertEqual(len(docs), 45)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len([doc for doc in docs if doc["kind"] == "coverage"]), 25)
        self.assertEqual(len([doc for doc in docs if doc["kind"] == "exclusion"]), 20)
        self.assertTrue(all(doc["source"] == "data/policy.json" for doc in docs))
        self.assertEqual(next(d for d in docs if d["rule_id"] == "COV-010")["category"], "Blood Tests")

    def test_build_is_repeatable_and_does_not_touch_original_policy(self):
        original = (ROOT / "data/policy.json").read_bytes()
        first = self.build()
        self.assertEqual(first["status"], "built")
        self.assertEqual(first["rule_count"], 45)
        self.assertEqual(len(self.client.collections["policy_rules"]), 45)
        second = self.build()
        self.assertEqual(second["status"], "reused")
        self.assertEqual(original, (ROOT / "data/policy.json").read_bytes())
        third = self.build(rebuild=True)
        self.assertEqual(third["status"], "built")
        self.assertEqual(len(self.client.collections["policy_rules"]), 45)

    def test_stale_manifest_causes_explicit_failure(self):
        self.build()
        metadata = json.loads(self.manifest.read_text())
        metadata["policy_sha256"] = "no-longer-valid"
        self.manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            self.build()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.retriever()

    def test_retriever_explicitly_loads_released_collection(self):
        """Regression: a persisted Milvus collection starts in released state."""
        self.build()
        self.assertNotIn("policy_rules", self.client.loaded)
        retriever = self.retriever()
        self.assertIn("policy_rules", self.client.loaded)
        self.assertTrue(retriever.search("Personal Care Kit", top_k=3))

    def test_milvus_search_returns_actual_rule_metadata(self):
        self.build()
        result = self.retriever().search("MRI Scan", top_k=3)
        self.assertEqual(len(result), 3)
        known_ids = {rule.rule_id for rule in self.policy.coverage_rules + self.policy.exclusion_rules}
        self.assertTrue(all(hit["rule_id"] in known_ids for hit in result))
        self.assertTrue(all(hit["source"] == "data/policy.json" for hit in result))
        self.assertTrue(all(isinstance(hit["score"], float) for hit in result))
        self.assertTrue(all(hit["kind"] == "exclusion" for hit in
                            self.retriever().search("personal care", kind="exclusion")))

    def test_specific_coverage_and_exclusions(self):
        resolver = PolicyRuleResolver(self.policy)
        examples = {
            "Specialist Consultation": "COV-004",
            "Consultation": "COV-003",
            "Blood Test (CBC)": "COV-010",
            "MRI Scan": "COV-007",
            "CT Scan": "COV-008",
            "X-Ray": "COV-009",
            "Personal Care Kit": "EXC-001",
            "Health Supplement": "EXC-004",
            "Diet Meal": "EXC-006",
            "Television Charges": "EXC-007",
            "Surgical Gloves": "COV-019",
        }
        for item, rule_id in examples.items():
            with self.subTest(item=item):
                result = resolver.resolve(item)
                self.assertEqual(result["rule_id"], rule_id)
                self.assertEqual(result["conditions"], resolver.by_id[rule_id].conditions)
                self.assertEqual(result["status"], "excluded" if rule_id.startswith("EXC-") else "covered")
        self.assertIn("verify treatment context", resolver.resolve("Surgical Gloves")["review_reason"])

    def test_unknown_item_does_not_silently_approve(self):
        resolver = PolicyRuleResolver(self.policy)
        result = resolver.resolve("Unexpected miscellaneous experimental gadget", [
            {"rule_id": "COV-005", "score": 0.999},
            {"rule_id": "FAKE-000", "score": 1.0},
        ])
        self.assertIsNone(result["rule_id"])
        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(len(result["candidates"]), 1)

    def test_retrieval_evaluation_reports_real_denominators(self):
        self.build()
        cases = json.loads((ROOT / "data/evaluation/policy_rag_cases.json").read_text())
        report = evaluate(self.retriever(), PolicyRuleResolver(self.policy), cases)
        self.assertEqual(report["cases"], 20)
        for metric in report["metrics"].values():
            self.assertEqual(metric["total"], 20)
            self.assertLessEqual(metric["correct"], metric["total"])
        self.assertEqual(len(report["results"]), 20)
        self.assertEqual(report["metrics"]["grounded"]["correct"], 20)


if __name__ == "__main__":
    unittest.main()
