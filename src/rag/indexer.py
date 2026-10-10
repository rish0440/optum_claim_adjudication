"""Index exactly the 25 coverage and 20 exclusion rules in local Milvus Lite.

The original policy.json remains authoritative; the vector store is a disposable
search index, not a policy database of record. Never add invoice/test contents.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.adjudication.policy import load_policy
from src.rag.embeddings import embed_texts, get_embedder
from src.config.settings import ROOT, get_settings

SCHEMA_VERSION = 1


def source_documents(policy) -> list[dict]:
    """Produce one retrievable chunk per actual policy rule with source metadata."""
    output: list[dict] = []
    for kind, rules in (("coverage", policy.coverage_rules), ("exclusion", policy.exclusion_rules)):
        for rule in rules:
            limit = getattr(rule, "maximum_reimbursement", None)
            limit_type = getattr(rule, "limit_type", "")
            unit = getattr(rule, "unit", None) or ""
            conditions = rule.conditions
            text = (
                f"Health insurance policy rule. Rule ID: {rule.rule_id}. "
                f"Type: {kind}. Category: {rule.category}. "
                f"Conditions: {conditions}. "
                f"Reimbursement basis: {limit_type or 'not reimbursable'}. "
                f"Maximum INR: {limit if limit is not None else 'none'}. "
                f"Unit: {unit or 'none'}."
            )
            output.append({
                "rule_id": rule.rule_id,
                "kind": kind,
                "category": rule.category,
                "conditions": conditions,
                "limit_type": limit_type,
                "max_amount": str(limit) if limit is not None else "",
                "unit": unit,
                "source": "data/policy.json",
                "text": text,
            })
    return output


def policy_sha256(policy_path: Path) -> str:
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def build_policy_index(*, policy_path: Path | None = None,
                       db_path: Path | None = None, manifest_path: Path | None = None,
                       model_name: str | None = None, collection: str | None = None,
                       rebuild: bool = False, client=None, embedder=None) -> dict:
    """Build/reuse a persistent local Milvus collection and a compatibility manifest.

    Refuses to reuse an index when the policy hash or embedding model changed;
    pass --rebuild explicitly. Dependency injection supports fast CPU unit tests.
    """
    settings = get_settings()
    policy_path = Path(policy_path or settings.paths.policy)
    db_path = Path(db_path or settings.rag.db_path)
    manifest_path = Path(manifest_path or settings.rag.manifest_path)
    model_name = model_name or settings.rag.embedding_model
    collection = collection or settings.rag.collection
    policy = load_policy(policy_path)
    documents = source_documents(policy)
    policy_hash = policy_sha256(policy_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if client is None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("Install Milvus: python -m pip install -r requirements-rag.txt") from exc
        client = MilvusClient(uri=str(db_path))

    exists = client.has_collection(collection_name=collection)
    if exists and not rebuild:
        if not manifest_path.exists():
            raise RuntimeError("Index exists without a manifest. Re-run with --rebuild.")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        needed = ("policy_sha256", "embedding_model", "collection", "schema_version")
        expected = (policy_hash, model_name, collection, SCHEMA_VERSION)
        if tuple(old.get(k) for k in needed) != expected or old.get("rule_count") != len(documents):
            raise RuntimeError("Policy/index configuration changed; rebuild with --rebuild.")
        return {**old, "status": "reused"}

    embedder = embedder or get_embedder(model_name)
    vectors = embed_texts(embedder, [doc["text"] for doc in documents])
    if not vectors or any(len(vec) != len(vectors[0]) for vec in vectors):
        raise RuntimeError("Embedding model produced empty or inconsistent vectors")

    if exists:
        client.drop_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection, dimension=len(vectors[0]),
        metric_type="COSINE", auto_id=False,
    )
    records = [{"id": idx, "vector": vector, **doc}
               for idx, (doc, vector) in enumerate(zip(documents, vectors), start=1)]
    inserted = client.insert(collection_name=collection, data=records)
    inserted_count = inserted.get("insert_count", 0) if isinstance(inserted, dict) else len(records)
    if inserted_count != len(records):
        raise RuntimeError(f"Inserted {inserted_count} of {len(records)} policy rules")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_hash,
        "embedding_model": model_name,
        "collection": collection,
        "db_path": str(db_path),
        "dimension": len(vectors[0]),
        "rule_count": len(documents),
        "coverage_count": len(policy.coverage_rules),
        "exclusion_count": len(policy.exclusion_rules),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "status": "built"}
