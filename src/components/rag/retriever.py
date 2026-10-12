"""Top-k Milvus semantic policy search with an auditable source-rule contract."""

import json
from pathlib import Path

from src.components.rag.embeddings import embed_texts, get_embedder
from src.components.rag.indexer import SCHEMA_VERSION, policy_sha256
from src.config.settings import get_settings


class PolicyRetriever:
    """Search the built policy-rule index. No coverage decisions are made here."""

    def __init__(self, *, client=None, embedder=None, policy_path: Path | None = None,
                 db_path: Path | None = None, manifest_path: Path | None = None,
                 model_name: str | None = None, collection: str | None = None):
        settings = get_settings()
        self.policy_path = Path(policy_path or settings.paths.policy)
        db_path = Path(db_path or settings.rag.db_path)
        manifest_path = Path(manifest_path or settings.rag.manifest_path)
        self.model_name = model_name or settings.rag.embedding_model
        self.collection = collection or settings.rag.collection
        if not manifest_path.exists():
            raise RuntimeError("Policy index missing. Run: python -m scripts.build_policy_index")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("policy_sha256") != policy_sha256(self.policy_path)
                or self.manifest.get("embedding_model") != self.model_name
                or self.manifest.get("collection") != self.collection
                or self.manifest.get("schema_version") != SCHEMA_VERSION):
            raise RuntimeError("Policy index is stale; rebuild with --rebuild.")
        if client is None:
            if not db_path.exists():
                raise RuntimeError("Policy database missing. Run manually: python -m scripts.build_policy_index")
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("Install Milvus: python -m pip install -r requirements.txt") from exc
            client = MilvusClient(uri=str(db_path))
        self.client = client
        if not client.has_collection(collection_name=self.collection):
            raise RuntimeError("Milvus policy collection missing; rebuild the index.")
        dimension = self.manifest.get("dimension")
        if type(dimension) is not int or dimension < 1:
            raise RuntimeError("Invalid index dimension; run python -m scripts.build_policy_index --rebuild")
        if hasattr(client, "describe_collection"):
            fields = client.describe_collection(collection_name=self.collection).get("fields", [])
            vector = next((field for field in fields if field.get("name") == "vector"), None)
            if vector is None or int(vector.get("params", {}).get("dim", 0)) != dimension:
                raise RuntimeError("Collection schema differs from manifest; run python -m scripts.build_policy_index --rebuild")
        # Milvus Lite persists the collection on disk but reopens it as released.
        # Search/query require an explicit load in *each new application process*.
        # Calling load_collection on an already-loaded collection is safe.
        self.client.load_collection(collection_name=self.collection)
        self.embedder = embedder or get_embedder(self.model_name)

    def search(self, item: str, top_k: int = 5, kind: str | None = None, *, timeout=None) -> list[dict]:
        """Return ranked real Milvus hits; scores are similarities, not confidence."""
        if not item or not item.strip():
            raise ValueError("Item name cannot be blank")
        if top_k < 1 or top_k > 45:
            raise ValueError("top_k must be between 1 and 45")
        if kind not in (None, "coverage", "exclusion"):
            raise ValueError("kind must be coverage, exclusion or None")
        vec = embed_texts(self.embedder, [item.strip()])[0]
        if len(vec) != self.manifest.get("dimension"):
            raise RuntimeError("Policy embedding dimension differs from the index manifest. "
                               "Run manually: python -m scripts.build_policy_index --rebuild")
        results = self.client.search(
            collection_name=self.collection,
            data=[vec],
            limit=top_k,
            filter=f'kind == "{kind}"' if kind else "",
            output_fields=["rule_id", "kind", "category", "conditions", "limit_type",
                           "max_amount", "unit", "source", "text"],
            **({"timeout": timeout} if timeout is not None else {}),
        )
        found = []
        for hit in (results[0] if results else []):
            entity = hit.get("entity", {})
            found.append({
                "rule_id": entity["rule_id"],
                "kind": entity["kind"],
                "category": entity["category"],
                "conditions": entity["conditions"],
                "limit_type": entity.get("limit_type", ""),
                "maximum_reimbursement": entity.get("max_amount", "") or None,
                "unit": entity.get("unit", ""),
                "source": entity.get("source", "data/policy.json"),
                "text": entity.get("text", ""),
                "score": float(hit.get("distance", 0.0)),
            })
        return found
