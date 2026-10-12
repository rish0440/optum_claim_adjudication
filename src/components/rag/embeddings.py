"""CPU embeddings for the small policy knowledge base; no LLM/API key needed."""

from functools import lru_cache


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    """Load a cached SentenceTransformer on the CPU (downloads once initially)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Policy RAG requires sentence-transformers. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(embedder, texts: list[str]) -> list[list[float]]:
    """Compute normalized cosine-search vectors; support small test doubles."""
    if not texts:
        return []
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[float(value) for value in row] for row in vectors]
