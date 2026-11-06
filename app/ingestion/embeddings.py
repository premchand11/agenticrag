import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import get_settings


@lru_cache
def _model() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model, cache_dir=settings.embedding_cache_dir)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    # The model is CPU-bound, so run it off the event loop.
    return await asyncio.to_thread(_embed_sync, texts)


def _embed_sync(texts: list[str]) -> list[list[float]]:
    expected = get_settings().embedding_dimensions
    vectors = [vector.tolist() for vector in _model().embed(texts)]
    if any(len(vector) != expected for vector in vectors):
        raise RuntimeError(f"embedding model returned vectors that are not {expected}-dimensional")
    return vectors
