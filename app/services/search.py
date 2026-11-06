from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ingestion.embeddings import embed_texts
from app.models.chunk import Chunk
from app.models.document import Document
from app.retrieval.rrf import reciprocal_rank_fusion


class SearchRejected(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


@dataclass(frozen=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    version: int
    page: int | None
    content: str
    score: float


async def search_chunks(session: AsyncSession, query: str, limit: int) -> list[SearchHit]:
    query = query.strip()
    if not query:
        raise SearchRejected("query is empty")
    if limit < 1 or limit > 50:
        raise SearchRejected("limit must be between 1 and 50")

    settings = get_settings()
    candidates = max(limit, settings.search_candidate_limit)
    vector = (await embed_texts([query]))[0]
    vector_ids = await _vector_ids(session, vector, candidates)
    text_ids = await _text_ids(session, query, candidates)
    fused = reciprocal_rank_fusion([vector_ids, text_ids], k=settings.rrf_k, limit=limit)
    return await _load_hits(session, fused)


async def _vector_ids(session: AsyncSession, vector: list[float], limit: int) -> list[UUID]:
    distance = Chunk.embedding.cosine_distance(vector)
    statement = (
        select(Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.is_current.is_(True), Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    return list(await session.scalars(statement))


async def _text_ids(session: AsyncSession, query: str, limit: int) -> list[UUID]:
    # plainto_tsquery ANDs the words and ignores tsquery operators in the user text.
    tsquery = func.plainto_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
    statement = (
        select(Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.is_current.is_(True), Chunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    return list(await session.scalars(statement))


async def _load_hits(session: AsyncSession, fused: list) -> list[SearchHit]:
    if not fused:
        return []
    scores = {item.chunk_id: item.score for item in fused}
    rows = await session.execute(
        select(Chunk, Document).join(Document, Document.id == Chunk.document_id).where(Chunk.id.in_(scores))
    )
    found = {chunk.id: (chunk, document) for chunk, document in rows}
    return [
        SearchHit(
            chunk_id=chunk_id,
            document_id=document.id,
            filename=document.filename,
            version=document.version,
            page=chunk.chunk_metadata.get("page"),
            content=chunk.content,
            score=scores[chunk_id],
        )
        for chunk_id in scores
        if (pair := found.get(chunk_id)) is not None
        for chunk, document in [pair]
    ]
