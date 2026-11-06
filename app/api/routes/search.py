from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.security import require_api_key
from app.services.search import SearchRejected, search_chunks

router = APIRouter(dependencies=[Depends(require_api_key)])


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)


class SearchHitResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    version: int
    page: int | None
    content: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SearchResponse:
    try:
        hits = await search_chunks(session, body.query, body.limit)
    except SearchRejected as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.detail) from None
    return SearchResponse(
        hits=[
            SearchHitResponse(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                version=hit.version,
                page=hit.page,
                content=hit.content,
                score=hit.score,
            )
            for hit in hits
        ]
    )
