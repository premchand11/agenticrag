from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.security import require_api_key
from app.services.answers import answer_question
from app.services.llm import LlmError
from app.services.search import SearchRejected

router = APIRouter(dependencies=[Depends(require_api_key)])


class AnswerRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)


class CitationResponse(BaseModel):
    marker: int
    chunk_id: UUID
    document_id: UUID
    filename: str
    version: int
    page: int | None
    content: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@router.post("/answer", response_model=AnswerResponse)
async def answer(
    body: AnswerRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AnswerResponse:
    try:
        result = await answer_question(session, body.query, body.limit)
    except SearchRejected as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.detail) from None
    except LlmError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.detail) from None
    return AnswerResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                marker=citation.marker,
                chunk_id=citation.hit.chunk_id,
                document_id=citation.hit.document_id,
                filename=citation.hit.filename,
                version=citation.hit.version,
                page=citation.hit.page,
                content=citation.hit.content,
            )
            for citation in result.citations
        ],
    )
