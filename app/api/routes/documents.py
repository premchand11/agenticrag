from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.security import require_api_key
from app.ingestion.files import UploadRejected, UploadTooLarge
from app.services.documents import count_chunks, upload_document

router = APIRouter(prefix="/documents", dependencies=[Depends(require_api_key)])


class DocumentResponse(BaseModel):
    document_id: UUID
    family_id: UUID
    filename: str
    file_type: str
    version: int
    is_current: bool
    status: str
    byte_size: int
    content_sha256: str
    duplicate: bool
    chunk_count: int = 0
    created_at: datetime
    metadata: dict
    error_message: str | None = None


@router.post("/upload", response_model=DocumentResponse)
async def upload(
    response: Response,
    file: UploadFile,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DocumentResponse:
    content = await file.read()
    try:
        document, duplicate = await upload_document(session, file.filename or "", content)
    except UploadTooLarge:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="file is too large") from None
    except UploadRejected as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.detail) from None
    chunk_count = await count_chunks(session, document.id)
    if not duplicate:
        response.status_code = status.HTTP_201_CREATED
    return DocumentResponse(
        document_id=document.id,
        family_id=document.family_id,
        filename=document.filename,
        file_type=document.file_type,
        version=document.version,
        is_current=document.is_current,
        status=document.status,
        byte_size=document.byte_size,
        content_sha256=document.content_sha256,
        duplicate=duplicate,
        chunk_count=chunk_count,
        created_at=document.created_at,
        metadata=document.doc_metadata,
        error_message=document.error_message,
    )
