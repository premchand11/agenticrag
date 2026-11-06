import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ingestion.chunking.splitter import chunk_document
from app.ingestion.chunking.tokenizer import load_tokenizer
from app.ingestion.embeddings import embed_texts
from app.ingestion.files import UploadRejected, UploadTooLarge, file_type_for, safe_filename, sha256_hex
from app.ingestion.parsers.extract import ExtractedDocument, ExtractionError, extract_document
from app.models.chunk import Chunk
from app.models.document import Document


async def upload_document(session: AsyncSession, filename: str, content: bytes) -> tuple[Document, bool]:
    settings = get_settings()
    if len(content) > settings.max_upload_bytes:
        raise UploadTooLarge  # route maps size separately; see below
    safe_name = safe_filename(filename)
    file_type = file_type_for(safe_name)
    if not content:
        raise UploadRejected("file is empty")

    digest = sha256_hex(content)
    existing = await session.scalar(select(Document).where(Document.content_sha256 == digest))
    if existing is not None:
        return existing, True

    current = await session.scalar(
        select(Document).where(Document.filename == safe_name, Document.is_current.is_(True))
    )
    document_id = uuid4()
    if current is None:
        family_id = document_id
        version = 1
    else:
        current.is_current = False
        family_id = current.family_id
        version = current.version + 1

    storage_key = f"{document_id}/{safe_name}"
    path = Path(settings.storage_dir) / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    document = Document(
        id=document_id,
        family_id=family_id,
        version=version,
        is_current=True,
        filename=safe_name,
        file_type=file_type,
        content_sha256=digest,
        byte_size=len(content),
        storage_key=storage_key,
        status="uploaded",
    )
    session.add(document)
    try:
        extracted = extract_document(file_type, content)
    except ExtractionError as exc:
        document.status = "failed"
        document.error_message = exc.detail
    else:
        document.doc_metadata = {"page_count": len(extracted.pages)}
        document.status = "processed"
        try:
            chunks = await _build_chunks(document_id, extracted)
        except Exception:
            # Infrastructure failure (tokenizer or embedding model): commit nothing and remove the file.
            await session.rollback()
            path.unlink(missing_ok=True)
            raise
        await session.flush()
        session.add_all(chunks)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        path.unlink(missing_ok=True)
        raise
    await session.refresh(document)
    return document, False


async def count_chunks(session: AsyncSession, document_id: UUID) -> int:
    count = await session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id))
    return count or 0


async def _build_chunks(document_id: UUID, extracted: ExtractedDocument) -> list[Chunk]:
    settings = get_settings()
    tokenizer = load_tokenizer(settings.tokenizer_path)
    pieces = await asyncio.to_thread(
        chunk_document,
        extracted,
        tokenizer,
        settings.chunk_max_tokens,
        settings.chunk_overlap_tokens,
    )
    vectors = await embed_texts([piece.content for piece in pieces])
    return [
        Chunk(
            document_id=document_id,
            chunk_index=piece.chunk_index,
            content=piece.content,
            token_count=piece.token_count,
            chunk_metadata={"page": piece.page},
            embedding=vector,
        )
        for piece, vector in zip(pieces, vectors, strict=True)
    ]
