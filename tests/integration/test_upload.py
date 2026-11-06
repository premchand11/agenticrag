from uuid import UUID, uuid4

import pytest
from app.core.config import get_settings
from app.core.database import session_factory
from app.ingestion.chunking.tokenizer import TokenizerUnavailable
from app.main import app
from app.models.chunk import Chunk
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

pytestmark = pytest.mark.integration


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
async def client(storage_dir):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def _chunk_rows(document_id: str) -> list[Chunk]:
    async with session_factory() as session:
        result = await session.scalars(
            select(Chunk).where(Chunk.document_id == UUID(document_id)).order_by(Chunk.chunk_index)
        )
        return list(result)


async def test_duplicate_upload_returns_the_same_document(client):
    name = f"{uuid4().hex}.html"
    body = f"<p>{uuid4().hex}</p>".encode()
    files = {"file": (name, body, "text/html")}
    first = await client.post("/documents/upload", files=files)
    second = await client.post("/documents/upload", files=files)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]


async def test_same_filename_with_new_bytes_creates_a_version(client):
    name = f"{uuid4().hex}.html"
    first = await client.post(
        "/documents/upload",
        files={"file": (name, f"<p>{uuid4().hex}</p>".encode(), "text/html")},
    )
    second = await client.post(
        "/documents/upload",
        files={"file": (name, f"<p>{uuid4().hex}</p>".encode(), "text/html")},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert second.json()["family_id"] == first.json()["family_id"]
    assert first.json()["document_id"] != second.json()["document_id"]


async def test_upload_records_the_extracted_page_count(client):
    name = f"{uuid4().hex}.html"
    body = f"<p>Redis caches {uuid4().hex}</p>".encode()
    response = await client.post("/documents/upload", files={"file": (name, body, "text/html")})
    payload = response.json()
    assert response.status_code == 201
    assert payload["status"] == "processed"
    assert payload["metadata"]["page_count"] == 1
    assert payload["error_message"] is None


async def test_upload_marks_a_blank_document_failed(client):
    name = f"{uuid4().hex}.html"
    body = f"<html><style>p{{{uuid4().hex}}}</style></html>".encode()
    response = await client.post("/documents/upload", files={"file": (name, body, "text/html")})
    payload = response.json()
    assert response.status_code == 201
    assert payload["status"] == "failed"
    assert payload["error_message"] == "no text extracted"


async def test_upload_stores_embedded_chunks(client):
    name = f"{uuid4().hex}.html"
    body = f"<p>Hybrid retrieval combines BM25 and vectors {uuid4().hex}.</p>".encode()
    response = await client.post("/documents/upload", files={"file": (name, body, "text/html")})
    payload = response.json()
    assert response.status_code == 201
    assert payload["chunk_count"] == 1

    rows = await _chunk_rows(payload["document_id"])
    assert [row.chunk_index for row in rows] == [0]
    assert rows[0].token_count > 0
    assert rows[0].chunk_metadata == {"page": 1}
    assert len(rows[0].embedding) == 384


async def test_long_document_is_split_into_contiguous_chunks(client):
    name = f"{uuid4().hex}.html"
    words = " ".join(f"term{i}" for i in range(600))
    body = f"<p>{words} {uuid4().hex}</p>".encode()
    response = await client.post("/documents/upload", files={"file": (name, body, "text/html")})
    payload = response.json()
    assert response.status_code == 201
    assert payload["chunk_count"] > 1

    rows = await _chunk_rows(payload["document_id"])
    assert [row.chunk_index for row in rows] == list(range(len(rows)))
    assert all(row.token_count > 0 for row in rows)


async def test_missing_tokenizer_fails_the_upload_and_keeps_no_file(client, storage_dir, monkeypatch):
    monkeypatch.setenv("TOKENIZER_PATH", str(storage_dir / "missing.json"))
    get_settings.cache_clear()
    name = f"{uuid4().hex}.html"
    body = f"<p>{uuid4().hex}</p>".encode()

    with pytest.raises(TokenizerUnavailable):
        await client.post("/documents/upload", files={"file": (name, body, "text/html")})

    assert [path for path in storage_dir.rglob("*") if path.is_file()] == []
