import socket

import pytest
from alembic import command
from alembic.config import Config
from app.core.database import engine
from app.main import app
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

pytestmark = pytest.mark.integration

EXPECTED_INDEXES = {
    "pk_documents",
    "pk_chunks",
    "uq_documents_content_sha256",
    "uq_documents_family_version",
    "uq_documents_one_current_per_family",
    "uq_documents_one_current_per_filename",
    "uq_chunks_document_index",
    "ix_chunks_embedding_hnsw",
    "ix_chunks_content_tsv",
}


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def database_ready():
    if not _port_open(5433) or not _port_open(6380):
        pytest.skip("Postgres on localhost:5433 and Redis on localhost:6380 must be running")
    command.upgrade(Config("alembic.ini"), "head")


async def test_schema_and_health(database_ready):
    async with engine.connect() as connection:
        extension = await connection.scalar(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        table_rows = await connection.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename IN ('documents', 'chunks')")
        )
        index_rows = await connection.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename IN ('documents', 'chunks')")
        )
    tables = {row[0] for row in table_rows}
    indexes = {row[0] for row in index_rows}

    assert extension == "vector"
    assert tables == {"documents", "chunks"}
    assert EXPECTED_INDEXES <= indexes

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["checks"] == {"postgres": "ok", "redis": "ok"}
