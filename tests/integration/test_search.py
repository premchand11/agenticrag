from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.main import app
from httpx import ASGITransport, AsyncClient

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


async def test_blank_query_is_rejected(client):
    response = await client.post("/search", json={"query": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "query is empty"


async def test_search_returns_the_current_version_only(client):
    name = f"{uuid4().hex}.html"
    old_word = f"oldword{uuid4().hex}"
    new_word = f"newword{uuid4().hex}"
    first = await client.post(
        "/documents/upload",
        files={"file": (name, f"<p>{old_word}</p>".encode(), "text/html")},
    )
    second = await client.post(
        "/documents/upload",
        files={"file": (name, f"<p>{new_word} is the current text</p>".encode(), "text/html")},
    )
    assert first.status_code == 201
    assert second.status_code == 201

    stale = await client.post("/search", json={"query": old_word, "limit": 5})
    assert stale.status_code == 200
    assert all(hit["document_id"] != first.json()["document_id"] for hit in stale.json()["hits"])

    found = await client.post("/search", json={"query": new_word, "limit": 5})
    hits = found.json()["hits"]
    assert found.status_code == 200
    match = next(hit for hit in hits[:2] if hit["document_id"] == second.json()["document_id"])
    assert match["version"] == 2
    assert match["page"] == 1
    assert new_word in match["content"]
    assert match["score"] > 0
