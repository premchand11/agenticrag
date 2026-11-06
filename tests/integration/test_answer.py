from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.main import app
from app.services.llm import LlmError
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


async def _upload(client: AsyncClient, filename: str, text: str):
    response = await client.post(
        "/documents/upload",
        files={"file": (filename, f"<p>{text}</p>".encode(), "text/html")},
    )
    assert response.status_code == 201
    return response.json()


async def test_answer_cites_the_retrieved_passage(client, monkeypatch):
    word = f"answerword{uuid4().hex}"
    uploaded = await _upload(client, f"{uuid4().hex}.html", f"{word} is the stored fact")
    seen: dict[str, str] = {}

    async def fake_complete(messages, *, client=None):
        seen["user"] = messages[1]["content"]
        return "The stored fact is in [1]."

    monkeypatch.setattr("app.services.answers.complete_chat", fake_complete)
    response = await client.post("/answer", json={"query": word, "limit": 5})
    payload = response.json()
    assert response.status_code == 200
    assert word in seen["user"]
    assert payload["answer"] == "The stored fact is in [1]."
    assert payload["citations"][0]["marker"] == 1
    assert payload["citations"][0]["document_id"] == uploaded["document_id"]


async def test_model_failure_returns_502(client, monkeypatch):
    word = f"answerword{uuid4().hex}"
    await _upload(client, f"{uuid4().hex}.html", word)

    async def broken(messages, *, client=None):
        raise LlmError("chat model request failed")

    monkeypatch.setattr("app.services.answers.complete_chat", broken)
    response = await client.post("/answer", json={"query": word, "limit": 5})
    assert response.status_code == 502
    assert response.json()["detail"] == "chat model request failed"
