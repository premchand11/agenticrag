import json
from typing import cast
from uuid import UUID

import httpx
import pytest
from app.core.config import get_settings
from app.services.answers import NO_PASSAGES, answer_question, citations_for
from app.services.llm import LlmError, complete_chat
from app.services.search import SearchHit
from sqlalchemy.ext.asyncio import AsyncSession


def _hit(number: int, content: str) -> SearchHit:
    return SearchHit(
        chunk_id=UUID(int=number),
        document_id=UUID(int=100 + number),
        filename=f"{number}.html",
        version=1,
        page=number,
        content=content,
        score=0.1,
    )


def test_citations_follow_first_use_and_ignore_unknown_markers():
    hits = [_hit(1, "alpha"), _hit(2, "beta")]
    citations = citations_for("See [2], then [2] again, then [1]. Ignore [9].", hits)
    assert [(item.marker, item.hit.content) for item in citations] == [(2, "beta"), (1, "alpha")]


async def test_no_passages_skips_the_model():
    async def retrieve(session, query, limit):
        return []

    async def complete(messages, *, client=None):
        raise AssertionError("model should not be called")

    result = await answer_question(
        cast(AsyncSession, None),
        "anything",
        5,
        retrieve=retrieve,
        complete=complete,
    )
    assert result.answer == NO_PASSAGES
    assert result.citations == []


async def test_answer_keeps_the_cited_passage():
    hits = [_hit(1, "alpha"), _hit(2, "beta")]

    async def retrieve(session, query, limit):
        return hits

    async def complete(messages, *, client=None):
        assert "Question: where?" in messages[1]["content"]
        assert "[1]" in messages[1]["content"]
        return "It is in [2]."

    result = await answer_question(
        cast(AsyncSession, None),
        "where?",
        5,
        retrieve=retrieve,
        complete=complete,
    )
    assert result.citations[0].marker == 2
    assert result.citations[0].hit.content == "beta"


@pytest.fixture
def llm_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_complete_chat_posts_to_the_configured_endpoint(llm_settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://llm.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["temperature"] == 0
        return httpx.Response(200, json={"choices": [{"message": {"content": "  cited [1]  "}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await complete_chat([{"role": "user", "content": "question"}], client=client)
    assert text == "cited [1]"


async def test_complete_chat_omits_the_header_when_no_key_is_set(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = await complete_chat([{"role": "user", "content": "question"}], client=client)
    assert text == "ok"
    get_settings.cache_clear()


async def test_missing_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    get_settings.cache_clear()
    with pytest.raises(LlmError, match="LLM_BASE_URL"):
        await complete_chat([{"role": "user", "content": "question"}])
    get_settings.cache_clear()
