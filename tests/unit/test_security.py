import pytest
from app.core.config import get_settings
from app.core.security import require_api_key
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_allows_requests_when_api_key_is_unset(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()
    await require_api_key(None)


async def test_rejects_a_missing_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(None)
    assert exc_info.value.status_code == 401


async def test_rejects_a_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key("nope")
    assert exc_info.value.status_code == 401


async def test_accepts_a_matching_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    await require_api_key("secret")
