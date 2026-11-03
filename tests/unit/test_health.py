import pytest
from app.core.config import get_settings
from app.main import app
from app.services.health import HealthReport, report_from_checks
from httpx import ASGITransport, AsyncClient


def test_report_is_ok_when_every_check_passes():
    report = report_from_checks({"postgres": "ok", "redis": "ok"})
    assert report.status == "ok"


def test_report_is_degraded_when_any_check_fails():
    report = report_from_checks({"postgres": "error", "redis": "ok"})
    assert report.status == "degraded"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def test_health_returns_503_when_a_check_fails(client, monkeypatch):
    async def fake_collect_health():
        return HealthReport(status="degraded", checks={"postgres": "error", "redis": "ok"})

    monkeypatch.setattr("app.api.routes.health.collect_health", fake_collect_health)
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "checks": {"postgres": "error", "redis": "ok"}}


async def test_health_stays_open_when_an_api_key_is_configured(client, monkeypatch):
    async def fake_collect_health():
        return HealthReport(status="ok", checks={"postgres": "ok", "redis": "ok"})

    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    monkeypatch.setattr("app.api.routes.health.collect_health", fake_collect_health)
    response = await client.get("/health")
    assert response.status_code == 200
