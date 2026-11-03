import logging
from dataclasses import dataclass

from sqlalchemy import text

from app.core.database import engine
from app.core.redis import redis_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthReport:
    status: str
    checks: dict[str, str]


async def _check(name: str, probe) -> str:
    try:
        await probe()
    except Exception as exc:
        logger.warning("%s health check failed: %s", name, exc.__class__.__name__)
        return "error"
    return "ok"


async def _postgres() -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def _redis() -> None:
    await redis_client.ping()


async def collect_health() -> HealthReport:
    checks = {
        "postgres": await _check("postgres", _postgres),
        "redis": await _check("redis", _redis),
    }
    return report_from_checks(checks)


def report_from_checks(checks: dict[str, str]) -> HealthReport:
    status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    return HealthReport(status=status, checks=checks)
