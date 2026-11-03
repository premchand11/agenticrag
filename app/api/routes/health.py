from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.services.health import collect_health

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, Literal["ok", "error"]]


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    report = await collect_health()
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status=report.status, checks=report.checks)
