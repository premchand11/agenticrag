import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LlmError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


async def complete_chat(messages: list[dict[str, str]], *, client: httpx.AsyncClient | None = None) -> str:
    settings = get_settings()
    base_url = settings.llm_base_url.strip()
    model = settings.llm_model.strip()
    if not base_url:
        raise LlmError("LLM_BASE_URL is not set")
    if not model:
        raise LlmError("LLM_MODEL is not set")

    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {"model": model, "messages": messages, "temperature": 0}
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as owned:
                response = await owned.post(url, headers=headers, json=payload)
        else:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("chat model request failed: %s", type(exc).__name__)
        raise LlmError("chat model request failed") from exc
    if not isinstance(content, str) or not content.strip():
        raise LlmError("chat model returned an empty answer")
    return content.strip()
