# Agentic RAG

A document question-answering service. Upload stores each current document version as embedded chunks. `POST /search` returns ranked chunks, and `POST /answer` answers from them with citations. The agent is not built yet.

## Run the stack

From the repository root, with Docker running:

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8000/health
```

A healthy process returns HTTP 200:

```json
{"status":"ok","checks":{"postgres":"ok","redis":"ok"}}
```

If either dependency is unreachable, the same URL returns HTTP 503 and `"status": "degraded"`. The response does not include connection details.

| Service | Host address | Inside Compose |
|---|---|---|
| API | `http://127.0.0.1:8000` | — |
| Postgres | `localhost:5433` | `db:5432` |
| Redis | `localhost:6380` | `redis:6379` |

Host ports `5433` and `6380` avoid other Postgres and Redis processes already bound to `5432` and `6379`. The API container does not use those host ports. Compose sets `DATABASE_URL` and `REDIS_URL` to the service names above. The container entrypoint runs `alembic upgrade head` before Uvicorn.

Stop the stack with `docker compose down`. That keeps the `pgdata` volume. `docker compose down -v` deletes the database data.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

`requirements-dev.txt` installs `requirements.txt` and Ruff. `.env.example` points at the host ports. Copy it to `.env` for commands you run on the host. Do not commit `.env`.

```bash
pytest tests/unit -q
pytest -q
```

Unit tests do not need Docker. `pytest -q` also runs the integration tests. `tests/integration/test_health.py` skips when Postgres on `localhost:5433` or Redis on `localhost:6380` is closed. The upload, search, and answer integration tests need Postgres running, and they fail instead of skipping when it is down. Those tests also need `models/bge-small-en-v1.5/tokenizer.json` and the fastembed cache at `models/fastembed`.

Ruff checks and formats the code. Configuration is in `pyproject.toml`.

```bash
ruff check app tests
ruff format --check app tests
```

## Answers

`POST /answer` calls an OpenAI-compatible chat API. Set `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` in `.env`. For Ollama, use `LLM_BASE_URL=http://127.0.0.1:11434/v1` and leave `LLM_API_KEY` empty. If no passages match, `/answer` returns a fixed sentence without calling the model. If passages match and the model is not configured or fails, it returns 502. Search works without any LLM settings.

## Schema

The first migration enables the `vector` extension and creates `documents` and `chunks`.

A document family groups versions of one file. The first version uses its own `id` as `family_id`. A new version of the same filename keeps that `family_id`, increments `version`, and is the only row with `is_current` true. `content_sha256` is unique: uploading the same bytes again returns the existing document instead of creating a version. Filename is the version key for now.

Chunks belong to one document version. Deleting that version deletes its chunks. `embedding` is `vector(384)` for `BAAI/bge-small-en-v1.5`, searched with cosine distance. Changing the embedding model requires a migration. `content_tsv` is a stored generated column. Search reads it for keyword ranking and does not write it.

`GET /health` is unauthenticated. `POST /documents/upload`, `POST /search`, and `POST /answer` require the `X-API-Key` header when `API_KEY` is set. Leave `API_KEY` empty for local use.

## What this slice does not do

There is no agent, conversation memory, or evaluation harness. `app/agents` and `app/evaluation` are reserved for those stages.
