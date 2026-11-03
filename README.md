# Agentic RAG

A document question-answering service. This repository currently contains the foundation: an API, Postgres with pgvector, Redis, and the schema later stages will fill. Uploading, chunking, embeddings, retrieval, and the agent are not built yet.

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

`.env.example` points at the host ports. Copy it to `.env` for commands you run on the host. Do not commit `.env`.

```bash
pytest tests/unit -q
pytest -q
```

Unit tests do not need Docker. The integration test needs the Compose Postgres and Redis. It skips if `localhost:5433` or `localhost:6380` is closed.

Ruff checks and formats the code. Configuration is in `pyproject.toml`.

```bash
ruff check app tests
ruff format --check app tests
```

## Schema

The first migration enables the `vector` extension and creates `documents` and `chunks`.

A document family groups versions of one file. The first version uses its own `id` as `family_id`. A new version of the same filename keeps that `family_id`, increments `version`, and is the only row with `is_current` true. `content_sha256` is unique: uploading the same bytes again returns the existing document instead of creating a version. Filename is the version key for now.

Chunks belong to one document version. Deleting that version deletes its chunks. `embedding` is `vector(384)`, reserved for `BAAI/bge-small-en-v1.5` with cosine distance. Changing the embedding model requires a migration. `content_tsv` is a stored generated column for later keyword search. The application does not write it.

`GET /health` is unauthenticated. When `API_KEY` is set, later routes will require the `X-API-Key` header. Leave `API_KEY` empty for local use.

## What this slice does not do

There is no upload API, parser, chunker, embedding client, retrieval endpoint, or agent. `app/ingestion`, `app/retrieval`, `app/agents`, and `app/evaluation` are empty packages reserved for those stages.
