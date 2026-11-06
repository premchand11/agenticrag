import pytest
from app.core.database import engine


@pytest.fixture(autouse=True)
async def dispose_db_connections():
    yield
    await engine.dispose()
