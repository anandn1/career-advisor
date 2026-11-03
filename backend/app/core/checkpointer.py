"""Postgres-backed LangGraph checkpointer, shared by graphs that pause and resume."""

from functools import cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings

POOL_MAX_SIZE = 10


@cache
def get_checkpointer() -> PostgresSaver:
    """Open the connection pool and create the checkpoint tables on first use."""
    pool = ConnectionPool(
        conninfo=get_settings().database_url,
        max_size=POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
    except Exception:
        pool.close()
        raise
    return checkpointer
