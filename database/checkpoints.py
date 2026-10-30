
"""PostgreSQL checkpoint saver lifecycle."""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver


@contextmanager
def postgres_checkpointer():
    """Yield a configured PostgresSaver for the application's lifetime."""

    load_dotenv()

    database_url = os.getenv("LANGGRAPH_CHECKPOINT_URL")

    if not database_url:
        raise RuntimeError(
            "LANGGRAPH_CHECKPOINT_URL is missing from .env."
        )

    # Checkpoint serialization should only deserialize known-safe types.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    pool = ConnectionPool[Connection[dict]](
        conninfo=database_url,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        },
        open=False,
    )

    try:
        pool.open(wait=True, timeout=30)

        saver = PostgresSaver(pool)
        saver.setup()

        yield saver

    finally:
        pool.close()
