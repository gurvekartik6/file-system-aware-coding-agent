
"""SQLAlchemy engine and session management."""

import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""

    pass


def get_database_url() -> str:
    """Return the configured SQLAlchemy database URL."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is missing. Configure it in .env."
        )

    return url


database_url = get_database_url()
engine_options: dict[str, Any] = {"pool_pre_ping": True}

if not database_url.startswith("sqlite"):
    engine_options.update(
        pool_size=5,
        max_overflow=5,
        pool_timeout=30,
    )

engine = create_engine(database_url, **engine_options)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def check_database_connection() -> bool:
    """Check that PostgreSQL is reachable."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True


def create_tables() -> None:
    """Create or upgrade the ORM schema."""
    from database.migrations import upgrade

    upgrade()


def get_session():
    """Provide a transaction-scoped session."""
    return SessionLocal()
