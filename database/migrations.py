"""Idempotent PostgreSQL schema upgrades for existing installations."""

from sqlalchemy import inspect, text

from database.connection import Base, engine


def upgrade() -> None:
    """Create current tables and add constraints missing from older schemas."""
    from database import models

    _ = models

    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    existing = {
        constraint["name"]
        for table in inspector.get_table_names()
        for constraint in inspector.get_check_constraints(table)
        if constraint.get("name")
    }

    statements = {
        "ck_messages_role": (
            "ALTER TABLE messages ADD CONSTRAINT ck_messages_role "
            "CHECK (role IN ('user', 'assistant', 'system', 'tool'))"
        ),
        "ck_agent_runs_status": (
            "ALTER TABLE agent_runs ADD CONSTRAINT ck_agent_runs_status "
            "CHECK (status IN ('pending', 'running', 'waiting_approval', "
            "'completed', 'rejected', 'failed', 'interrupted', "
            "'limit_reached'))"
        ),
        "ck_tool_executions_status": (
            "ALTER TABLE tool_executions ADD CONSTRAINT "
            "ck_tool_executions_status CHECK "
            "(status IN ('running', 'completed', 'failed'))"
        ),
        "ck_approval_requests_status": (
            "ALTER TABLE approval_requests ADD CONSTRAINT "
            "ck_approval_requests_status CHECK "
            "(status IN ('pending', 'approved', 'rejected'))"
        ),
        "ck_file_changes_status": (
            "ALTER TABLE file_changes ADD CONSTRAINT ck_file_changes_status "
            "CHECK (status IN ('completed', 'failed', 'rejected'))"
        ),
    }

    if engine.dialect.name != "postgresql":
        return

    with engine.begin() as connection:
        for name, statement in statements.items():
            if name not in existing:
                connection.execute(text(statement))
