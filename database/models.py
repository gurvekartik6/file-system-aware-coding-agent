
"""SQLAlchemy models for coding-agent persistence."""

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(250), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(30))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="ck_messages_role",
        ),
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages"
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    thread_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    final_response: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'waiting_approval', "
            "'completed', 'rejected', 'failed', 'interrupted', "
            "'limit_reached')",
            name="ck_agent_runs_status",
        ),
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="runs"
    )
    tool_executions: Mapped[list["ToolExecution"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )
    approval_requests: Mapped[list["ApprovalRequest"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )
    file_changes: Mapped[list["FileChange"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30))
    arguments_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    result_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_tool_executions_status",
        ),
    )

    run: Mapped["AgentRun"] = relationship(
        back_populates="tool_executions"
    )


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    relative_path: Mapped[str] = mapped_column(String(1000))
    operation: Mapped[str] = mapped_column(String(30))
    original_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    proposed_sha256: Mapped[str] = mapped_column(String(64))
    diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    decision_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_approval_requests_status",
        ),
    )

    run: Mapped["AgentRun"] = relationship(
        back_populates="approval_requests"
    )


class FileChange(Base):
    __tablename__ = "file_changes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    relative_path: Mapped[str] = mapped_column(String(1000))
    operation: Mapped[str] = mapped_column(String(30))
    original_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    proposed_sha256: Mapped[str] = mapped_column(String(64))
    backup_relative_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True
    )
    status: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('completed', 'failed', 'rejected')",
            name="ck_file_changes_status",
        ),
    )

    run: Mapped["AgentRun"] = relationship(
        back_populates="file_changes"
    )


Index("ix_messages_conversation_created", Message.conversation_id, Message.created_at)
Index("ix_runs_conversation_started", AgentRun.conversation_id, AgentRun.started_at)
