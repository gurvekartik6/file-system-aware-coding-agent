
"""Database operations for conversations, tasks, and audit records."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select

from database.connection import SessionLocal
from database.models import (
    AgentRun,
    ApprovalRequest,
    Conversation,
    FileChange,
    Message,
    ToolExecution,
)


def new_id() -> str:
    return str(uuid4())


def now_utc():
    return datetime.now(timezone.utc)


class AgentRepository:
    """Small repository for the coding-agent application."""

    def create_conversation(self, title: str = "New conversation") -> str:
        conversation_id = new_id()

        with SessionLocal.begin() as session:
            session.add(
                Conversation(
                    id=conversation_id,
                    title=title[:250],
                )
            )

        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> str:
        message_id = new_id()

        with SessionLocal.begin() as session:
            conversation = session.get(Conversation, conversation_id)

            if conversation is None:
                raise ValueError("Conversation does not exist.")

            session.add(
                Message(
                    id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
            )
            conversation.updated_at = now_utc()

        return message_id

    def create_run(
        self,
        conversation_id: str,
        thread_id: str,
        task: str,
    ) -> str:
        run_id = new_id()

        with SessionLocal.begin() as session:
            if session.get(Conversation, conversation_id) is None:
                raise ValueError("Conversation does not exist.")

            session.add(
                AgentRun(
                    id=run_id,
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    task=task,
                    status="running",
                )
            )

        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        allowed = {
            "completed",
            "waiting_approval",
            "rejected",
            "failed",
            "interrupted",
            "limit_reached",
        }

        if status not in allowed:
            raise ValueError(f"Unsupported run status: {status}")

        with SessionLocal.begin() as session:
            run = session.get(AgentRun, run_id)

            if run is None:
                raise ValueError("Agent run does not exist.")

            if run.status in {"completed", "failed", "rejected", "interrupted"}:
                if status != run.status:
                    raise ValueError(
                        f"Run is already terminal with status {run.status}."
                    )

            run.status = status
            run.final_response = response
            run.error_message = error

            if status not in {"waiting_approval", "interrupted"}:
                run.finished_at = now_utc()
            else:
                run.finished_at = None

    def list_pending_approvals(self, conversation_id: str | None = None) -> list[dict]:
        """Return pending approvals for recovery after a UI/process restart."""
        with SessionLocal() as session:
            query = (
                select(ApprovalRequest, AgentRun)
                .join(AgentRun, ApprovalRequest.run_id == AgentRun.id)
                .where(ApprovalRequest.status == "pending")
                .order_by(ApprovalRequest.created_at)
            )
            if conversation_id is not None:
                query = query.where(
                    AgentRun.conversation_id == conversation_id
                )

            return [
                {
                    "id": approval.id,
                    "run_id": run.id,
                    "thread_id": run.thread_id,
                    "conversation_id": run.conversation_id,
                    "relative_path": approval.relative_path,
                    "operation": approval.operation,
                    "original_sha256": approval.original_sha256,
                    "proposed_sha256": approval.proposed_sha256,
                    "diff": approval.diff,
                }
                for approval, run in session.execute(query).all()
            ]

    def record_tool(
        self,
        run_id: str,
        tool_name: str,
        status: str,
        arguments: dict | None = None,
        result_summary: str | None = None,
    ) -> str:
        tool_id = new_id()

        with SessionLocal.begin() as session:
            if session.get(AgentRun, run_id) is None:
                raise ValueError("Agent run does not exist.")

            session.add(
                ToolExecution(
                    id=tool_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    status=status,
                    arguments_summary=arguments or {},
                    result_summary=(
                        result_summary[:4000]
                        if result_summary is not None
                        else None
                    ),
                    finished_at=now_utc(),
                )
            )

        return tool_id

    def record_approval(
        self,
        run_id: str,
        relative_path: str,
        operation: str,
        original_sha256: str | None,
        proposed_sha256: str,
        diff: str | None,
        status: str = "pending",
        approval_id: str | None = None,
    ) -> str:
        """Record an approval request using a stable ID when provided."""

        approval_id = approval_id or new_id()

        with SessionLocal.begin() as session:
            existing = session.get(ApprovalRequest, approval_id)

            if existing is not None:
                if (
                    existing.run_id != run_id
                    or existing.relative_path != relative_path
                    or existing.operation != operation
                    or existing.proposed_sha256 != proposed_sha256
                ):
                    raise ValueError(
                        "Approval ID already belongs to a different proposal."
                    )

                return approval_id

            if session.get(AgentRun, run_id) is None:
                raise ValueError("Agent run does not exist.")

            session.add(
                ApprovalRequest(
                    id=approval_id,
                    run_id=run_id,
                    relative_path=relative_path,
                    operation=operation,
                    original_sha256=original_sha256,
                    proposed_sha256=proposed_sha256,
                    diff=diff,
                    status=status,
                    decision_at=(
                        now_utc() if status != "pending" else None
                    ),
                )
            )

        return approval_id

    def decide_approval(self, approval_id: str, status: str) -> None:
        if status not in {"approved", "rejected"}:
            raise ValueError("Approval status must be approved or rejected.")

        with SessionLocal.begin() as session:
            approval = session.get(ApprovalRequest, approval_id)

            if approval is None:
                raise ValueError("Approval request does not exist.")

            if approval.status != "pending":
                raise ValueError("Approval request has already been decided.")

            approval.status = status
            approval.decision_at = now_utc()

    def record_file_change(
        self,
        run_id: str,
        relative_path: str,
        operation: str,
        original_sha256: str | None,
        proposed_sha256: str,
        backup_relative_path: str | None,
        status: str,
    ) -> str:
        change_id = new_id()

        with SessionLocal.begin() as session:
            if session.get(AgentRun, run_id) is None:
                raise ValueError("Agent run does not exist.")

            session.add(
                FileChange(
                    id=change_id,
                    run_id=run_id,
                    relative_path=relative_path,
                    operation=operation,
                    original_sha256=original_sha256,
                    proposed_sha256=proposed_sha256,
                    backup_relative_path=backup_relative_path,
                    status=status,
                )
            )

        return change_id

    def get_conversation_history(
        self,
        conversation_id: str,
    ) -> list[dict]:
        with SessionLocal() as session:
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
            ).all()

            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def list_conversations(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(limit, 100))

        with SessionLocal() as session:
            rows = session.scalars(
                select(Conversation)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            ).all()

            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def list_runs(self, conversation_id: str) -> list[dict]:
        with SessionLocal() as session:
            rows = session.scalars(
                select(AgentRun)
                .where(AgentRun.conversation_id == conversation_id)
                .order_by(AgentRun.started_at.desc())
            ).all()

            return [
                {
                    "id": row.id,
                    "thread_id": row.thread_id,
                    "task": row.task,
                    "status": row.status,
                    "started_at": row.started_at.isoformat(),
                    "finished_at": (
                        row.finished_at.isoformat()
                        if row.finished_at
                        else None
                    ),
                    "final_response": row.final_response,
                    "error_message": row.error_message,
                }
                for row in rows
            ]
