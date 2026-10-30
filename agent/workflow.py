"""Shared persistence helpers for CLI and Streamlit agent workflows."""

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip())
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"].strip()
    return str(content)


def record_tool_messages(
    repository: Any,
    run_id: str,
    messages: list[Any],
    logged_message_ids: set[str] | None = None,
) -> set[str]:
    """Persist each tool result once and return the updated ID set."""
    if logged_message_ids is None:
        logged_message_ids = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        message_id = str(
            getattr(message, "id", None)
            or f"{getattr(message, 'tool_call_id', '')}:{message.content!s}"
        )
        if message_id in logged_message_ids:
            continue

        logged_message_ids.add(message_id)
        repository.record_tool(
            run_id=run_id,
            tool_name=getattr(message, "name", None) or "unknown",
            status="completed",
            arguments={
                "tool_call_id": getattr(message, "tool_call_id", None)
            },
            result_summary=message_text(message.content)[:4000],
        )

    return logged_message_ids


def record_successful_file_changes(
    repository: Any,
    run_id: str,
    messages: list[Any],
) -> None:
    """Persist structured successful write/edit tool results."""
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) not in {"write_file", "edit_file"}:
            continue

        try:
            result = json.loads(message_text(message.content))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if not isinstance(result, dict):
            continue
        if str(result.get("status", "")).lower() not in {
            "success",
            "applied",
            "completed",
        }:
            continue

        required = (
            "relative_path",
            "operation",
            "proposed_sha256",
        )
        if any(not result.get(field) for field in required):
            continue

        repository.record_file_change(
            run_id=run_id,
            relative_path=result["relative_path"],
            operation=result["operation"],
            original_sha256=result.get("original_sha256"),
            proposed_sha256=result["proposed_sha256"],
            backup_relative_path=result.get("backup_relative_path"),
            status="completed",
        )


def save_final_response(
    repository: Any,
    conversation_id: str,
    messages: list[Any],
) -> str:
    """Persist the final assistant message and return it."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(
            message, "tool_calls", None
        ):
            response = message_text(message.content)
            if response:
                break
    else:
        response = "The agent finished without returning a text response."

    repository.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response,
    )
    return response


def finish_run(
    repository: Any,
    run_id: str,
    status: str,
    *,
    response: str | None = None,
    error: str | None = None,
) -> None:
    """Use one stable repository contract for all frontends."""
    repository.finish_run(
        run_id=run_id,
        status=status,
        response=response,
        error=error,
    )
