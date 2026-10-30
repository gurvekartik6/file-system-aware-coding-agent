
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from agent.workflow import (
    finish_run as workflow_finish_run,
    record_successful_file_changes as workflow_record_file_changes,
    record_tool_messages as workflow_record_tool_messages,
    save_final_response,
)

logger = logging.getLogger(__name__)


def load_configuration() -> None:
    """Load environment variables and validate required settings."""

    load_dotenv()

    if not (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    ):
        raise RuntimeError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    if not os.getenv("GEMINI_MODEL"):
        raise RuntimeError(
            "Missing GEMINI_MODEL in your .env file. "
            "Set it to a Gemini model available to your API key."
        )

    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("Missing DATABASE_URL in your .env file.")

    if not os.getenv("LANGGRAPH_CHECKPOINT_URL"):
        raise RuntimeError(
            "Missing LANGGRAPH_CHECKPOINT_URL in your .env file."
        )


def create_model() -> Any:
    """Create the Gemini chat model."""

    from langchain_google_genai import ChatGoogleGenerativeAI
    from model_config import sampling_kwargs

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    return ChatGoogleGenerativeAI(
        model=os.environ["GEMINI_MODEL"],
        google_api_key=api_key,
        max_retries=2,
        **sampling_kwargs(),
    )


def object_id(value: Any) -> str:
    """Extract an ID from a returned ORM object, dictionary, or string."""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("id", "run_id", "conversation_id"):
            if key in value:
                return str(value[key])

    for key in ("id", "run_id", "conversation_id"):
        if hasattr(value, key):
            return str(getattr(value, key))

    raise ValueError(
        f"Could not extract an ID from repository result: {value!r}"
    )


def stringify_content(content: Any) -> str:
    """Convert message content into readable text."""

    if isinstance(content, str):
        return content

    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def extract_interrupts(result: dict[str, Any]) -> list[Any]:
    """Extract approval interrupts returned by LangGraph."""

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        return []

    if isinstance(interrupts, (list, tuple)):
        return list(interrupts)

    return [interrupts]


def interrupt_payload(interrupt: Any) -> dict[str, Any]:
    """Get the payload from a LangGraph interrupt object."""

    value = getattr(interrupt, "value", None)

    if value is None and isinstance(interrupt, dict):
        value = interrupt.get("value", interrupt)

    if not isinstance(value, dict):
        raise ValueError(
            "The agent returned an unsupported approval payload."
        )

    return value


def display_approval(payload: dict[str, Any]) -> bool:
    """Show the proposed change and ask the user to approve or reject it."""

    print("\n" + "=" * 65)
    print("FILE MODIFICATION REQUIRES APPROVAL")
    print("=" * 65)
    print(f"Operation : {payload.get('operation', 'unknown')}")
    print(f"File      : {payload.get('relative_path', 'unknown')}")

    diff = payload.get("diff")

    if diff:
        print("\nProposed changes:\n")
        print(diff)
    else:
        print("\nNo diff was included in the approval payload.")

    print("\nDo you want to apply this change?")
    print("1. Approve")
    print("2. Reject")

    while True:
        choice = input("Choose 1 or 2: ").strip()

        if choice == "1":
            return True

        if choice == "2":
            return False

        print("Invalid choice. Enter 1 or 2.")


def log_tool_messages(
    repository: Any,
    run_id: str,
    messages: list[Any],
    logged_message_ids: set[str],
) -> None:
    """
    Persist tool results.

    ToolExecution records are useful for auditing. This function uses
    the repository's record_tool method from Part 4.
    """

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
            result_summary=stringify_content(message.content)[:4000],
        )


def save_file_change_records(
    repository: Any,
    run_id: str,
    messages: list[Any],
) -> None:
    """
    Save successful file-change results when a tool returns structured JSON.

    This expects successful write/edit tools to return JSON containing:
    relative_path, operation, status, original_sha256, proposed_sha256,
    and optionally backup_relative_path.

    Unstructured or unsuccessful tool responses are not recorded as
    successful file changes.
    """

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        if getattr(message, "name", None) not in {
            "write_file",
            "edit_file",
        }:
            continue

        try:
            result = json.loads(stringify_content(message.content))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if not isinstance(result, dict):
            continue

        status = str(result.get("status", "")).lower()

        if status not in {"success", "applied", "completed"}:
            continue

        relative_path = result.get("relative_path")
        operation = result.get("operation")

        if not relative_path or not operation:
            continue

        repository.record_file_change(
            run_id=run_id,
            relative_path=relative_path,
            operation=operation,
            original_sha256=result.get("original_sha256"),
            proposed_sha256=result.get("proposed_sha256"),
            backup_relative_path=result.get("backup_relative_path"),
            status="completed",
        )


def save_assistant_response(
    repository: Any,
    conversation_id: str,
    messages: list[Any],
) -> str:
    """Save the final assistant response and return its text."""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(
            message, "tool_calls", None
        ):
            response = stringify_content(message.content)

            if response.strip():
                repository.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response,
                )

                return response

    response = "The agent finished without returning a text response."

    repository.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response,
    )

    return response


def save_approval_request(
    repository: Any,
    run_id: str,
    payload: dict[str, Any],
) -> str:
    """
    Record an approval request idempotently.

    ApprovalRequest IDs are deterministic for the same run and proposed
    change, so resuming the same interrupt does not create a new ID.
    """

    run_uuid = uuid.UUID(run_id)
    approval_id = str(
        uuid.uuid5(
            run_uuid,
            "|".join(
                [
                    str(payload.get("operation", "")),
                    str(payload.get("relative_path", "")),
                    str(payload.get("proposed_sha256", "")),
                ]
            ),
        )
    )

    repository.record_approval(
        run_id=run_id,
        relative_path=str(payload.get("relative_path", "")),
        operation=str(payload.get("operation", "")),
        original_sha256=payload.get("original_sha256"),
        proposed_sha256=payload.get("proposed_sha256"),
        diff=str(payload.get("diff", "")),
        approval_id=approval_id,
    )

    return approval_id


def set_run_status(
    repository: Any,
    run_id: str,
    status: str,
    *,
    final_response: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update the run status using the Part 4 repository."""

    repository.finish_run(
        run_id=run_id,
        status=status,
        response=final_response,
        error=error_message,
    )


def run_task(
    graph: Any,
    repository: Any,
    conversation_id: str,
    request: str,
) -> None:
    """Run one user request, handling approval interrupts when needed."""

    thread_id = str(uuid.uuid4())

    run_record = repository.create_run(
        conversation_id=conversation_id,
        thread_id=thread_id,
        task=request,
    )
    run_id = object_id(run_record)

    repository.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 50,
    }

    logged_message_ids: set[str] = set()

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request)],
                "tool_rounds": 0,
            },
            config=config,
        )

        while True:
            messages = result.get("messages", [])

            workflow_record_tool_messages(
                repository,
                run_id,
                messages,
                logged_message_ids,
            )
            workflow_record_file_changes(repository, run_id, messages)

            pending_interrupts = extract_interrupts(result)

            if not pending_interrupts:
                break

            # Process each returned interrupt in order. The current
            # approval tool is expected to request one file change at a time.
            resume_values = []

            for pending in pending_interrupts:
                payload = interrupt_payload(pending)

                approval_id = save_approval_request(
                    repository,
                    run_id,
                    payload,
                )

                approved = display_approval(payload)

                decision = "approved" if approved else "rejected"

                repository.decide_approval(
                    approval_id=approval_id,
                    status=decision,
                )

                resume_values.append(
                    {
                        **payload,
                        "decision": "approve" if approved else "reject",
                    }
                )

            # LangGraph resumes the same checkpointed thread.
            # For a single interrupt, pass its resume value directly.
            if len(resume_values) == 1:
                result = graph.invoke(
                    Command(resume=resume_values[0]),
                    config=config,
                )
            else:
                # Multiple simultaneous interrupts require mapping each
                # resume value to its interrupt ID. This CLI deliberately
                # refuses to guess the mapping.
                raise RuntimeError(
                    "Multiple simultaneous approval interrupts are not "
                    "supported by this CLI. Configure sequential approval "
                    "handling before enabling parallel write operations."
                )

        response = save_final_response(
            repository,
            conversation_id,
            result.get("messages", []),
        )

        workflow_finish_run(
            repository,
            run_id,
            "completed",
            response=response,
        )

        print("\n" + "-" * 65)
        print("AGENT RESPONSE")
        print("-" * 65)
        print(response)

    except KeyboardInterrupt:
        workflow_finish_run(
            repository,
            run_id,
            "interrupted",
            error="Cancelled by user from the terminal.",
        )
        print("\nRun interrupted by the user.")

    except Exception as exc:
        try:
            workflow_finish_run(
                repository,
                run_id,
                "failed",
                error=str(exc)[:4000],
            )
        except Exception as logging_error:
            logger.exception(
                "Could not persist failed agent run",
                extra={
                    "run_id": run_id,
                    "error_type": type(logging_error).__name__,
                },
            )
            print(f"Warning: could not save run failure: {logging_error}")

        print(f"\nAgent error: {exc}")
        print(
            "Check your terminal logs, API configuration, database "
            "connection, and tool definitions."
        )


def main() -> None:
    """Initialize the CLI coding agent."""

    load_configuration()

    from database.checkpoints import postgres_checkpointer
    from database.connection import (
        check_database_connection,
        create_tables,
        engine,
    )
    from database.repository import AgentRepository
    from filesystem_tools import FilesystemTools
    from agent.graph import create_graph

    workspace_path = os.getenv("WORKSPACE_PATH", "workspace")

    os.makedirs(workspace_path, exist_ok=True)

    check_database_connection()
    create_tables()

    model = create_model()
    filesystem = FilesystemTools(workspace_path)

    repository = AgentRepository()

    conversation = repository.create_conversation(
        title="Filesystem Coding Agent CLI"
    )
    conversation_id = object_id(conversation)

    print("=" * 65)
    print("FILESYSTEM-AWARE CODING AGENT")
    print("=" * 65)
    print(f"Workspace: {os.path.abspath(workspace_path)}")
    print("Type 'exit' or 'quit' to stop.")
    print("File modifications require your approval.")
    print()

    try:
        with postgres_checkpointer() as checkpointer:
            graph = create_graph(
                model=model,
                filesystem=filesystem,
                checkpointer=checkpointer,
            )

            while True:
                try:
                    request = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting agent.")
                    break

                if request.lower() in {"exit", "quit"}:
                    print("Exiting agent.")
                    break

                if not request:
                    continue

                run_task(
                    graph=graph,
                    repository=repository,
                    conversation_id=conversation_id,
                    request=request,
                )

                print()

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
