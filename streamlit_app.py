
from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Any

st = import_module("streamlit")
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from agent.workflow import (
    finish_run as workflow_finish_run,
    record_successful_file_changes as workflow_record_file_changes,
    record_tool_messages as workflow_record_tool_messages,
    save_final_response,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Filesystem Coding Agent",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CONFIGURATION AND RESOURCE INITIALIZATION
# ---------------------------------------------------------

def load_configuration() -> None:
    """Load and validate the required environment configuration."""

    load_dotenv()

    required = [
        "GEMINI_MODEL",
        "DATABASE_URL",
        "LANGGRAPH_CHECKPOINT_URL",
    ]

    for variable in required:
        if not os.getenv(variable):
            raise RuntimeError(
                f"Missing {variable}. Configure it in your .env file."
            )

    if not (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    ):
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )


def make_model() -> Any:
    """Initialize Gemini through LangChain."""

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


def get_object_id(value: Any) -> str:
    """Extract an ID from a repository result."""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("id", "conversation_id", "run_id"):
            if value.get(key) is not None:
                return str(value[key])

    for key in ("id", "conversation_id", "run_id"):
        if hasattr(value, key):
            return str(getattr(value, key))

    raise ValueError(f"Cannot extract an ID from {value!r}")


def get_field(value: Any, field: str, default: Any = None) -> Any:
    """Read a field from an ORM object or dictionary."""

    if isinstance(value, dict):
        return value.get(field, default)

    return getattr(value, field, default)


@st.cache_resource
def initialize_resources() -> dict[str, Any]:
    """
    Initialize shared resources once per Streamlit server process.

    The PostgreSQL checkpointer stays open for the lifetime of the
    cached resources. Database sessions should still be opened and
    closed by AgentRepository methods.
    """

    load_configuration()

    from agent.graph import create_graph
    from database.checkpoints import postgres_checkpointer
    from database.connection import (
        check_database_connection,
        create_tables,
        engine,
    )
    from database.repository import AgentRepository
    from filesystem_tools import FilesystemTools

    workspace = Path(
        os.getenv("WORKSPACE_PATH", "workspace")
    ).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    check_database_connection()
    create_tables()

    model = make_model()
    filesystem = FilesystemTools(str(workspace))
    repository = AgentRepository()

    checkpoint_context: AbstractContextManager[Any] = (
        postgres_checkpointer()
    )
    checkpointer = checkpoint_context.__enter__()

    try:
        graph = create_graph(
            model=model,
            filesystem=filesystem,
            checkpointer=checkpointer,
        )
    except Exception:
        checkpoint_context.__exit__(*__import__("sys").exc_info())
        engine.dispose()
        raise

    return {
        "graph": graph,
        "repository": repository,
        "filesystem": filesystem,
        "workspace": workspace,
        "checkpoint_context": checkpoint_context,
        "engine": engine,
    }


# ---------------------------------------------------------
# MESSAGE AND INTERRUPT HELPERS
# ---------------------------------------------------------


def message_text(content: Any) -> str:
    """Render text content without provider metadata or raw objects."""

    if isinstance(content, str):
        return content.strip()

    if content is None:
        return ""

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    value = item.get("text")
                    if isinstance(value, str):
                        text_parts.append(value)
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])

        return "\n".join(
            part.strip() for part in text_parts if part.strip()
        )

    if isinstance(content, dict):
        value = content.get("text")
        if isinstance(value, str):
            return value.strip()

    return str(content)


def get_interrupts(result: dict[str, Any]) -> list[Any]:
    """Extract pending interrupts from a LangGraph result."""

    value = result.get("__interrupt__", [])

    if not value:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def get_interrupt_payload(item: Any) -> dict[str, Any]:
    """Extract the structured approval payload."""

    value = getattr(item, "value", None)

    if value is None and isinstance(item, dict):
        value = item.get("value", item)

    if not isinstance(value, dict):
        raise ValueError("The agent returned an invalid approval payload.")

    return value


def make_approval_id(run_id: str, payload: dict[str, Any]) -> str:
    """Create a stable approval ID for this run and proposed change."""

    return str(
        uuid.uuid5(
            uuid.UUID(run_id),
            "|".join(
                [
                    str(payload.get("operation", "")),
                    str(payload.get("relative_path", "")),
                    str(payload.get("proposed_sha256", "")),
                ]
            ),
        )
    )


def record_approval(
    repository: Any,
    run_id: str,
    payload: dict[str, Any],
) -> str:
    """Persist the approval request in PostgreSQL."""

    approval_id = make_approval_id(run_id, payload)

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


# ---------------------------------------------------------
# AUDIT LOGGING
# ---------------------------------------------------------

def record_tool_messages(
    repository: Any,
    run_id: str,
    messages: list[Any],
) -> None:
    """Persist tool results for auditing."""

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        repository.record_tool(
            run_id=run_id,
            tool_name=getattr(message, "name", None) or "unknown",
            status="completed",
            arguments={
                "tool_call_id": getattr(message, "tool_call_id", None)
            },
            result_summary=message_text(message.content)[:4000],
        )


def record_successful_file_changes(
    repository: Any,
    run_id: str,
    messages: list[Any],
) -> None:
    """
    Record successful write/edit results if the tools return structured JSON.

    The expected result fields are status, relative_path, operation,
    original_sha256, proposed_sha256,     and optionally backup_relative_path.
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

        path = result.get("relative_path")
        operation = result.get("operation")

        if not path or not operation:
            continue

        repository.record_file_change(
            run_id=run_id,
            relative_path=path,
            operation=operation,
            original_sha256=result.get("original_sha256"),
            proposed_sha256=result.get("proposed_sha256"),
            backup_relative_path=result.get("backup_relative_path"),
            status="completed",
        )


# ---------------------------------------------------------
# RUN MANAGEMENT
# ---------------------------------------------------------

def create_run(
    repository: Any,
    conversation_id: str,
    request: str,
) -> dict[str, str]:
    """Create a database run and its unique checkpoint thread."""

    thread_id = str(uuid.uuid4())

    run_record = repository.create_run(
        conversation_id=conversation_id,
        thread_id=thread_id,
        task=request,
    )
    run_id = get_object_id(run_record)

    repository.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request,
    )

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "conversation_id": conversation_id,
    }


def complete_run(
    repository: Any,
    run_id: str,
    conversation_id: str,
    result: dict[str, Any],
) -> str:
    """Save the final response and mark the run as completed."""

    messages = result.get("messages", [])

    workflow_record_tool_messages(repository, run_id, messages)
    workflow_record_file_changes(repository, run_id, messages)
    response = save_final_response(repository, conversation_id, messages)
    workflow_finish_run(
        repository,
        run_id,
        "completed",
        response=response,
    )

    return response


def fail_run(
    repository: Any,
    run_id: str,
    error: Exception,
) -> None:
    """Record a failed run without hiding the original error."""

    try:
        workflow_finish_run(
            repository,
            run_id,
            "failed",
            error=str(error)[:4000],
        )
    except Exception as logging_error:
        logger.exception(
            "Could not persist failed agent run",
            extra={"run_id": run_id, "error_type": type(logging_error).__name__},
        )


def submit_request(
    resources: dict[str, Any],
    conversation_id: str,
    request: str,
) -> None:
    """Start a request and either finish it or display an approval."""

    repository = resources["repository"]
    graph = resources["graph"]

    run_info = create_run(repository, conversation_id, request)

    config = {
        "configurable": {
            "thread_id": run_info["thread_id"],
        },
        "recursion_limit": 50,
    }

    try:
        with st.spinner("The agent is inspecting your workspace..."):
            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=request)],
                    "tool_rounds": 0,
                },
                config=config,
            )

        process_result(
            resources=resources,
            run_info=run_info,
            result=result,
            config=config,
        )

    except Exception as error:
        fail_run(repository, run_info["run_id"], error)
        st.session_state["last_error"] = (
            f"Request failed: {type(error).__name__}: {error}"
        )
        st.error(f"The agent could not complete this request: {error}")


def process_result(
    resources: dict[str, Any],
    run_info: dict[str, str],
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """
    Save a pending approval or finalize the run.

    A pending approval is stored in Streamlit session state so that the
    approval buttons remain available after a normal UI rerun.
    """

    repository = resources["repository"]
    run_id = run_info["run_id"]
    conversation_id = run_info["conversation_id"]

    messages = result.get("messages", [])

    record_tool_messages(repository, run_id, messages)
    record_successful_file_changes(repository, run_id, messages)

    interrupts = get_interrupts(result)

    if interrupts:
        if len(interrupts) != 1:
            raise RuntimeError(
                "This frontend supports one pending approval at a time. "
                "Parallel approval interrupts are not supported."
            )

        payload = get_interrupt_payload(interrupts[0])
        approval_id = record_approval(repository, run_id, payload)

        workflow_finish_run(
            repository,
            run_id,
            "waiting_approval",
            response=None,
            error=None,
        )

        st.session_state["pending_approval"] = {
            **run_info,
            "payload": payload,
            "approval_id": approval_id,
            "config": config,
        }

        st.rerun()

    response = complete_run(
        repository,
        run_id,
        conversation_id,
        result,
    )

    st.session_state["pending_approval"] = None
    st.session_state["last_response"] = response
    st.rerun()


def resume_approval(
    resources: dict[str, Any],
    approved: bool,
) -> None:
    """Resume a paused graph after the user approves or rejects a change."""

    pending = st.session_state.get("pending_approval")

    if not pending:
        st.warning("There is no pending approval.")
        return

    repository = resources["repository"]
    graph = resources["graph"]

    decision = "approved" if approved else "rejected"

    try:
        repository.decide_approval(
            approval_id=pending["approval_id"],
            status=decision,
        )

        payload = pending["payload"]
        resume_value = {
            **payload,
            "decision": "approve" if approved else "reject",
        }

        with st.spinner("Resuming the agent..."):
            result = graph.invoke(
                Command(resume=resume_value),
                config=pending["config"],
            )

        # Remove the old pending record from the UI before processing
        # the result. process_result may install a new pending approval.
        st.session_state["pending_approval"] = None

        process_result(
            resources=resources,
            run_info={
                "run_id": pending["run_id"],
                "thread_id": pending["thread_id"],
                "conversation_id": pending["conversation_id"],
            },
            result=result,
            config=pending["config"],
        )

    except Exception as error:
        fail_run(repository, pending["run_id"], error)
        st.session_state["pending_approval"] = None
        st.error(f"Could not resume the agent: {error}")


# ---------------------------------------------------------
# CONVERSATION HELPERS
# ---------------------------------------------------------

def list_conversations(repository: Any) -> list[Any]:
    """Get saved conversations, returning an empty list on failure."""

    try:
        return list(repository.list_conversations())
    except Exception as error:
        st.sidebar.error(f"Could not load conversations: {error}")
        return []


def new_conversation(repository: Any) -> str:
    """Create and return a new conversation ID."""

    conversation = repository.create_conversation(
        title="New coding conversation"
    )

    return get_object_id(conversation)


def load_history(repository: Any, conversation_id: str) -> list[Any]:
    """Load saved messages for a conversation."""

    try:
        return list(
            repository.get_conversation_history(conversation_id)
        )
    except Exception as error:
        st.error(f"Could not load conversation history: {error}")
        return []


# ---------------------------------------------------------
# WORKSPACE FILE VIEWER
# ---------------------------------------------------------

def get_workspace_files(workspace: Path) -> list[str]:
    """
    List regular files under the configured workspace.

    Symlinks and common generated directories are excluded. This is
    a display-only listing; the filesystem tools still enforce access
    rules for actual file operations.
    """

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".agent_backups",
        "node_modules",
    }

    files: list[str] = []

    try:
        for path in workspace.rglob("*"):
            try:
                if path.is_symlink() or not path.is_file():
                    continue

                relative = path.relative_to(workspace)

                if any(part in ignored_directories for part in relative.parts):
                    continue

                files.append(relative.as_posix())

                if len(files) >= 200:
                    break

            except (OSError, ValueError):
                continue

    except OSError:
        return []

    return sorted(files)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def render_sidebar(resources: dict[str, Any]) -> str:
    """Render the conversation list and workspace information."""

    repository = resources["repository"]
    workspace = resources["workspace"]

    st.sidebar.title("🧑‍💻 Coding Agent")
    st.sidebar.caption("Gemini · LangGraph · PostgreSQL")

    if st.sidebar.button(
        "＋ New conversation",
        use_container_width=True,
    ):
        conversation_id = new_conversation(repository)
        st.session_state["conversation_id"] = conversation_id
        st.session_state["pending_approval"] = None
        st.session_state["last_response"] = None
        st.rerun()

    conversations = list_conversations(repository)

    conversation_options: dict[str, str] = {}

    for conversation in conversations:
        conversation_id = str(get_field(conversation, "id", ""))
        title = str(
            get_field(conversation, "title", "Untitled conversation")
        )

        if conversation_id:
            conversation_options[conversation_id] = title

    active_id = st.session_state.get("conversation_id")

    if active_id not in conversation_options:
        if conversation_options:
            active_id = next(iter(conversation_options))
        else:
            active_id = new_conversation(repository)
            conversation_options[active_id] = "New coding conversation"

        st.session_state["conversation_id"] = active_id

    ids = list(conversation_options.keys())
    current_index = ids.index(active_id) if active_id in ids else 0

    selected_id = st.sidebar.selectbox(
        "Conversations",
        options=ids,
        index=current_index,
        format_func=lambda item: conversation_options.get(
            item, "Untitled conversation"
        ),
        key="conversation_selector",
    )

    if selected_id != st.session_state.get("conversation_id"):
        st.session_state["conversation_id"] = selected_id
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Workspace")

    st.sidebar.code(str(workspace), language=None)

    with st.sidebar.expander("Browse workspace files"):
        files = get_workspace_files(workspace)

        if not files:
            st.caption("No files found yet.")

        for file_name in files:
            st.text(f"📄 {file_name}")

        if len(files) >= 200:
            st.caption("Showing at most 200 files.")

    st.sidebar.divider()
    st.sidebar.caption("File writes require explicit approval.")

    return active_id


# ---------------------------------------------------------
# MAIN CHAT UI
# ---------------------------------------------------------

def render_chat(resources: dict[str, Any], conversation_id: str) -> None:
    """Render history, approval cards, and the chat input."""

    repository = resources["repository"]

    st.title("Filesystem-Aware Coding Agent")
    st.caption(
        "Ask the agent to inspect, explain, search, or propose changes "
        "to files in your configured workspace."
    )

    history = load_history(repository, conversation_id)
    history_contents = {
        str(get_field(message, "content", ""))
        for message in history
        if str(get_field(message, "role", "")).lower() == "user"
    }

    for message in history:
        role = str(get_field(message, "role", "assistant")).lower()
        content = get_field(message, "content", "")

        if role not in {"user", "assistant", "system"}:
            role = "assistant"

        with st.chat_message(
            "user" if role == "user" else "assistant"
        ):
            st.markdown(message_text(content))

    active_request = st.session_state.get("active_request")
    if active_request in history_contents:
        st.session_state["active_request"] = None
        active_request = None

    if active_request and active_request not in history_contents:
        with st.chat_message("user"):
            st.markdown(active_request)

    last_error = st.session_state.pop("last_error", None)
    if last_error:
        st.error(last_error)

    pending = st.session_state.get("pending_approval")

    if not pending:
        try:
            recovered = repository.list_pending_approvals(
                conversation_id=conversation_id
            )
        except Exception as error:
            logger.exception(
                "Could not recover pending approvals",
                extra={"conversation_id": conversation_id},
            )
            recovered = []
            st.warning(
                "Pending approval recovery is unavailable. "
                f"Database error: {error}"
            )

        if recovered:
            approval = recovered[0]
            pending = {
                "run_id": approval["run_id"],
                "thread_id": approval["thread_id"],
                "conversation_id": approval["conversation_id"],
                "approval_id": approval["id"],
                "payload": {
                    key: approval[key]
                    for key in (
                        "relative_path",
                        "operation",
                        "original_sha256",
                        "proposed_sha256",
                        "diff",
                    )
                },
                "config": {
                    "configurable": {
                        "thread_id": approval["thread_id"]
                    },
                    "recursion_limit": 50,
                },
            }
            st.session_state["pending_approval"] = pending

    if pending and pending.get("conversation_id") == conversation_id:
        payload = pending["payload"]

        st.divider()
        st.subheader("🛡️ File change requires your approval")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Operation**")
            st.code(str(payload.get("operation", "unknown")))

        with col2:
            st.write("**File**")
            st.code(str(payload.get("relative_path", "unknown")))

        st.warning(
            "Review the proposed change carefully. Approving it allows "
            "the agent's tool to attempt the modification."
        )

        diff = payload.get("diff")

        if diff:
            st.markdown("**Proposed diff**")
            st.code(str(diff), language="diff")
        else:
            st.info("The tool did not provide a diff.")

        approve_col, reject_col = st.columns(2)

        with approve_col:
            if st.button(
                "Approve change",
                type="primary",
                use_container_width=True,
                key=f"approve_{pending['approval_id']}",
            ):
                resume_approval(resources, approved=True)

        with reject_col:
            if st.button(
                "Reject change",
                use_container_width=True,
                key=f"reject_{pending['approval_id']}",
            ):
                resume_approval(resources, approved=False)

    elif pending:
        st.info(
            "Another conversation has a pending file approval. "
            "Switch to that conversation to review it."
        )

    request = st.chat_input(
        "Ask the agent to inspect or improve your code...",
        disabled=bool(pending),
    )

    if request:
        if pending:
            st.warning(
                "Resolve the pending approval before submitting another task."
            )
            return

        st.session_state["active_request"] = request

        submit_request(
            resources=resources,
            conversation_id=conversation_id,
            request=request,
        )


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

def main() -> None:
    """Run the Streamlit application."""

    # Render the shell before connecting to PostgreSQL, opening the
    # checkpoint pool, or initializing the model. Those operations can take
    # several seconds and must not leave the browser with a blank page.
    st.title("Filesystem-Aware Coding Agent")
    st.caption(
        "Ask the agent to inspect, explain, search, or propose changes "
        "to files in your configured workspace."
    )

    resources: dict[str, Any] | None = None
    try:
        with st.status(
            "Connecting to the agent services...",
            expanded=True,
        ) as status:
            status.write("Loading configuration")
            status.write("Connecting to the database")
            status.write("Opening the checkpoint store")
            status.write("Initializing the language model")
            resources = initialize_resources()
            status.update(
                label="Agent ready",
                state="complete",
                expanded=False,
            )
    except Exception as error:
        st.error(
            "The application could not initialize. "
            f"{type(error).__name__}: {error}"
        )
        st.exception(error)
        st.info(
            "Check your .env settings, PostgreSQL service, database "
            "tables, Gemini API key, and installed dependencies."
        )
        st.stop()

    if resources is None:
        return

    try:
        conversation_id = render_sidebar(resources)
        render_chat(resources, conversation_id)
    except Exception as error:
        logger.exception("Streamlit rendering failed")
        st.error(
            "The chat interface encountered an error while processing "
            f"the request: {type(error).__name__}: {error}"
        )
        st.info(
            "Your request and any pending approval remain in the database. "
            "Refresh the page after resolving the reported issue."
        )


if __name__ == "__main__":
    main()
