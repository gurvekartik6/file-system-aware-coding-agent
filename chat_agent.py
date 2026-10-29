"""CLI for the LangGraph coding agent with approval-gated edits."""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from filesystem_tools import FilesystemTools
from agent.graph import create_graph


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT / "workspace"


def create_model():
    """Create the configured Gemini chat model."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Configure it in .env."
        )

    model_name = os.getenv("GEMINI_MODEL")

    if not model_name:
        raise RuntimeError(
            "GEMINI_MODEL is missing from .env. "
            "Set it to a model available to your Gemini API account."
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def get_interrupts(result):
    """Extract pending interrupt payloads from a graph result."""
    interrupts = result.get("__interrupt__", ())

    return [
        item.value
        for item in interrupts
        if isinstance(item.value, dict)
    ]


def review_change(payload: dict) -> dict:
    """Display the proposal and obtain a human decision."""

    print("\n" + "=" * 65)
    print("FILE CHANGE REQUIRES APPROVAL")
    print("=" * 65)
    print(f"Operation: {payload.get('operation')}")
    print(f"File:      {payload.get('relative_path')}")
    print(f"Exists:    {payload.get('original_exists')}")
    print(f"Old hash:  {payload.get('original_sha256')}")
    print(f"New hash:  {payload.get('proposed_sha256')}")
    print("\nProposed diff:\n")

    print(payload.get("diff") or "(No textual diff available)")

    print("\n1. Approve and apply")
    print("2. Reject the change")

    while True:
        choice = input("Your decision (1/2): ").strip()

        if choice == "1":
            decision = "approve"
            break

        if choice == "2":
            decision = "reject"
            break

        print("Enter 1 or 2.")

    # Preserve the proposal metadata so the tool can validate that
    # the resumed approval refers to the exact original proposal.
    return {
        **payload,
        "decision": decision,
    }


def get_final_answer(result) -> str:
    """Get the most recent assistant response."""
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            if not getattr(message, "tool_calls", None):
                return str(message.content)

    return "The agent stopped without producing a final response."


def run_task(graph, user_request: str) -> str:
    """Run one task, handling any approval interrupts."""

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_request)],
            "tool_rounds": 0,
        },
        config=config,
    )

    # A graph can pause more than once during a single task.
    while result.get("__interrupt__"):
        pending = get_interrupts(result)

        if not pending:
            return (
                "The graph paused, but the approval payload could not "
                "be interpreted. No further changes were requested."
            )

        if len(pending) > 1:
            return (
                "Multiple simultaneous approvals were returned. "
                "This CLI supports reviewing one proposal at a time; "
                "no approval was automatically granted."
            )

        decision = review_change(pending[0])

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

    return get_final_answer(result)


def main():
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    try:
        filesystem = FilesystemTools(WORKSPACE)
        model = create_model()
        graph = create_graph(model, filesystem)
    except Exception as exc:
        print(f"Startup error: {type(exc).__name__}: {exc}")
        return

    print("\nFilesystem-Aware Coding Agent — Part 3")
    print("Workspace:", WORKSPACE)
    print("Mode: write approval required")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            request = input("You: ").strip()

            if request.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not request:
                continue

            answer = run_task(graph, request)
            print(f"\nAssistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(
                f"\nTask failed: {type(exc).__name__}: {exc}\n"
                "Inspect the error before retrying."
            )


if __name__ == "__main__":
    main()
