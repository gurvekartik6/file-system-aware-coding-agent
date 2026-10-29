
"""State definitions for the coding agent."""

from langgraph.graph import MessagesState


class CodingAgentState(MessagesState):
    """Shared state for the coding-agent workflow."""

    tool_rounds: int
