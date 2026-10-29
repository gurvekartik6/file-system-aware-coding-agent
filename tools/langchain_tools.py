"""LangChain wrappers around our filesystem toolkit."""

import json

from langchain_core.tools import tool

from filesystem_tools import FilesystemTools


MAX_TOOL_OUTPUT_CHARS = 12_000


def format_result(data) -> str:
    """Convert a tool result into a bounded text response."""
    result = json.dumps(data, ensure_ascii=False, indent=2)

    if len(result) > MAX_TOOL_OUTPUT_CHARS:
        result = (
            result[:MAX_TOOL_OUTPUT_CHARS]
            + "\n...[output truncated by application]"
        )

    return result


def create_filesystem_tools(
    filesystem: FilesystemTools,
) -> list:
    """Create the read-only tools available to the LLM."""

    @tool
    def list_files(directory: str = ".") -> str:
        """List files and directories in the workspace.

        Args:
            directory: A relative directory inside the workspace.
        """
        return format_result(filesystem.list_files(directory))

    @tool
    def read_file(relative_path: str) -> str:
        """Read a UTF-8 text file inside the workspace.

        Args:
            relative_path: The relative path of the file to read.
        """
        content = filesystem.read_file(relative_path)

        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = (
                content[:MAX_TOOL_OUTPUT_CHARS]
                + "\n...[file content truncated by application]"
            )

        return content

    @tool
    def search_files(
        query: str,
        directory: str = ".",
        use_regex: bool = False,
    ) -> str:
        """Search workspace text files for a query.

        Args:
            query: Text to find, or a regex if use_regex is true.
            directory: Relative directory to search.
            use_regex: Whether query should be interpreted as regex.
        """
        results = filesystem.search_files(
            query=query,
            directory=directory,
            use_regex=use_regex,
        )

        return format_result(results)

    return [list_files, read_file, search_files]
