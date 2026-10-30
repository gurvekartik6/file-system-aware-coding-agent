"""Read-only and approval-gated filesystem tools."""

import difflib
import hashlib
import json
from datetime import datetime, timezone

from langchain_core.tools import tool
from langgraph.types import interrupt

from filesystem_tools import FilesystemTools


MAX_TOOL_OUTPUT_CHARS = 12_000


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _format_result(data) -> str:
    result = json.dumps(data, ensure_ascii=False, indent=2)

    if len(result) > MAX_TOOL_OUTPUT_CHARS:
        return (
            result[:MAX_TOOL_OUTPUT_CHARS]
            + "\n...[output truncated]"
        )

    return result


def _make_diff(old: str, new: str, filename: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


def _apply_approved_change(
    filesystem: FilesystemTools,
    relative_path: str,
    new_content: str,
    operation: str,
    overwrite: bool,
) -> str:
    """Pause for approval, revalidate the proposal, then apply it."""

    if not isinstance(new_content, str):
        raise TypeError("File content must be a string.")

    if len(new_content.encode("utf-8")) > filesystem.MAX_FILE_BYTES:
        raise ValueError("Proposed content exceeds the file-size limit.")

    # Resolve the target using the existing workspace validator.
    target = filesystem._resolve_path(relative_path)

    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"Parent directory does not exist: {target.parent}"
        )

    if target.exists() and not target.is_file():
        raise IsADirectoryError(relative_path)

    exists = target.exists()

    if exists and not overwrite:
        raise FileExistsError(
            f"{relative_path} already exists. "
            "Set overwrite=True to propose a replacement."
        )

    if exists:
        old_content = filesystem.read_file(relative_path)
    else:
        old_content = ""

    original_hash = _sha256(old_content) if exists else None
    proposed_hash = _sha256(new_content)

    if original_hash == proposed_hash and exists:
        return "No changes needed: the file already has this content."

    diff = _make_diff(old_content, new_content, relative_path)

    # This call pauses graph execution. No write has happened yet.
    decision = interrupt({
        "type": "file_change_approval",
        "operation": operation,
        "relative_path": relative_path,
        "original_exists": exists,
        "original_sha256": original_hash,
        "proposed_sha256": proposed_hash,
        "diff": diff[:MAX_TOOL_OUTPUT_CHARS],
        "message": (
            "Approve this exact file change? "
            "Reply with approve or reject."
        ),
    })

    if not isinstance(decision, dict):
        return "Change rejected: invalid approval response."

    if decision.get("decision") != "approve":
        return f"Change rejected for {relative_path}. No file was modified."

    # Bind the decision to the exact path, operation and proposed content.
    if decision.get("relative_path") != relative_path:
        return "Change rejected: approved path does not match."

    if decision.get("operation") != operation:
        return "Change rejected: approved operation does not match."

    if decision.get("proposed_sha256") != proposed_hash:
        return "Change rejected: proposed content does not match approval."

    # Resolve and inspect the target again after resuming.
    target = filesystem._resolve_path(relative_path)
    now_exists = target.exists()

    if now_exists != exists:
        return (
            f"Change rejected: file existence changed while awaiting "
            f"approval: {relative_path}"
        )

    if now_exists:
        current_content = filesystem.read_file(relative_path)

        if _sha256(current_content) != original_hash:
            return (
                f"Change rejected: {relative_path} changed after the "
                "proposal was generated. Generate a new proposal."
            )

    if decision.get("original_sha256") != original_hash:
        return "Change rejected: original file version does not match."

    # Save a backup before replacing existing content.
    backup_relative_path = None
    current_content = old_content

    if now_exists:
        backup_dir = filesystem.root / ".agent_backups"

        if backup_dir.is_symlink():
            raise ValueError("Backup directory cannot be a symbolic link.")

        if backup_dir.exists() and not backup_dir.is_dir():
            raise NotADirectoryError(
                "Backup path exists but is not a directory."
            )

        backup_dir.mkdir(exist_ok=True)

        # Use a unique filename and exclusive creation.
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        backup_name = f"{timestamp}-{proposed_hash[:12]}.bak"
        backup_path = backup_dir / backup_name

        with backup_path.open("x", encoding="utf-8", newline="") as file:
            file.write(current_content)

        backup_relative_path = backup_path.relative_to(
            filesystem.root
        ).as_posix()

    # Revalidate immediately before writing.
    final_target = filesystem._resolve_path(relative_path)

    if final_target.exists() != exists:
        return (
            "Change rejected: target changed before the write. "
            "Generate a new proposal."
        )

    if final_target.exists():
        final_content = filesystem.read_file(relative_path)

        if _sha256(final_content) != original_hash:
            return (
                "Change rejected: target content changed before the write."
            )

    filesystem.write_file(
        relative_path,
        new_content,
        overwrite=exists,
    )

    return _format_result({
        "status": "success",
        "operation": operation,
        "relative_path": relative_path,
        "original_sha256": original_hash,
        "proposed_sha256": proposed_hash,
        "backup_relative_path": backup_relative_path,
        "message": "Approved change applied successfully.",
    })


def create_agent_tools(
    filesystem: FilesystemTools,
) -> list:
    """Register read-only and approval-gated tools."""

    @tool
    def list_files(directory: str = ".") -> str:
        """List files and directories inside the workspace.

        Args:
            directory: Relative directory inside the workspace.
        """
        return _format_result(filesystem.list_files(directory))

    @tool
    def read_file(relative_path: str) -> str:
        """Read a UTF-8 text file inside the workspace.

        Args:
            relative_path: Relative path of the file to read.
        """
        content = filesystem.read_file(relative_path)

        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            return (
                content[:MAX_TOOL_OUTPUT_CHARS]
                + "\n...[file content truncated]"
            )

        return content

    @tool
    def search_files(
        query: str,
        directory: str = ".",
        use_regex: bool = False,
    ) -> str:
        """Search workspace files for matching text.

        Args:
            query: Text or regular expression to search for.
            directory: Relative directory to search.
            use_regex: True to interpret query as a regular expression.
        """
        results = filesystem.search_files(
            query=query,
            directory=directory,
            use_regex=use_regex,
        )
        return _format_result(results)

    @tool
    def write_file(
        relative_path: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """Propose creating or replacing a workspace text file.

        Existing files require overwrite=True. The user must approve
        the exact proposed change before it is applied.

        Args:
            relative_path: Relative path of the target file.
            content: Complete proposed file content.
            overwrite: Whether an existing file may be replaced.
        """
        return _apply_approved_change(
            filesystem=filesystem,
            relative_path=relative_path,
            new_content=content,
            operation="write",
            overwrite=overwrite,
        )

    @tool
    def edit_file(
        relative_path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """Propose replacing one unique text section in a file.

        The user must approve the resulting change before it is applied.

        Args:
            relative_path: Relative path of the existing file.
            old_text: Exact text to replace; must occur once.
            new_text: Replacement text.
        """
        current = filesystem.read_file(relative_path)
        matches = current.count(old_text)

        if not old_text:
            raise ValueError("old_text cannot be empty.")

        if matches != 1:
            raise ValueError(
                f"Expected one match, found {matches}. "
                "Provide a more specific text section."
            )

        updated = current.replace(old_text, new_text, 1)

        return _apply_approved_change(
            filesystem=filesystem,
            relative_path=relative_path,
            new_content=updated,
            operation="edit",
            overwrite=True,
        )

    return [
        list_files,
        read_file,
        search_files,
        write_file,
        edit_file,
    ]
