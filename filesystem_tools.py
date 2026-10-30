import os
import tempfile
from pathlib import Path
import re


class FilesystemTools:
    """Safely manage text files inside one configured workspace."""

    MAX_FILE_BYTES = 1_000_000
    MAX_SEARCH_FILES = 5_000

    def __init__(self, workspace: str | Path):
        root = Path(workspace).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise NotADirectoryError(
                "The workspace must be an existing directory."
            )

        self.root = root

    def _resolve_path(self, relative_path: str) -> Path:
        """Validate a relative path and keep it inside the workspace."""

        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("A non-empty relative path is required.")

        if "\x00" in relative_path:
            raise ValueError("Null characters are not allowed in paths.")

        # Accept Windows separators even when running on another OS.
        normalized = relative_path.replace("\\", "/")
        path = Path(normalized)

        if path.is_absolute() or path.drive or normalized.startswith("/"):
            raise ValueError("Absolute paths are not allowed.")

        if ".." in path.parts:
            raise ValueError("Parent-directory traversal is not allowed.")

        candidate = self.root / path

        # Reject symlinks in any existing component.
        current = self.root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symbolic links are not allowed.")

        resolved = candidate.resolve(strict=False)

        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes the workspace.")

        return resolved

    def _check_file_size(self, path: Path) -> None:
        if path.stat().st_size > self.MAX_FILE_BYTES:
            raise ValueError("File exceeds the 1 MB size limit.")

    def list_files(self, directory: str = ".") -> list[dict]:
        """List the immediate contents of a workspace directory."""

        folder = self._resolve_path(directory)

        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results = []

        for item in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            # Show symlinks as entries, but never follow them.
            if item.is_symlink():
                results.append({
                    "path": item.relative_to(self.root).as_posix(),
                    "type": "symlink",
                    "size_bytes": None,
                })
                continue

            results.append({
                "path": item.relative_to(self.root).as_posix(),
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })

        return results

    def read_file(self, relative_path: str) -> str:
        """Read a UTF-8 text file."""

        path = self._resolve_path(relative_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {relative_path}")

        self._check_file_size(path)
        return path.read_text(encoding="utf-8")

    def write_file(
        self,
        relative_path: str,
        content: str,
        overwrite: bool = False,
    ) -> str:
        """Create a file, or replace it when overwrite=True."""

        if not isinstance(content, str):
            raise TypeError("Content must be a string.")

        if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
            raise ValueError("New content exceeds the 1 MB size limit.")

        path = self._resolve_path(relative_path)

        if path.exists():
            if not path.is_file():
                raise IsADirectoryError(f"Not a file: {relative_path}")

            if not overwrite:
                raise FileExistsError(
                    f"{relative_path} already exists. "
                    "Use overwrite=True to replace it."
                )

            self._check_file_size(path)

        # Do not create arbitrary directory trees implicitly.
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"Parent directory does not exist: {path.parent}"
            )

        # Write through a sibling temporary file and replace atomically. The
        # final replacement does not follow a symlink at the target path.
        encoded = content.encode("utf-8")
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
                temp_path = Path(file.name)

            if not overwrite and path.exists():
                raise FileExistsError(
                    f"{relative_path} already exists. "
                    "Use overwrite=True to replace it."
                )

            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        return f"Successfully wrote {relative_path}"

    def edit_file(
        self,
        relative_path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """Replace one exact text match in a file."""

        if not old_text:
            raise ValueError("The text to replace cannot be empty.")

        content = self.read_file(relative_path)
        matches = content.count(old_text)

        if matches == 0:
            raise ValueError("The specified text was not found.")

        if matches > 1:
            raise ValueError(
                f"Found {matches} matches. "
                "Provide a more specific text section."
            )

        updated = content.replace(old_text, new_text, 1)

        return self.write_file(
            relative_path,
            updated,
            overwrite=True,
        )

    def search_files(
        self,
        query: str,
        directory: str = ".",
        use_regex: bool = False,
    ) -> list[dict]:
        """Search text files and return matching lines."""

        if not isinstance(query, str) or not query:
            raise ValueError("Search query cannot be empty.")

        folder = self._resolve_path(directory)

        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = re.compile(query) if use_regex else None
        results = []
        files_checked = 0

        def walk_error(error):
            raise error

        for current_dir, dirs, filenames in __import__("os").walk(
            folder,
            followlinks=False,
            onerror=walk_error,
        ):
            current = Path(current_dir)

            # Do not descend into symlinked directories.
            dirs[:] = [
                name for name in dirs
                if not (current / name).is_symlink()
            ]

            for filename in filenames:
                path = current / filename

                if path.is_symlink() or not path.is_file():
                    continue

                files_checked += 1
                if files_checked > self.MAX_SEARCH_FILES:
                    raise ValueError("Search exceeded the file-count limit.")

                try:
                    self._check_file_size(path)
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # Binary or non-UTF-8 file: skip it.
                    continue

                for line_number, line in enumerate(content.splitlines(), 1):
                    matched = (
                        bool(pattern.search(line))
                        if pattern is not None
                        else query in line
                    )

                    if matched:
                        results.append({
                            "path": path.relative_to(self.root).as_posix(),
                            "line": line_number,
                            "text": line,
                        })

        return results
