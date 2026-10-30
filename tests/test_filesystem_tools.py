import json

import pytest

from filesystem_tools import FilesystemTools
from tools import agent_tools


def test_paths_cannot_escape_workspace(tmp_path):
    filesystem = FilesystemTools(tmp_path)

    with pytest.raises(ValueError):
        filesystem.read_file("../outside.txt")


def test_overwrite_is_atomic_and_preserves_new_content(tmp_path):
    filesystem = FilesystemTools(tmp_path)
    filesystem.write_file("file.txt", "before")
    filesystem.write_file("file.txt", "after", overwrite=True)

    assert filesystem.read_file("file.txt") == "after"
    assert not list(tmp_path.glob("*.tmp"))


def test_approved_change_returns_hashes_and_real_backup(
    tmp_path,
    monkeypatch,
):
    filesystem = FilesystemTools(tmp_path)
    filesystem.write_file("file.txt", "before")

    def approve(payload):
        return {
            "decision": "approve",
            "relative_path": payload["relative_path"],
            "operation": payload["operation"],
            "original_sha256": payload["original_sha256"],
            "proposed_sha256": payload["proposed_sha256"],
        }

    monkeypatch.setattr(agent_tools, "interrupt", approve)
    result = json.loads(
        agent_tools._apply_approved_change(
            filesystem,
            "file.txt",
            "after",
            "edit",
            True,
        )
    )

    assert filesystem.read_file("file.txt") == "after"
    assert result["original_sha256"]
    assert result["proposed_sha256"]
    assert result["backup_relative_path"]
    backup = filesystem.read_file(result["backup_relative_path"])
    assert backup == "before"
