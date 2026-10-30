import inspect
import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
from database.repository import AgentRepository


def test_repository_contracts_match_frontend_calls():
    finish = inspect.signature(AgentRepository.finish_run)
    decide = inspect.signature(AgentRepository.decide_approval)
    change = inspect.signature(AgentRepository.record_file_change)

    finish.bind(
        None,
        run_id="run",
        status="completed",
        response="done",
        error=None,
    )
    decide.bind(None, approval_id="approval", status="approved")
    change.bind(
        None,
        run_id="run",
        relative_path="file.py",
        operation="edit",
        original_sha256=None,
        proposed_sha256="hash",
        backup_relative_path=None,
        status="completed",
    )
