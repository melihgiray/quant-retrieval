from pathlib import Path

import yaml


def test_ci_has_bounded_read_only_execution():
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert workflow["jobs"]["test"]["timeout-minutes"] == 20


def test_ci_runs_python_lint_and_browser_checks():
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    commands = [step["run"] for step in workflow["jobs"]["test"]["steps"] if "run" in step]

    assert any(command.startswith("pytest") for command in commands)
    assert any(command.startswith("ruff check") for command in commands)
    assert any(command.startswith("node --test") for command in commands)
