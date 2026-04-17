"""End-to-end tests for cli-anything-list.

Tests the full CLI pipeline including subprocess invocation.
"""
import json
import os
import subprocess
import tempfile

import pytest


def _resolve_cli(name="cli-anything-list"):
    """Resolve CLI command for subprocess testing.

    Uses CLI_ANYTHING_FORCE_INSTALLED=1 to test installed binary.
    """
    env = os.environ.copy()
    env["CLI_ANYTHING_FORCE_INSTALLED"] = "1"

    # Try the installed binary first
    which_result = subprocess.run(
        ["which", name],
        capture_output=True, text=True, env=env
    )
    if which_result.returncode == 0:
        return name

    # Fallback to python -m invocation
    return ["python", "-m", "cli_anything.list.list_cli"]


class TestCLIHelp:
    """Test CLI help and basic invocation."""

    def test_help_flag(self):
        """--help should show usage info and exit 0."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli, "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "CLI-Anything" in result.stdout or "cli-anything" in result.stdout.lower()

    def test_default_invocation(self):
        """Running without args should list tools and exit 0."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "CLI-Anything" in result.stdout

    def test_json_flag(self):
        """--json should produce valid JSON output."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli, "--json"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "tools" in data
        assert "total" in data
        assert "installed" in data
        assert "generated_only" in data

    def test_json_contains_wenbi(self):
        """wenbi should appear in installed tools."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli, "--json"],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        names = [t["name"] for t in data["tools"]]
        assert "wenbi" in names
        # Find wenbi entry
        wenbi = next(t for t in data["tools"] if t["name"] == "wenbi")
        assert wenbi["status"] == "installed"
        assert wenbi["version"] == "0.1.0"


class TestCLIPathOption:
    """Test --path option."""

    def test_explicit_path(self):
        """--path should find tools at the specified directory."""
        cli = _resolve_cli()
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        result = subprocess.run(
            [cli, "--path", project_dir],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "wenbi" in result.stdout

    def test_nonexistent_path(self):
        """Nonexistent path should produce an error."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli, "--path", "/nonexistent/path/xyz123"],
            capture_output=True, text=True
        )
        assert result.returncode != 0


class TestCLIDepthOption:
    """Test --depth option."""

    def test_depth_zero(self):
        """--depth 0 should only scan the current directory."""
        cli = _resolve_cli()
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        result = subprocess.run(
            [cli, "--path", project_dir, "--depth", "0"],
            capture_output=True, text=True
        )
        assert result.returncode == 0

    def test_depth_json(self):
        """--depth with --json should work together."""
        cli = _resolve_cli()
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        result = subprocess.run(
            [cli, "--path", project_dir, "--depth", "0", "--json"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "total" in data


class TestCLISubprocess:
    """Subprocess tests verifying the installed CLI binary."""

    def test_resolve_cli_works(self):
        """_resolve_cli should return a valid command."""
        cli = _resolve_cli()
        result = subprocess.run(
            [cli, "--json"],
            capture_output=True, text=True,
            env={**os.environ, "CLI_ANYTHING_FORCE_INSTALLED": "1"}
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["total"] >= 1

    def test_list_and_wenbi_both_found(self):
        """Both list and wenbi tools should be discoverable."""
        cli = _resolve_cli()
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )))
        result = subprocess.run(
            [cli, "--json", "--path", project_dir],
            capture_output=True, text=True,
            env={**os.environ, "CLI_ANYTHING_FORCE_INSTALLED": "1"}
        )
        data = json.loads(result.stdout)
        names = [t["name"] for t in data["tools"]]
        assert "wenbi" in names
        assert "list" in names