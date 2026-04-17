"""Unit tests for cli-anything-wenbi core modules.

Tests use synthetic data and mock external dependencies (wenbi).
No real media files or LLM calls required.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from cli_anything.wenbi.core.project import Project
from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import (
    format_output,
    success_result,
    error_result,
)
from cli_anything.wenbi.utils.formatting import (
    format_file_list,
    format_history,
    truncate_text,
    format_session_info,
)


# ── Project Tests ──────────────────────────────────────────────

class TestProject(unittest.TestCase):
    """Tests for Project dataclass."""

    def test_project_default_creation(self):
        project = Project()
        self.assertEqual(project.name, "")
        self.assertEqual(project.files, [])
        self.assertEqual(project.metadata, {})

    def test_project_creation_with_params(self):
        project = Project(name="test", input_path="/tmp/input.mp4", output_dir="/tmp/out")
        self.assertEqual(project.name, "test")
        self.assertEqual(project.input_path, "/tmp/input.mp4")

    def test_project_add_file(self):
        project = Project()
        project.add_file("/tmp/test.md")
        self.assertIn(os.path.abspath("/tmp/test.md"), project.files)
        project.add_file("/tmp/test.md")
        self.assertEqual(len(project.files), 1)

    def test_project_remove_file(self):
        project = Project()
        project.add_file("/tmp/test.md")
        project.remove_file("/tmp/test.md")
        self.assertEqual(len(project.files), 0)

    def test_project_list_files(self):
        project = Project()
        project.add_file("/tmp/a.md")
        project.add_file("/tmp/b.md")
        self.assertEqual(len(project.list_files()), 2)

    def test_project_to_dict(self):
        project = Project(name="my-project")
        d = project.to_dict()
        self.assertEqual(d["name"], "my-project")
        self.assertIsInstance(d, dict)

    def test_project_from_dict(self):
        data = {"name": "test", "input_path": "/in", "output_dir": "/out",
                "files": [], "metadata": {"key": "value"}}
        project = Project.from_dict(data)
        self.assertEqual(project.name, "test")
        self.assertEqual(project.metadata["key"], "value")

    def test_project_from_dict_ignores_unknown_keys(self):
        data = {"name": "test", "unknown_key": "ignored"}
        project = Project.from_dict(data)
        self.assertEqual(project.name, "test")
        self.assertFalse(hasattr(project, "unknown_key"))

    def test_project_validate_nonexistent_file(self):
        project = Project(files=["/nonexistent/file.md"])
        errors = project.validate()
        self.assertTrue(len(errors) > 0)
        self.assertIn("File not found", errors[0])

    def test_project_validate_nonexistent_output_dir(self):
        project = Project(output_dir="/nonexistent/dir")
        errors = project.validate()
        self.assertTrue(len(errors) > 0)

    def test_project_info(self):
        project = Project(name="test", files=["/tmp/a.md"])
        info = project.info()
        self.assertEqual(info["name"], "test")
        self.assertEqual(info["file_count"], 1)


# ── Session Tests ──────────────────────────────────────────────

class TestSession(unittest.TestCase):
    """Tests for Session dataclass."""

    def test_session_default_creation(self):
        session = Session()
        self.assertEqual(session.llm, "ollama/qwen3")
        self.assertEqual(session.lang, "Chinese")
        self.assertEqual(session.history, [])

    def test_session_get_process_params(self):
        session = Session(llm="ollama/qwen3", lang="Japanese", chunk_length=30)
        params = session.get_process_params()
        self.assertEqual(params["llm"], "ollama/qwen3")
        self.assertEqual(params["lang"], "Japanese")
        self.assertEqual(params["chunk_length"], 30)
        self.assertNotIn("history", params)

    def test_session_record_history(self):
        session = Session()
        session.record("rewrite", "/input.md", "/output.md", status="ok")
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0]["command"], "rewrite")
        self.assertEqual(session.history[0]["status"], "ok")

    def test_session_record_error_history(self):
        session = Session()
        session.record("translate", "/in.md", status="error", error="LLM unavailable")
        self.assertEqual(session.history[0]["status"], "error")
        self.assertEqual(session.history[0]["error"], "LLM unavailable")

    def test_session_set_value_string(self):
        session = Session()
        result = session.set_value("lang", "Japanese")
        self.assertEqual(session.lang, "Japanese")
        self.assertIn("lang", result)

    def test_session_set_value_int(self):
        session = Session()
        session.set_value("chunk_length", "50")
        self.assertEqual(session.chunk_length, 50)

    def test_session_set_value_float(self):
        session = Session()
        session.set_value("temperature", "0.5")
        self.assertEqual(session.temperature, 0.5)

    def test_session_set_value_bool(self):
        session = Session()
        session.set_value("verbose", "true")
        self.assertTrue(session.verbose)
        session.set_value("verbose", "false")
        self.assertFalse(session.verbose)

    def test_session_set_value_unknown_key(self):
        session = Session()
        with self.assertRaises(ValueError):
            session.set_value("nonexistent_key", "value")

    def test_session_reset(self):
        session = Session(lang="Japanese", chunk_length=50)
        session.record("test", "/in", status="ok")
        session.reset()
        self.assertEqual(session.lang, "Chinese")
        self.assertEqual(session.chunk_length, 20)
        self.assertEqual(len(session.history), 0)

    def test_session_save_and_load(self):
        session = Session(lang="German", llm="openai/gpt-4")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            session.save(path)
            loaded = Session.load(path)
            self.assertEqual(loaded.lang, "German")
            self.assertEqual(loaded.llm, "openai/gpt-4")
        finally:
            os.unlink(path)

    def test_session_show(self):
        session = Session(llm="test-model")
        info = session.show()
        self.assertEqual(info["llm"], "test-model")
        self.assertIn("history_count", info)

    def test_session_to_dict(self):
        session = Session(lang="French")
        d = session.to_dict()
        self.assertEqual(d["lang"], "French")
        self.assertIsInstance(d, dict)


# ── Core Rewrite Tests (with mocked process_input) ─────────────

class TestRewriteCore(unittest.TestCase):
    """Tests for the rewrite core module with mocked process_input."""

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_success(self, mock_process):
        from cli_anything.wenbi.core.rewrite import rewrite

        mock_process.return_value = ("Rewritten text here.", "/out/rewritten.md", None, "input")
        session = Session(output_dir="/tmp")
        result = rewrite("/input.md", session, style="rewrite")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["command"], "rewrite")

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_academic_style(self, mock_process):
        from cli_anything.wenbi.core.rewrite import rewrite

        mock_process.return_value = ("Academic text.", "/out/academic.md", None, "input")
        session = Session(output_dir="/tmp")
        result = rewrite("/input.md", session, style="academic")
        self.assertEqual(result["status"], "ok")

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_error(self, mock_process):
        from cli_anything.wenbi.core.rewrite import rewrite

        mock_process.return_value = ("Error: Failed", None, None, None)
        session = Session(output_dir="/tmp")
        result = rewrite("/input.md", session)
        self.assertEqual(result["status"], "error")

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_exception(self, mock_process):
        from cli_anything.wenbi.core.rewrite import rewrite

        mock_process.side_effect = Exception("LLM not available")
        session = Session(output_dir="/tmp")
        result = rewrite("/input.md", session)
        self.assertEqual(result["status"], "error")
        self.assertIn("LLM not available", result["error"])


# ── Core Translate Tests (with mocked process_input) ────────────

class TestTranslateCore(unittest.TestCase):
    """Tests for the translate core module with mocked process_input."""

    @patch("cli_anything.wenbi.core.translate.process_input")
    def test_translate_success(self, mock_process):
        from cli_anything.wenbi.core.translate import translate

        mock_process.return_value = ("Translated text.", "/out/translated.md", None, "input")
        session = Session(output_dir="/tmp", lang="Japanese")
        result = translate("/input.md", session)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["command"], "translate")

    @patch("cli_anything.wenbi.core.translate.process_input")
    def test_translate_error(self, mock_process):
        from cli_anything.wenbi.core.translate import translate

        mock_process.return_value = ("Error: translation failed", None, None, None)
        session = Session(output_dir="/tmp")
        result = translate("/input.md", session)
        self.assertEqual(result["status"], "error")


# ── Core Academic Tests (with mocked process_input) ─────────────

class TestAcademicCore(unittest.TestCase):
    """Tests for the academic core module with mocked process_input."""

    @patch("cli_anything.wenbi.core.academic.process_input")
    def test_academic_success(self, mock_process):
        from cli_anything.wenbi.core.academic import academic

        mock_process.return_value = ("Academic text.", "/out/academic.md", None, "input")
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(output_dir=tmpdir)
            result = academic("/input.md", session)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["command"], "academic")

    @patch("cli_anything.wenbi.core.academic.process_input")
    def test_academic_error(self, mock_process):
        from cli_anything.wenbi.core.academic import academic

        mock_process.side_effect = Exception("Processing failed")
        session = Session(output_dir="/tmp")
        result = academic("/input.md", session)
        self.assertEqual(result["status"], "error")


# ── Export Tests ────────────────────────────────────────────────

class TestExport(unittest.TestCase):
    """Tests for export formatting utilities."""

    def test_format_output_json(self):
        data = {"key": "value", "number": 42}
        result = format_output(data, json_mode=True)
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")

    def test_format_output_dict(self):
        data = {"name": "test", "count": 5}
        result = format_output(data, json_mode=False)
        self.assertIn("name: test", result)
        self.assertIn("count: 5", result)

    def test_format_output_list(self):
        data = ["a", "b", "c"]
        result = format_output(data, json_mode=False)
        self.assertIn("a", result)

    def test_format_output_string(self):
        result = format_output("hello world", json_mode=False)
        self.assertEqual(result, "hello world")

    def test_success_result_truncation(self):
        long_text = "A" * 300
        result = success_result("rewrite", output_file="/out.md", text=long_text)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["text_preview"].endswith("..."))
        self.assertEqual(len(result["text_preview"]), 203)

    def test_success_result_short_text(self):
        result = success_result("test", text="Short")
        self.assertEqual(result["text_preview"], "Short")

    def test_error_result(self):
        result = error_result("translate", "LLM unavailable", extra_key="value")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "LLM unavailable")
        self.assertEqual(result["extra_key"], "value")


# ── Formatting Tests ────────────────────────────────────────────

class TestFormatting(unittest.TestCase):
    """Tests for formatting utilities."""

    def test_format_file_list_empty(self):
        self.assertEqual(format_file_list([]), "(no files)")

    def test_format_file_list(self):
        result = format_file_list(["/a.md", "/b.md"])
        self.assertIn("/a.md", result)
        self.assertIn("/b.md", result)

    def test_format_history_empty(self):
        self.assertEqual(format_history([]), "(no history)")

    def test_format_history(self):
        history = [
            {"command": "rewrite", "input": "/in.md", "output": "/out.md", "status": "ok", "error": ""},
        ]
        result = format_history(history)
        self.assertIn("rewrite", result)
        self.assertIn("✓", result)

    def test_format_history_error(self):
        history = [
            {"command": "translate", "input": "/in.md", "output": "", "status": "error", "error": "failed"},
        ]
        result = format_history(history)
        self.assertIn("✗", result)

    def test_truncate_text_short(self):
        self.assertEqual(truncate_text("hello", 10), "hello")

    def test_truncate_text_long(self):
        text = "a" * 300
        result = truncate_text(text, 200)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(len(result), 203)

    def test_format_session_info(self):
        info = {"name": "test", "llm": "ollama/qwen3"}
        result = format_session_info(info)
        self.assertIn("name: test", result)


# ── CLI Tests ──────────────────────────────────────────────────

class TestCLI(unittest.TestCase):
    """Tests for Click CLI commands."""

    def setUp(self):
        from click.testing import CliRunner
        from cli_anything.wenbi.wenbi_cli import cli
        self.runner = CliRunner()
        self.cli = cli

    def test_cli_help(self):
        result = self.runner.invoke(self.cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Stateful CLI harness", result.output)

    def test_cli_json_flag(self):
        result = self.runner.invoke(self.cli, ["--json", "session", "show"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("llm", data)

    def test_session_show(self):
        result = self.runner.invoke(self.cli, ["session", "show"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ollama/qwen3", result.output)

    def test_session_set(self):
        result = self.runner.invoke(self.cli, ["session", "set", "lang", "Japanese"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Japanese", result.output)

    def test_session_set_unknown_key(self):
        """Setting an unknown key should produce an error message."""
        result = self.runner.invoke(self.cli, ["session", "set", "nonexistent_key", "value"])
        self.assertIn("error", result.output.lower())

    def test_session_reset(self):
        result = self.runner.invoke(self.cli, ["session", "reset"])
        self.assertEqual(result.exit_code, 0)

    def test_session_save_load_roundtrip(self):
        """Test save and load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_session.json")
            result = self.runner.invoke(self.cli, ["session", "save", path])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(path))
            loaded = Session.load(path)
            self.assertEqual(loaded.llm, "ollama/qwen3")

    def test_session_history_empty(self):
        result = self.runner.invoke(self.cli, ["session", "history"])
        self.assertEqual(result.exit_code, 0)

    def test_project_new(self):
        result = self.runner.invoke(self.cli, ["project", "new", "my-project"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("my-project", result.output)

    def test_project_add_file_not_exist(self):
        result = self.runner.invoke(self.cli, ["project", "add-file", "/nonexistent/file.md"])
        self.assertIn("error", result.output.lower())

    def test_project_list_files_empty(self):
        result = self.runner.invoke(self.cli, ["project", "list-files"])
        self.assertEqual(result.exit_code, 0)

    def test_project_info(self):
        result = self.runner.invoke(self.cli, ["project", "info"])
        self.assertEqual(result.exit_code, 0)

    def test_project_validate_no_files(self):
        result = self.runner.invoke(self.cli, ["project", "validate"])
        self.assertEqual(result.exit_code, 0)

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_command(self, mock_process):
        mock_process.return_value = ("Rewritten text.", "/out/rewritten.md", None, "input")
        result = self.runner.invoke(self.cli, ["rewrite", "/tmp/input.md"])
        self.assertEqual(result.exit_code, 0)

    @patch("cli_anything.wenbi.core.translate.process_input")
    def test_translate_command(self, mock_process):
        mock_process.return_value = ("Translated text.", "/out/translated.md", None, "input")
        result = self.runner.invoke(self.cli, ["translate", "/tmp/input.md"])
        self.assertEqual(result.exit_code, 0)

    @patch("cli_anything.wenbi.core.academic.process_input")
    def test_academic_command(self, mock_process):
        mock_process.return_value = ("Academic text.", "/out/academic.md", None, "input")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(self.cli, ["academic", "/tmp/input.md", "-o", tmpdir])
            self.assertEqual(result.exit_code, 0)

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_json_output(self, mock_process):
        mock_process.return_value = ("Text.", "/out.md", None, "input")
        result = self.runner.invoke(self.cli, ["--json", "rewrite", "/tmp/input.md"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")


# ── Subprocess CLI Tests ────────────────────────────────────────

class TestCLISubprocess(unittest.TestCase):
    """Test the installed CLI via subprocess."""

    @staticmethod
    def _resolve_cli(name):
        """Resolve the CLI command path, handling dev and installed modes."""
        import shutil
        cli_path = shutil.which(name)
        if cli_path:
            return cli_path
        # Check for development installation
        dev_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "setup.py")
        if os.path.exists(dev_path):
            return name
        return name

    @unittest.skipUnless(
        os.environ.get("CLI_ANYTHING_FORCE_INSTALLED"),
        "Skipping subprocess test. Set CLI_ANYTHING_FORCE_INSTALLED=1 to run."
    )
    def test_cli_installed_and_runs(self):
        import subprocess
        cli_cmd = self._resolve_cli("cli-anything-wenbi")
        result = subprocess.run(
            [cli_cmd, "--help"],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Stateful CLI harness", result.stdout)

    @unittest.skipUnless(
        os.environ.get("CLI_ANYTHING_FORCE_INSTALLED"),
        "Skipping subprocess test. Set CLI_ANYTHING_FORCE_INSTALLED=1 to run."
    )
    def test_cli_session_show_subprocess(self):
        import subprocess
        cli_cmd = self._resolve_cli("cli-anything-wenbi")
        result = subprocess.run(
            [cli_cmd, "session", "show"],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("ollama/qwen3", result.stdout)

    @unittest.skipUnless(
        os.environ.get("CLI_ANYTHING_FORCE_INSTALLED"),
        "Skipping subprocess test. Set CLI_ANYTHING_FORCE_INSTALLED=1 to run."
    )
    def test_cli_json_output_subprocess(self):
        import subprocess
        cli_cmd = self._resolve_cli("cli-anything-wenbi")
        result = subprocess.run(
            [cli_cmd, "--json", "session", "show"],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertIn("llm", data)


if __name__ == "__main__":
    unittest.main()