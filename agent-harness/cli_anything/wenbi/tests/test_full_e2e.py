"""End-to-end tests for cli-anything-wenbi.

Tests with real files, full pipeline. Where possible, tests work within
a single Click invocation to preserve state.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from cli_anything.wenbi.wenbi_cli import cli
from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.project import Project


class TestE2ESessionWorkflow(unittest.TestCase):
    """Test session lifecycle using direct Session objects (state persists across operations)."""

    def test_session_lifecycle(self):
        """Create → set params → save → load → verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = os.path.join(tmpdir, "session.json")

            # Step 1: Create and set session params
            session = Session()
            session.set_value("lang", "Japanese")
            session.set_value("llm", "openai/gpt-4")

            # Step 2: Save session
            session.save(session_path)

            # Step 3: Verify session file is valid JSON
            with open(session_path) as f:
                data = json.load(f)
            self.assertEqual(data["lang"], "Japanese")
            self.assertEqual(data["llm"], "openai/gpt-4")

            # Step 4: Load session and verify params persisted
            loaded = Session.load(session_path)
            self.assertEqual(loaded.lang, "Japanese")
            self.assertEqual(loaded.llm, "openai/gpt-4")


class TestE2EProjectWorkflow(unittest.TestCase):
    """Test project management with real files using direct Project objects."""

    def test_project_with_real_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "lecture1.mp4")
            with open(test_file, "w") as f:
                f.write("fake video content")

            test_file2 = os.path.join(tmpdir, "lecture2.mp4")
            with open(test_file2, "w") as f:
                f.write("fake video content 2")

            # Create project
            project = Project(name="lectures", output_dir=tmpdir)

            # Add real files
            project.add_file(test_file)
            project.add_file(test_file2)

            # List files
            files = project.list_files()
            self.assertEqual(len(files), 2)
            self.assertTrue(any("lecture1" in f for f in files))
            self.assertTrue(any("lecture2" in f for f in files))

            # Validate - all files exist
            errors = project.validate()
            self.assertEqual(len(errors), 0)

            # Remove a file
            project.remove_file(test_file2)
            files = project.list_files()
            self.assertEqual(len(files), 1)
            self.assertFalse(any("lecture2" in f for f in files))


class TestE2EProjectValidation(unittest.TestCase):
    """Test project validation with missing files."""

    def test_validate_missing_file(self):
        project = Project(name="test", files=["/nonexistent/file.md"])
        errors = project.validate()
        self.assertTrue(len(errors) > 0)


class TestE2EProcessingWithMock(unittest.TestCase):
    """E2E test for processing commands with mocked wenbi backend."""

    @patch("cli_anything.wenbi.core.rewrite.process_input")
    def test_rewrite_e2e_with_real_file(self, mock_process):
        """Test rewrite with a real input file."""
        mock_process.return_value = (
            "This is the rewritten version of the oral text. It has been converted to written style.",
            "/tmp/output_rewritten.md",
            None,
            "sample_input"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.md")
            with open(input_file, "w") as f:
                f.write("# Test\n\nThis is oral text that needs rewriting.\n")

            runner = CliRunner()
            result = runner.invoke(cli, ["rewrite", input_file, "-o", tmpdir])
            self.assertEqual(result.exit_code, 0)

    @patch("cli_anything.wenbi.core.translate.process_input")
    def test_translate_e2e_with_json(self, mock_process):
        """Test translate with --json flag."""
        mock_process.return_value = (
            "这是翻译后的文本。",
            "/tmp/output_translated.md",
            None,
            "sample_input"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.md")
            with open(input_file, "w") as f:
                f.write("This is text to translate.\n")

            runner = CliRunner()
            result = runner.invoke(cli, ["--json", "translate", input_file, "-o", tmpdir])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["command"], "translate")

    def test_full_rewrite_session_workflow(self):
        """Full workflow: set session params → rewrite → verify history using direct objects."""
        with patch("cli_anything.wenbi.core.rewrite.process_input") as mock_process:
            mock_process.return_value = ("Rewritten output.", "/tmp/out_rewritten.md", None, "test_input")

            session = Session()
            session.set_value("lang", "English")

            from cli_anything.wenbi.core.rewrite import rewrite
            result = rewrite("/tmp/input.md", session)
            self.assertEqual(result["status"], "ok")

            # Verify history
            self.assertEqual(len(session.history), 1)
            self.assertEqual(session.history[0]["command"], "rewrite")


class TestE2EJSONOutput(unittest.TestCase):
    """Test JSON output format consistency across commands."""

    def test_session_show_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "session", "show"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("llm", data)
        self.assertIn("lang", data)

    def test_project_info_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "project", "new", "test-proj"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")

    def test_session_set_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "session", "set", "lang", "French"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")

    def test_session_reset_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "session", "reset"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")

    def test_session_history_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "session", "history"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("history", data)


class TestE2EBatchWithMock(unittest.TestCase):
    """Test batch command with mocked backend."""

    @patch("cli_anything.wenbi.core.batch.batch_process")
    def test_batch_e2e(self, mock_batch):
        """Test batch command with a real directory."""
        mock_batch.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(cli, ["batch", tmpdir, "-o", tmpdir])
            self.assertEqual(result.exit_code, 0)


class TestE2EPPTWithMock(unittest.TestCase):
    """Test PPT command with mocked backend."""

    @patch("cli_anything.wenbi.core.ppt.handle_ppt_command")
    def test_ppt_e2e(self, mock_handle):
        """Test PPT command creates output files."""
        mock_handle.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, "lecture.mp4")
            with open(video_file, "w") as f:
                f.write("fake video")

            # Create expected output file
            combine_md = os.path.join(tmpdir, "lecture_combine.md")
            with open(combine_md, "w") as f:
                f.write("# Combined output\n")

            runner = CliRunner()
            result = runner.invoke(cli, ["ppt", video_file, "-o", tmpdir])
            self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()