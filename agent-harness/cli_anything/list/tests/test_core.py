"""Unit tests for cli-anything-list core modules.

Tests scanner and export logic with synthetic data, no external deps.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from cli_anything.list.core.scanner import (
    scan_installed,
    scan_generated,
    merge_tools,
    extract_version_from_setup,
    extract_version_from_init,
    _build_glob_patterns,
)
from cli_anything.list.core.export import format_json, format_table


# ── Scanner tests ──────────────────────────────────────────────


class TestScanInstalled:
    """Tests for scan_installed()."""

    def test_returns_dict(self):
        """scan_installed always returns a dict."""
        result = scan_installed()
        assert isinstance(result, dict)

    def test_wenbi_installed(self):
        """wenbi should be found as installed (we installed it)."""
        result = scan_installed()
        assert "wenbi" in result
        assert result["wenbi"]["status"] == "installed"
        assert result["wenbi"]["version"] == "0.1.0"

    def test_installed_has_executable(self):
        """Installed tools should have an executable path."""
        result = scan_installed()
        if "wenbi" in result:
            assert result["wenbi"]["executable"] is not None
            assert "cli-anything-wenbi" in result["wenbi"]["executable"]


class TestBuildGlobPatterns:
    """Tests for _build_glob_patterns()."""

    def test_unlimited_depth(self):
        """Unlimited depth should produce a recursive glob."""
        patterns = _build_glob_patterns("/tmp", None)
        assert len(patterns) == 1
        assert "**" in patterns[0]

    def test_depth_zero(self):
        """Depth 0 should produce a single non-recursive pattern."""
        patterns = _build_glob_patterns("/tmp", 0)
        assert len(patterns) == 1
        assert "**" not in patterns[0]
        assert "cli_anything/*/__init__.py" in patterns[0]

    def test_depth_one(self):
        """Depth 1 should produce two patterns (direct + one level)."""
        patterns = _build_glob_patterns("/tmp", 1)
        assert len(patterns) == 2

    def test_depth_three(self):
        """Depth 3 should produce four patterns."""
        patterns = _build_glob_patterns("/tmp", 3)
        assert len(patterns) == 4


class TestExtractVersion:
    """Tests for version extraction functions."""

    def test_extract_version_from_setup(self):
        """Should extract version from setup.py content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('version="1.2.3"\n')
            f.flush()
            result = extract_version_from_setup(f.name)
            assert result == "1.2.3"
        os.unlink(f.name)

    def test_extract_version_from_setup_single_quotes(self):
        """Should handle single quotes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("version='4.5.6'\n")
            f.flush()
            result = extract_version_from_setup(f.name)
            assert result == "4.5.6"
        os.unlink(f.name)

    def test_extract_version_from_setup_missing(self):
        """Should return None if no version found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("name='test'\n")
            f.flush()
            result = extract_version_from_setup(f.name)
            assert result is None
        os.unlink(f.name)

    def test_extract_version_from_init(self):
        """Should extract version from __init__.py content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "0.1.0"\n')
            f.flush()
            result = extract_version_from_init(f.name)
            assert result == "0.1.0"
        os.unlink(f.name)

    def test_extract_version_from_init_missing(self):
        """Should return None if no version found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(""""Module docstring.""\n""")
            f.flush()
            result = extract_version_from_init(f.name)
            assert result is None
        os.unlink(f.name)


class TestScanGenerated:
    """Tests for scan_generated()."""

    def _create_mock_harness(self, tmpdir, name="testapp", version="1.0.0"):
        """Create a mock agent-harness directory structure."""
        harness_dir = os.path.join(tmpdir, "agent-harness")
        pkg_dir = os.path.join(harness_dir, "cli_anything", name)
        os.makedirs(pkg_dir, exist_ok=True)

        # __init__.py
        init_path = os.path.join(pkg_dir, "__init__.py")
        with open(init_path, "w") as f:
            f.write(f'"""cli-anything {name}."""\n__version__ = "{version}"\n')

        # setup.py
        setup_path = os.path.join(harness_dir, "setup.py")
        with open(setup_path, "w") as f:
            f.write(f'version="{version}"\n')

        return tmpdir

    def test_finds_generated_tool(self):
        """Should find a generated tool in the scan path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_harness(tmpdir, "testapp")
            result = scan_generated(tmpdir)
            assert "testapp" in result
            assert result["testapp"]["status"] == "generated"

    def test_generated_has_version(self):
        """Should extract version from the generated tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_harness(tmpdir, "myapp", "2.5.1")
            result = scan_generated(tmpdir)
            assert result["myapp"]["version"] == "2.5.1"

    def test_generated_no_executable(self):
        """Generated tools should have executable=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_harness(tmpdir, "myapp")
            result = scan_generated(tmpdir)
            assert result["myapp"]["executable"] is None

    def test_nonexistent_path(self):
        """Should return empty dict for nonexistent path."""
        result = scan_generated("/nonexistent/path/xyz")
        assert result == {}

    def test_depth_zero(self):
        """Depth 0 should only find tools directly in the path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_harness(tmpdir, "direct")
            result = scan_generated(tmpdir, depth=0)
            # depth=0 should find it since agent-harness is directly under tmpdir
            assert "direct" in result

    def test_depth_too_shallow(self):
        """Depth 0 should NOT find tools nested deeper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure: tmpdir/subdir/agent-harness/...
            nested = os.path.join(tmpdir, "subdir")
            self._create_mock_harness(nested, "nested")
            result = scan_generated(tmpdir, depth=0)
            assert "nested" not in result

    def test_multiple_tools(self):
        """Should find multiple generated tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness_dir = os.path.join(tmpdir, "agent-harness")
            for name in ["app1", "app2", "app3"]:
                pkg_dir = os.path.join(harness_dir, "cli_anything", name)
                os.makedirs(pkg_dir, exist_ok=True)
                with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
                    f.write(f'__version__ = "1.0.0"\n')
            result = scan_generated(tmpdir)
            assert len(result) == 3
            assert "app1" in result
            assert "app2" in result
            assert "app3" in result


class TestMergeTools:
    """Tests for merge_tools()."""

    def test_merge_empty(self):
        """Merging two empty dicts returns empty."""
        assert merge_tools({}, {}) == {}

    def test_merge_generated_only(self):
        """Generated-only tools should appear as 'generated'."""
        generated = {"inkscape": {"status": "generated", "version": "1.0.0",
                                    "executable": None, "source": "./inkscape/agent-harness"}}
        result = merge_tools({}, generated)
        assert result["inkscape"]["status"] == "generated"

    def test_merge_installed_only(self):
        """Installed-only tools should appear as 'installed'."""
        installed = {"gimp": {"status": "installed", "version": "1.0.0",
                               "executable": "/usr/local/bin/cli-anything-gimp", "source": None}}
        result = merge_tools(installed, {})
        assert result["gimp"]["status"] == "installed"

    def test_merge_both(self):
        """Tool in both installed and generated should show 'installed'."""
        installed = {"gimp": {"status": "installed", "version": "1.0.0",
                               "executable": "/usr/local/bin/cli-anything-gimp", "source": None}}
        generated = {"gimp": {"status": "generated", "version": "0.9.0",
                               "executable": None, "source": "./gimp/agent-harness"}}
        result = merge_tools(installed, generated)
        # Installed takes priority
        assert result["gimp"]["status"] == "installed"
        assert result["gimp"]["executable"] is not None
        # Source from generated is preserved
        assert result["gimp"]["source"] == "./gimp/agent-harness"

    def test_merge_mixed(self):
        """Mixed tools should be properly classified."""
        installed = {"gimp": {"status": "installed", "version": "1.0.0",
                               "executable": "/usr/bin/cli-anything-gimp", "source": None}}
        generated = {"inkscape": {"status": "generated", "version": "0.5.0",
                                   "executable": None, "source": "./inkscape/agent-harness"}}
        result = merge_tools(installed, generated)
        assert len(result) == 2
        assert result["gimp"]["status"] == "installed"
        assert result["inkscape"]["status"] == "generated"


# ── Export tests ────────────────────────────────────────────────


class TestFormatJson:
    """Tests for format_json()."""

    def test_empty_tools(self):
        """Empty tools produces valid JSON with zero counts."""
        result = json.loads(format_json({}))
        assert result["total"] == 0
        assert result["tools"] == []

    def test_single_tool(self):
        """Single tool produces valid JSON."""
        tools = {"gimp": {"status": "installed", "version": "1.0.0",
                           "executable": "/usr/bin/cli-anything-gimp", "source": None}}
        result = json.loads(format_json(tools))
        assert result["total"] == 1
        assert result["installed"] == 1
        assert result["generated_only"] == 0
        assert result["tools"][0]["name"] == "gimp"

    def test_mixed_tools(self):
        """Mixed tools produce correct counts."""
        tools = {
            "gimp": {"status": "installed", "version": "1.0.0",
                      "executable": "/usr/bin/cli-anything-gimp", "source": None},
            "inkscape": {"status": "generated", "version": "0.5.0",
                          "executable": None, "source": "./inkscape/agent-harness"},
        }
        result = json.loads(format_json(tools))
        assert result["total"] == 2
        assert result["installed"] == 1
        assert result["generated_only"] == 1


class TestFormatTable:
    """Tests for format_table()."""

    def test_empty_tools(self):
        """Empty tools produces a 'no tools found' message."""
        result = format_table({})
        assert "found 0" in result
        assert "No tools found" in result

    def test_single_tool(self):
        """Single tool produces a table with one row."""
        tools = {"gimp": {"status": "installed", "version": "1.0.0",
                           "executable": "/usr/bin/cli-anything-gimp", "source": None}}
        result = format_table(tools)
        assert "gimp" in result
        assert "installed" in result
        assert "found 1" in result

    def test_tools_sorted(self):
        """Tools should be sorted by name in the table."""
        tools = {
            "z-app": {"status": "generated", "version": "1.0.0",
                       "executable": None, "source": "./z-app/agent-harness"},
            "a-app": {"status": "installed", "version": "2.0.0",
                       "executable": "/usr/bin/cli-anything-a-app", "source": None},
        }
        result = format_table(tools)
        # Get data lines (after the --- separator)
        lines = result.split("\n")
        sep_idx = next(i for i, l in enumerate(lines) if l.startswith("─"))
        data_lines = [l for l in lines[sep_idx + 1:] if l.strip()]
        # a-app should come before z-app (sorted alphabetically)
        assert data_lines[0].startswith("a-app")