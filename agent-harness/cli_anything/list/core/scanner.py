"""Scanner module for discovering CLI-Anything tools.

Finds both installed packages (via importlib.metadata) and generated
local directories (via filesystem glob).
"""
import os
import re
import shutil
import glob
from pathlib import Path
from importlib.metadata import distributions


def scan_installed():
    """Find all installed cli-anything-* packages.

    Returns:
        dict: {software_name: {"status": "installed", "version": str,
               "executable": str|None, "source": str|None}}
    """
    installed = {}
    for dist in distributions():
        name = dist.metadata.get("Name", "")
        if name.startswith("cli-anything-"):
            software = name.replace("cli-anything-", "")
            version = dist.version
            executable = shutil.which(f"cli-anything-{software}")

            # Try to find source path from distribution location
            source = None
            try:
                # Try to find the package source from the distribution files
                dist_files = dist.files
                if dist_files:
                    for f in dist_files:
                        # Look for agent-harness in the path
                        f_str = str(f)
                        if "agent-harness" in f_str:
                            parts = Path(f_str).parts
                            for i, p in enumerate(parts):
                                if p == "agent-harness" and i > 0:
                                    source = str(Path(*parts[:i + 2]))
                                    break
                            break
            except Exception:
                pass

            installed[software] = {
                "status": "installed",
                "version": version,
                "executable": executable,
                "source": source,
            }
    return installed


def _build_glob_patterns(base_path, depth):
    """Build glob patterns for depth-limited scanning.

    Args:
        base_path: Root directory to search from.
        depth: Maximum recursion depth. None means unlimited.

    Returns:
        list: Glob pattern strings.
    """
    base = Path(base_path)
    suffix = "agent-harness/cli_anything/*/__init__.py"

    if depth is None:
        return [str(base / "**" / suffix)]

    patterns = []
    for d in range(depth + 1):
        if d == 0:
            patterns.append(str(base / suffix))
        else:
            prefix = "/".join(["*"] * d)
            patterns.append(str(base / prefix / suffix))
    return patterns


def extract_version_from_setup(setup_path):
    """Extract version string from a setup.py file.

    Args:
        setup_path: Path to setup.py file.

    Returns:
        str|None: Version string if found, else None.
    """
    try:
        content = Path(setup_path).read_text()
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        return match.group(1) if match else None
    except Exception:
        return None


def scan_generated(search_path=".", depth=None):
    """Find all generated cli-anything tool directories.

    Scans the filesystem for agent-harness/cli_anything/*/__init__.py
    patterns to discover locally generated tools.

    Args:
        search_path: Root directory to search from (default: current directory).
        depth: Maximum recursion depth. None for unlimited.

    Returns:
        dict: {software_name: {"status": "generated", "version": str|None,
               "executable": None, "source": str}}
    """
    generated = {}

    if not os.path.exists(search_path):
        return generated

    patterns = _build_glob_patterns(search_path, depth)

    for pattern in patterns:
        for init_file in glob.glob(pattern, recursive=True):
            parts = Path(init_file).parts
            for i, p in enumerate(parts):
                if p == "cli_anything" and i + 1 < len(parts):
                    software = parts[i + 1]

                    # Walk backwards to find agent-harness and the project root
                    agent_harness_idx = None
                    for j in range(len(parts)):
                        if parts[j] == "agent-harness":
                            agent_harness_idx = j
                            break

                    if agent_harness_idx is not None:
                        source = str(Path(*parts[:agent_harness_idx + 2]))
                        setup_path = Path(*parts[:agent_harness_idx + 1]) / "setup.py"
                    else:
                        source = str(Path(init_file).parent.parent.parent)
                        setup_path = Path(init_file).parent.parent.parent / "setup.py"

                    version = extract_version_from_setup(str(setup_path))

                    # Also try to extract version from __init__.py
                    if version is None:
                        init_path = Path(init_file)
                        version = extract_version_from_init(str(init_path))

                    generated[software] = {
                        "status": "generated",
                        "version": version,
                        "executable": None,
                        "source": source,
                    }
                    break

    return generated


def extract_version_from_init(init_path):
    """Extract version from a package's __init__.py file.

    Args:
        init_path: Path to __init__.py.

    Returns:
        str|None: Version string if found, else None.
    """
    try:
        content = Path(init_path).read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        return match.group(1) if match else None
    except Exception:
        return None


def merge_tools(installed, generated):
    """Merge installed and generated tool dicts.

    If a tool appears in both, installed takes priority but the
    source path from generated is preserved.

    Args:
        installed: Dict from scan_installed().
        generated: Dict from scan_generated().

    Returns:
        dict: Merged tool dict with all software names.
    """
    merged = {}

    # Add generated first
    for name, info in generated.items():
        merged[name] = dict(info)

    # Override with installed (but keep source from generated if we have it)
    for name, info in installed.items():
        if name in merged:
            # Merge: installed status wins, but preserve source from generated
            merged[name]["status"] = "installed"
            merged[name]["version"] = info["version"]
            merged[name]["executable"] = info["executable"]
            # Keep source from generated if installed didn't find one
            if not info.get("source") and merged[name].get("source"):
                pass  # Keep generated source
            elif info.get("source"):
                merged[name]["source"] = info["source"]
        else:
            merged[name] = dict(info)

    return merged