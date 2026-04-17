"""Formatting utilities for wenbi CLI harness."""
import json
from typing import Any, List, Dict


def format_file_list(files: List[str]) -> str:
    """Format a list of file paths for display."""
    if not files:
        return "(no files)"
    return "\n".join(f"  {i+1}. {f}" for i, f in enumerate(files))


def format_history(history: List[Dict[str, Any]], limit: int = 10) -> str:
    """Format session history for display."""
    if not history:
        return "(no history)"
    recent = history[-limit:]
    lines = []
    for i, entry in enumerate(reversed(recent)):
        status_marker = "✓" if entry.get("status") == "ok" else "✗"
        cmd = entry.get("command", "?")
        inp = entry.get("input", "")
        out = entry.get("output", "")
        lines.append(f"  {status_marker} [{cmd}] {inp} -> {out}")
    return "\n".join(lines)


def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_session_info(info: Dict[str, Any]) -> str:
    """Format session info dict for display."""
    lines = []
    for k, v in info.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)