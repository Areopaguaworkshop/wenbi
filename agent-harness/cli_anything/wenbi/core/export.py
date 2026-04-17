"""Export utilities for wenbi CLI harness."""
import json
from typing import Any


def format_output(data: Any, json_mode: bool = False) -> str:
    """Format output data for display.

    Args:
        data: Data to format (dict, list, str, etc.)
        json_mode: If True, output as JSON string

    Returns:
        Formatted string
    """
    if json_mode:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        return "\n".join(str(item) for item in data)
    return str(data)


def success_result(command: str, output_file: str = "", text: str = "",
                   base_name: str = "", **extra) -> dict:
    """Create a standardized success result dict.

    Truncates text preview to 200 chars.
    """
    result = {
        "status": "ok",
        "command": command,
        "output_file": output_file,
        "base_name": base_name,
    }
    if text:
        result["text_preview"] = (text[:200] + "...") if len(text) > 200 else text
    result.update(extra)
    return result


def error_result(command: str, error: str, **extra) -> dict:
    """Create a standardized error result dict."""
    result = {
        "status": "error",
        "command": command,
        "error": error,
    }
    result.update(extra)
    return result