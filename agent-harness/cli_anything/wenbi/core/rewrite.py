"""Rewrite operations for wenbi CLI harness."""
import os
import logging
from typing import Dict, Any

from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import success_result, error_result

try:
    from wenbi.main import process_input
except ImportError:
    process_input = None  # Will be mocked in tests


def rewrite(input_path: str, session: Session, style: str = "rewrite",
            start_time: str = "", end_time: str = "") -> Dict[str, Any]:
    """
    Rewrite oral/transcribed text into written form.

    Args:
        input_path: Path to input file or URL
        session: Current session with parameters
        style: 'rewrite' or 'academic'
        start_time: Optional start timestamp (HH:MM:SS)
        end_time: Optional end timestamp (HH:MM:SS)

    Returns:
        Result dict with status, output_file, text
    """
    logger = logging.getLogger(__name__)

    if process_input is None:
        return error_result(command=style, error="wenbi not installed. Install with: pip install wenbi")

    try:
        params = session.get_process_params()
        params["subcommand"] = "academic" if style == "academic" else "rewrite"

        # Handle timestamp
        if start_time and end_time:
            params["timestamp"] = {"start": start_time.strip(), "end": end_time.strip()}
        else:
            params["timestamp"] = None

        is_url = input_path.startswith(("http://", "https://", "www."))
        result = process_input(
            None if is_url else input_path,
            input_path if is_url else "",
            **params,
        )

        text_content = result[0] if result[0] and not str(result[0]).startswith("Error") else ""
        output_file = result[1] if len(result) > 1 else ""
        base_name = result[3] if len(result) > 3 else ""

        if text_content and not text_content.startswith("Error"):
            session.record("rewrite" if style == "rewrite" else "academic",
                           input_path, output_file or "", status="ok")
            return success_result(
                command=style,
                output_file=output_file or "",
                text=text_content,
                base_name=base_name,
            )
        else:
            error_msg = text_content or "Rewrite failed"
            session.record(style, input_path, status="error", error=error_msg)
            return error_result(command=style, error=error_msg)

    except Exception as e:
        session.record(style, input_path, status="error", error=str(e))
        return error_result(command=style, error=str(e))