"""Academic writing operations for wenbi CLI harness."""
import os
import logging
from typing import Dict, Any

from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import success_result, error_result

try:
    from wenbi.main import process_input
except ImportError:
    process_input = None  # Will be mocked in tests


def academic(input_path: str, session: Session,
             start_time: str = "", end_time: str = "") -> Dict[str, Any]:
    """
    Convert text to academic writing style.

    Args:
        input_path: Path to input file or URL
        session: Current session with parameters
        start_time: Optional start timestamp (HH:MM:SS)
        end_time: Optional end timestamp (HH:MM:SS)

    Returns:
        Result dict with status, output_file, text
    """
    logger = logging.getLogger(__name__)

    if process_input is None:
        return error_result(command="academic", error="wenbi not installed. Install with: pip install wenbi")

    try:
        params = session.get_process_params()
        params["subcommand"] = "academic"

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
        base_name = result[3] if len(result) > 3 else ""

        if text_content and not text_content.startswith("Error"):
            # Generate academic output file
            output_dir = session.output_dir or os.getcwd()
            out_name = f"{base_name}_academic.md" if base_name else "output_academic.md"
            output_file = os.path.join(output_dir, out_name)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text_content)

            session.record("academic", input_path, output_file, status="ok")
            return success_result(
                command="academic",
                output_file=output_file,
                text=text_content,
                base_name=base_name,
            )
        else:
            error_msg = text_content or "Academic conversion failed"
            session.record("academic", input_path, status="error", error=error_msg)
            return error_result(command="academic", error=error_msg)

    except Exception as e:
        session.record("academic", input_path, status="error", error=str(e))
        return error_result(command="academic", error=str(e))