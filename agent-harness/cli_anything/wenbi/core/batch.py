"""Batch processing operations for wenbi CLI harness."""
import os
import logging
from typing import Dict, Any, Optional

from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import success_result, error_result

try:
    from wenbi.batch import batch_process
except ImportError:
    batch_process = None  # Will be mocked in tests


def batch(input_dir: str, session: Session,
          md_output: str = "",
          config: str = "") -> Dict[str, Any]:
    """
    Batch process all media files in a directory.

    Args:
        input_dir: Path to directory containing media files
        session: Current session with parameters
        md_output: Path for combined markdown output file
        config: Path to YAML configuration file

    Returns:
        Result dict with status and summary
    """
    logger = logging.getLogger(__name__)

    if batch_process is None:
        return error_result(command="batch", error="wenbi not installed. Install with: pip install wenbi")

    try:
        output_dir = session.output_dir or ""
        kwargs = {
            "output_dir": output_dir,
            "verbose": session.verbose,
            "md_output": md_output if md_output else None,
        }

        batch_process(input_dir, **kwargs)

        session.record("batch", input_dir, md_output or output_dir, status="ok")
        return success_result(
            command="batch",
            output_file=md_output or output_dir,
            input_dir=input_dir,
        )

    except Exception as e:
        session.record("batch", input_dir, status="error", error=str(e))
        return error_result(command="batch", error=str(e))