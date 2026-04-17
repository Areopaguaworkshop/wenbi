"""PPT (slide extraction + speech combination) operations for wenbi CLI harness."""
import os
import logging
from typing import Dict, Any, List

from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import success_result, error_result

try:
    from wenbi.cli import handle_ppt_command
except ImportError:
    handle_ppt_command = None  # Will be mocked in tests


def ppt(video_path: str, session: Session,
        frame_interval: int = 60,
        cropped_slide: str = "",
        ppt_file: str = "",
        no_ocr: bool = False,
        no_clean: bool = False,
        ssim_threshold: float = 0.98,
        hist_threshold: float = 0.15,
        start_time: str = "",
        end_time: str = "") -> Dict[str, Any]:
    """
    Extract slides from video and combine with speech.

    Supports three methods:
    - Frame method (default): Extract frames, OCR, combine with speech
    - Cropped-slide method: Detect ROI, crop slides, OCR, combine
    - PPT method: Use existing PPT/PDF, align with video timestamps

    Args:
        video_path: Path to video file or URL
        session: Current session with parameters
        frame_interval: Seconds between frame extractions
        cropped_slide: ROI for cropped-slide method ('auto' or 'x0,y0,x1,y1')
        ppt_file: Path to PPT/PDF/image for PPT method
        no_ocr: Skip OCR, embed images as base64
        no_clean: Keep timestamps and image references
        ssim_threshold: SSIM threshold for deduplication
        hist_threshold: Histogram threshold for deduplication
        start_time: Optional start timestamp
        end_time: Optional end timestamp

    Returns:
        Result dict with status, output_files
    """
    logger = logging.getLogger(__name__)

    if handle_ppt_command is None:
        return error_result(command="ppt", error="wenbi not installed. Install with: pip install wenbi")

    try:
        output_dir = session.output_dir or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Build args namespace mimicking argparse result
        import argparse
        args = argparse.Namespace(
            input=video_path,
            output_dir=output_dir,
            config="",
            llm=session.llm,
            lang=session.lang,
            chunk_length=session.chunk_length,
            max_tokens=session.max_tokens,
            timeout=session.timeout,
            temperature=session.temperature,
            transcribe_model=session.transcribe_model,
            multi_language=session.multi_language,
            transcribe_lang=session.transcribe_lang,
            cite_timestamps=True,
            verbose=session.verbose,
            frame_interval=frame_interval,
            each_roi=False,
            roi=None,
            max_slides=20,
            no_deduplicate=False,
            similarity_threshold=0.85,
            dedup_method="both",
            ssim_threshold=ssim_threshold,
            hist_threshold=hist_threshold,
            cropped_slide=cropped_slide if cropped_slide else None,
            ppt=ppt_file if ppt_file else "",
            no_ocr=no_ocr,
            no_clean=no_clean,
            start_time=start_time,
            end_time=end_time,
        )

        handle_ppt_command(args)

        # Find output files
        combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
        combine_clean_md = os.path.join(output_dir, f"{base_name}_combine_clean.md")

        output_files = {}
        if os.path.exists(combine_md):
            output_files["combined"] = combine_md
        if os.path.exists(combine_clean_md):
            output_files["cleaned"] = combine_clean_md

        session.record("ppt", video_path, combine_md if os.path.exists(combine_md) else "", status="ok")
        return success_result(
            command="ppt",
            output_file=combine_md if os.path.exists(combine_md) else "",
            base_name=base_name,
            output_files=output_files,
        )

    except Exception as e:
        session.record("ppt", video_path, status="error", error=str(e))
        return error_result(command="ppt", error=str(e))