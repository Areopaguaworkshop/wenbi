#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import yaml

from wenbi.download import download_all
from wenbi.gui import launch_gui
from wenbi.main import process_input
from wenbi.model import (
    academic,
    combine_speech_and_slides,
    combine_speech_and_slides_enhanced,
    convert_slides_to_markdown,
    read_markdown_file,
    rewrite,
    translate,
)


def setup_logging(verbose=False):
    """Setup logging configuration based on verbose flag"""
    if verbose:
        level = logging.DEBUG
        format_str = "[VERBOSE] %(message)s"
    else:
        level = logging.INFO
        format_str = "%(message)s"

    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and add a new StreamHandler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    return logging.getLogger(__name__)


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def combine_markdown_files(
    outputs, output_dir, final_filename="combined_output.md", verbose=False
):
    """Combine multiple markdown outputs into a single file"""
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug(f"Combining {len(outputs)} markdown outputs into {final_filename}")

    combined_path = os.path.join(output_dir, final_filename)
    with open(combined_path, "w", encoding="utf-8") as f:
        for idx, (title, content) in enumerate(outputs):
            if verbose:
                logger.debug(f"Adding section {idx + 1}/{len(outputs)}: {title}")

            if idx > 0:
                f.write("\n---\n\n")  # Separator between sections
            f.write(f"# {title}\n\n")
            # Read and append content from markdown file
            if os.path.isfile(content):
                with open(content, "r", encoding="utf-8") as mf:
                    f.write(mf.read())
            else:
                f.write(content)

    if verbose:
        logger.debug(f"Combined output saved to: {combined_path}")

    return combined_path


def parse_timestamp(start_time, end_time):
    """Parse start and end time strings in the format HH:MM:SS"""
    try:
        return {"start": start_time.strip(), "end": end_time.strip()}
    except ValueError:
        print(
            "Error: Invalid timestamp format. Use HH:MM:SS for both start and end times."
        )
        sys.exit(1)


def process_yaml_config(config, verbose=False):
    """Process YAML config supporting both single and multiple input formats"""
    logger = logging.getLogger(__name__)
    outputs = []

    if verbose:
        logger.debug("Processing YAML configuration")
        logger.debug(f"Config keys: {list(config.keys())}")

    # Handle single input with segments
    if "input" in config and "segments" in config:
        input_path = config["input"]
        params = {**config}
        params.pop("input", None)
        params.pop("segments", None)
        params["verbose"] = verbose

        if verbose:
            logger.debug(f"Processing single input with segments: {input_path}")
            logger.debug(f"Found {len(config['segments'])} segments")

        for idx, segment in enumerate(config["segments"], 1):
            # Make all segment fields optional
            if not isinstance(segment, dict):
                continue

            if verbose:
                logger.debug(f"Processing segment {idx}/{len(config['segments'])}")

            # Get timestamp if provided, otherwise process whole file
            if "start_time" in segment and "end_time" in segment:
                params["timestamp"] = parse_timestamp(
                    segment["start_time"], segment["end_time"]
                )
                if verbose:
                    logger.debug(
                        f"Segment timestamp: {segment['start_time']} - {segment['end_time']}"
                    )
            else:
                params["timestamp"] = None

            # Get output_wav if provided
            params["output_wav"] = segment.get("output_wav", "")

            result = process_input(
                input_path
                if not input_path.startswith(("http://", "https://", "www."))
                else None,
                input_path
                if input_path.startswith(("http://", "https://", "www."))
                else "",
                **params,
            )

            if result[0] and result[3]:
                # Use title if provided, otherwise use generated base_name
                title = segment.get(
                    "title", f"Segment {idx}" if params["timestamp"] else result[3]
                )
                outputs.append((title, result[1] or result[0]))
                if verbose:
                    logger.debug(f"Segment {idx} processed successfully: {title}")

        # Combine outputs into single file
        if outputs:
            output_dir = config.get("output_dir", "")
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            final_output = combine_markdown_files(
                outputs, output_dir, f"{base_name}_combined.md", verbose
            )
            print(f"Combined output saved to: {final_output}")

    # Handle multiple inputs with or without segments
    if "inputs" in config:
        if verbose:
            logger.debug(f"Processing multiple inputs: {len(config['inputs'])} files")

        for input_idx, input_config in enumerate(config["inputs"], 1):
            input_path = input_config["input"]

            if verbose:
                logger.debug(
                    f"Processing input {input_idx}/{len(config['inputs'])}: {input_path}"
                )

            # If no segments defined, process the entire file
            if "segments" not in input_config:
                params = {**config, **input_config}
                params.pop("inputs", None)
                params.pop("input", None)
                params["verbose"] = verbose

                result = process_input(
                    input_path
                    if not input_path.startswith(("http://", "https://", "www."))
                    else None,
                    input_path
                    if input_path.startswith(("http://", "https://", "www."))
                    else "",
                    **params,
                )

                if result[0] and result[3]:
                    # Use filename as title for full file processing
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    outputs.append((base_name, result[1] or result[0]))
                continue

            # Process segments if they exist
            for idx, segment in enumerate(input_config.get("segments", []), 1):
                if not isinstance(segment, dict):
                    continue

                params = {**config, **input_config}
                params.pop("inputs", None)
                params.pop("input", None)
                params.pop("segments", None)
                params["verbose"] = verbose

                # Make timestamp optional
                if "start_time" in segment and "end_time" in segment:
                    params["timestamp"] = parse_timestamp(
                        segment["start_time"], segment["end_time"]
                    )
                else:
                    params["timestamp"] = None

                params["output_wav"] = segment.get("output_wav", "")

                result = process_input(
                    input_path
                    if not input_path.startswith(("http://", "https://", "www."))
                    else None,
                    input_path
                    if input_path.startswith(("http://", "https://", "www."))
                    else "",
                    **params,
                )

                if result[0] and result[3]:
                    title = segment.get(
                        "title", f"Segment {idx}" if params["timestamp"] else result[3]
                    )
                    outputs.append((title, result[1] or result[0]))

    return outputs


def is_video_audio_or_url(file_path, url):
    """Check if input is video, audio, or URL"""
    if url and url.strip():
        return True

    if file_path:
        video_extensions = (
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".flv",
            ".wmv",
            ".m4v",
            ".webm",
        )
        audio_extensions = (".mp3", ".flac", ".aac", ".ogg", ".m4a", ".opus")
        return file_path.lower().endswith(video_extensions + audio_extensions)

    return False


def validate_transcription_args(args):
    """Validate that transcription-related arguments are only used with video/audio/URL inputs"""
    # Check if input is video, audio, or URL
    is_media_input = is_video_audio_or_url(args.input, "")

    # Check for transcription-related arguments
    transcription_args = []
    if hasattr(args, "transcribe_model") and args.transcribe_model != "large-v3":
        transcription_args.append("--transcribe-model")
    if hasattr(args, "multi_language") and args.multi_language:
        transcription_args.append("--multi-language")
    if hasattr(args, "transcribe_lang") and args.transcribe_lang:
        transcription_args.append("--transcribe-lang")

    # If transcription arguments are used with non-media input, show error
    if transcription_args and not is_media_input:
        print(
            f"Error: The following options can only be used with video, audio, or URL inputs: {', '.join(transcription_args)}"
        )
        print(f"Your input '{args.input}' is not a video, audio file, or URL.")
        sys.exit(1)


def handle_rewrite_command(args):
    """Handle the rewrite subcommand"""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting rewrite command")
        logger.debug(f"Input: {args.input}")

    # Validate transcription arguments
    validate_transcription_args(args)

    # Load config if provided
    config = load_config(args.config)

    # Prepare parameters
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 130000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "lang": args.lang or config.get("lang", "Chinese"),
        "subcommand": "rewrite",
        "transcribe_model": args.transcribe_model
        or config.get("transcribe_model", "large-v3"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "verbose": args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != "verbose":
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params["timestamp"] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(
                f"Processing timestamp segment: {args.start_time} - {args.end_time}"
            )
    else:
        params["timestamp"] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input, args.input if is_url else "", **params
    )

    if result[0] and not result[0].startswith("Error"):
        print("Rewrite completed successfully!")
        print("Output file:", result[1] if result[1] else "Text output only")
        if result[1]:
            print("You can find the rewritten text in:", result[1])
    else:
        print("Error:", result[0])


def handle_translate_command(args):
    """Handle the translate subcommand"""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting translate command")
        logger.debug(f"Input: {args.input}")

    # Validate transcription arguments
    validate_transcription_args(args)

    # Load config if provided
    config = load_config(args.config)

    # Prepare parameters
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 130000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "lang": args.lang or config.get("lang", "Chinese"),
        "subcommand": "translate",
        "transcribe_model": args.transcribe_model
        or config.get("transcribe_model", "large-v3"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "verbose": args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != "verbose":
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params["timestamp"] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(
                f"Processing timestamp segment: {args.start_time} - {args.end_time}"
            )
    else:
        params["timestamp"] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input, args.input if is_url else "", **params
    )

    if result[0] and not result[0].startswith("Error"):
        print("Translation completed successfully!")
        print("Output file:", result[1] if result[1] else "Text output only")
        if result[1]:
            print("You can find the translated text in:", result[1])
    else:
        print("Error:", result[0])


def handle_academic_command(args):
    """Handle the academic subcommand"""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting academic command")
        logger.debug(f"Input: {args.input}")

    # Validate transcription arguments
    validate_transcription_args(args)

    # Load config if provided
    config = load_config(args.config)

    # Prepare parameters
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 130000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "lang": args.lang or config.get("lang", "English"),
        "subcommand": "academic",
        "transcribe_model": args.transcribe_model
        or config.get("transcribe_model", "large-v3"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "verbose": args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != "verbose":
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params["timestamp"] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(
                f"Processing timestamp segment: {args.start_time} - {args.end_time}"
            )
    else:
        params["timestamp"] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input, args.input if is_url else "", **params
    )

    if result[0] and not result[0].startswith("Error"):
        print("Academic rewriting completed successfully!")
        print("Output file:", result[1] if result[1] else "Text output only")
        if result[1]:
            print("You can find the academic text in:", result[1])
    else:
        print("Error:", result[0])


def add_global_args(subparser):
    """Add common arguments to subparsers"""
    subparser.add_argument("input", help="Path to input file or URL")
    subparser.add_argument(
        "--config", "-c", default="", help="Path to YAML configuration file"
    )
    subparser.add_argument(
        "--output-dir", "-o", default="", help="Output directory (optional)"
    )
    subparser.add_argument("--llm", default="", help="LLM model identifier (optional)")
    subparser.add_argument("--lang", "-l", default="", help="Target language")
    subparser.add_argument(
        "--chunk-length",
        "-cl",
        type=int,
        default=20,
        help="Number of sentences per paragraph (default: 20)",
    )
    subparser.add_argument(
        "--max-tokens",
        "-mt",
        type=int,
        default=130000,
        help="Maximum tokens for LLM output (default: 130000)",
    )
    subparser.add_argument(
        "--timeout",
        "-to",
        type=int,
        default=3600,
        help="LLM request timeout in seconds (default: 3600)",
    )
    subparser.add_argument(
        "--temperature",
        "-tm",
        type=float,
        default=0.1,
        help="LLM temperature parameter (default: 0.1)",
    )
    # Transcription-related arguments (only for video/audio/URL inputs)
    subparser.add_argument(
        "--transcribe-model",
        "-tsm",
        default="large-v3",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "turbo",
        ],
        help="Whisper model size for transcription (default: large-v3)",
    )
    subparser.add_argument(
        "--multi-language",
        "-m",
        action="store_true",
        help="Enable multi-language processing (video/audio/URL only)",
    )
    subparser.add_argument(
        "--transcribe-lang",
        "-s",
        default="",
        help="Transcribe language (video/audio/URL only)",
    )
    subparser.add_argument(
        "--output-wav",
        "-ow",
        default="",
        help="Filename for saving the segmented WAV (optional)",
    )
    subparser.add_argument(
        "--start-time",
        "-st",
        default="",
        help="Start time for extraction (format: HH:MM:SS)",
    )
    subparser.add_argument(
        "--end-time",
        "-et",
        default="",
        help="End time for extraction (format: HH:MM:SS)",
    )
    subparser.add_argument(
        "--cite-timestamps",
        action="store_true",
        default=False,
        help="Include timestamps as headers in markdown output for traceability",
    )
    subparser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output showing processing details",
    )


def is_markdown_file(file_path):
    """Check if file is markdown format"""
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in [".md", ".markdown"]


# ============================================================================
# PPT WORKFLOW UTILITY FUNCTIONS
# ============================================================================

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    import base64
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        return None


def extract_and_deduplicate_frames(video_path, start_time, end_time, frame_interval,
                                   ssim_threshold, hist_threshold, output_dir,
                                   logger, verbose):
    """Extract and deduplicate frames from video."""
    from wenbi.video_slides import extract_all_frames_from_video, deduplicate_slides_by_image

    if verbose:
        logger.debug(f"Extracting frames with interval {frame_interval}s...")

    all_frames = extract_all_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        start_time=start_time or "00:00:00",
        end_time=end_time or None,
        frame_interval=frame_interval,
        logger=logger,
        verbose=verbose
    )

    if not all_frames:
        print(f"Error: No frames extracted from {video_path}")
        sys.exit(1)

    if verbose:
        logger.debug(f"Extracted {len(all_frames)} frames")

    if verbose:
        logger.debug("Deduplicating frames...")

    deduplicated = deduplicate_slides_by_image(
        all_frames,
        ssim_threshold=ssim_threshold,
        hist_threshold=hist_threshold,
        logger=logger,
        verbose=verbose
    )

    removed = len(all_frames) - len(deduplicated)
    if verbose:
        logger.debug(f"Removed {removed} duplicates, {len(deduplicated)} unique frames remain")

    if not deduplicated:
        print("Error: No frames after deduplication")
        sys.exit(1)

    return deduplicated


def run_marker_pdf_on_image(image_path, output_dir, verbose=False, logger=None):
    """
    Run marker_single shell command on image file.
    Returns dict with text, base64_images, and success status.
    """
    import subprocess
    import base64
    import shutil

    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(image_path):
        return {
            "text": "",
            "base64_images": {},
            "success": False,
            "error": f"Image not found: {image_path}"
        }

    # Create temp output directory for marker
    temp_marker_dir = os.path.join(output_dir, f"marker_temp_{os.getpid()}")
    os.makedirs(temp_marker_dir, exist_ok=True)

    try:
        # Run marker_single command
        if verbose:
            logger.debug(f"Running marker_single on: {os.path.basename(image_path)}")

        cmd = [
            "marker_single",
            image_path,
            "--output_format", "markdown",
            "--output_dir", temp_marker_dir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            error_msg = result.stderr or "marker_single failed"
            if verbose:
                logger.warning(f"marker_single error: {error_msg}")
            return {
                "text": "",
                "base64_images": {},
                "success": False,
                "error": error_msg
            }

        # Read generated markdown file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        md_file = os.path.join(temp_marker_dir, f"{base_name}.md")

        if not os.path.exists(md_file):
            if verbose:
                logger.warning(f"Marker output file not found: {md_file}")
            return {
                "text": "",
                "base64_images": {},
                "success": False,
                "error": "Marker did not generate markdown output"
            }

        # Read markdown text
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # Look for extracted images in temp directory
        base64_images = {}
        for filename in os.listdir(temp_marker_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
                img_path = os.path.join(temp_marker_dir, filename)
                try:
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    b64_str = base64.b64encode(img_data).decode("utf-8")
                    base64_images[filename] = b64_str

                    if verbose:
                        logger.debug(f"Encoded image to base64: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to encode image {filename}: {e}")

        if verbose:
            logger.debug(f"Marker OCR: {len(base64_images)} images extracted")

        return {
            "text": markdown_text,
            "base64_images": base64_images,
            "success": True
        }

    except subprocess.TimeoutExpired:
        return {
            "text": "",
            "base64_images": {},
            "success": False,
            "error": "marker_single timeout (>5 min)"
        }

    except Exception as e:
        logger.error(f"Error running marker_single: {e}")
        return {
            "text": "",
            "base64_images": {},
            "success": False,
            "error": str(e)
        }

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_marker_dir)
        except:
            pass


def embed_frames_as_base64(frames_with_timestamps, output_dir, base_name, logger, verbose):
    """Generate markdown with frames embedded as base64."""
    content = []

    for frame_dict in frames_with_timestamps:
        timestamp = frame_dict["timestamp"]
        frame_path = frame_dict["frame_path"]

        try:
            b64 = image_to_base64(frame_path)
            if not b64:
                logger.warning(f"Failed to encode {timestamp}")
                continue

            section = f"\n### **{timestamp}**\n"
            section += f'<img src="data:image/png;base64,{b64}" />\n'

            content.append(section)

            if verbose:
                logger.debug(f"Encoded frame {timestamp} to base64")

        except Exception as e:
            logger.warning(f"Failed to encode {timestamp}: {e}")

    md_path = os.path.join(output_dir, f"{base_name}_slides.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(content))

    if verbose:
        logger.debug(f"Generated base64 markdown: {md_path}")

    return md_path


def clean_combined_markdown(combine_md_path, output_dir, base_name, logger, verbose):
    """
    Remove timestamps and image file references, keep base64 images.
    """
    if verbose:
        logger.debug("Cleaning combined markdown...")

    with open(combine_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove timestamp headers (### **HH:MM:SS**)
    import re
    content = re.sub(r'\n### \*\*\d{2}:\d{2}:\d{2}\*\*\n', '\n', content)

    # Remove image file references ![slide](/path/to/image.png)
    # but keep <img src="data:image/png;base64,..."/>
    content = re.sub(r'!\[.*?\]\([^)]*\.png\)', '', content)

    clean_path = os.path.join(output_dir, f"{base_name}_combine_clean.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(content)

    if verbose:
        logger.debug(f"Cleaned markdown: {clean_path}")

    return clean_path


def handle_ppt_command(args):
    """Handle ppt subcommand - main entry point for all 3 PPT workflow methods"""
    import subprocess

    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting PPT workflow")
        logger.debug(f"Input: {args.input}")

    # Validate input
    if not args.input:
        print("Error: Video file or URL is required")
        sys.exit(1)

    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    is_url = args.input.startswith(("http://", "https://", "www."))

    # Input validation for local files
    if not is_url:
        from wenbi.video_slides import validate_video_input

        if not validate_video_input(args.input, logger, args.verbose):
            print(f"Error: Invalid video file: {args.input}")
            sys.exit(1)

    video_path = args.input

    # Download video if URL
    if is_url:
        if args.verbose:
            logger.debug("Downloading video from URL...")

        from wenbi.utils import download_video

        try:
            download_result = download_video(
                args.input, output_dir=output_dir, verbose=args.verbose
            )
            if download_result:
                video_path = download_result
            else:
                print("Error: Failed to download video from URL")
                sys.exit(1)
        except Exception as e:
            print(f"Error downloading video: {e}")
            sys.exit(1)

        if args.verbose:
            logger.debug(f"Video downloaded to: {video_path}")

    # COMMON FOUNDATION: Extract frames + deduplicate
    if args.verbose:
        logger.debug("Step 1: Extracting and deduplicating frames...")

    deduplicated_frames = extract_and_deduplicate_frames(
        video_path=video_path,
        start_time=args.start_time,
        end_time=args.end_time,
        frame_interval=args.frame_interval,
        ssim_threshold=args.ssim_threshold,
        hist_threshold=args.hist_threshold,
        output_dir=output_dir,
        logger=logger,
        verbose=args.verbose
    )
    
    if args.verbose:
        logger.debug(f"Frame extraction complete: {len(deduplicated_frames)} deduplicated frames extracted")
        for i, frame in enumerate(deduplicated_frames[:5]):  # Show first 5
            logger.debug(f"  Frame {i+1}: {frame['timestamp']} -> {frame['frame_path']}")
        if len(deduplicated_frames) > 5:
            logger.debug(f"  ... and {len(deduplicated_frames) - 5} more frames")

    # ROUTE TO METHOD
    try:
        if args.ppt:
            # TYPE 3: PPT METHOD
            if args.verbose:
                logger.debug("Routing to TYPE 3: PPT Method")
            from wenbi.ppt_slide import execute_ppt_method

            combine_md, combine_clean_md = execute_ppt_method(
                video_path=video_path,
                deduplicated_frames=deduplicated_frames,
                ppt_path=args.ppt,
                output_dir=output_dir,
                no_ocr=args.no_ocr,
                no_clean=args.no_clean,
                base_name=base_name,
                cite_timestamps=args.cite_timestamps,
                llm=args.llm,
                chunk_length=args.chunk_length,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                temperature=args.temperature,
                lang=args.lang,
                transcribe_model=args.transcribe_model,
                multi_language=args.multi_language,
                transcribe_lang=args.transcribe_lang,
                logger=logger,
                verbose=args.verbose,
                ssim_threshold=args.ssim_threshold
            )

        elif args.cropped_slide is not None:
            # TYPE 2: CROPPED-SLIDE METHOD
            if args.verbose:
                logger.debug("Routing to TYPE 2: CROPPED-SLIDE Method")
            from wenbi.cropped_slide import execute_cropped_slide_method

            combine_md, combine_clean_md = execute_cropped_slide_method(
                video_path=video_path,
                deduplicated_frames=deduplicated_frames,
                roi_string=args.cropped_slide if args.cropped_slide != "auto" else None,
                output_dir=output_dir,
                no_ocr=args.no_ocr,
                no_clean=args.no_clean,
                base_name=base_name,
                cite_timestamps=args.cite_timestamps,
                llm=args.llm,
                chunk_length=args.chunk_length,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                temperature=args.temperature,
                lang=args.lang,
                transcribe_model=args.transcribe_model,
                multi_language=args.multi_language,
                transcribe_lang=args.transcribe_lang,
                logger=logger,
                verbose=args.verbose
            )

        else:
            # TYPE 1: FRAME METHOD (existing, refactored)
            if args.verbose:
                logger.debug("Routing to TYPE 1: FRAME Method")

            if args.no_ocr:
                if args.verbose:
                    logger.debug("--no-ocr: Embedding frames as base64...")

                slides_md = embed_frames_as_base64(
                    deduplicated_frames,
                    output_dir,
                    base_name,
                    logger,
                    args.verbose
                )
            else:
                if args.verbose:
                    logger.debug("Step 2: Running OCR on frames...")

                # OCR each frame
                markdown_sections = []

                for idx, frame_dict in enumerate(deduplicated_frames, 1):
                    timestamp = frame_dict["timestamp"]
                    frame_path = frame_dict["frame_path"]

                    if args.verbose:
                        logger.debug(f"OCR frame {idx}/{len(deduplicated_frames)}: {timestamp}")

                    ocr_result = run_marker_pdf_on_image(
                        frame_path, output_dir, args.verbose, logger
                    )

                    section = f"\n### **{timestamp}**\n"

                    if ocr_result["success"]:
                        section += ocr_result["text"]

                        # Add base64 images if any
                        for filename, b64 in ocr_result["base64_images"].items():
                            section += f'\n<img src="data:image/png;base64,{b64}" />\n'
                    else:
                        # OCR failed, embed as base64
                        if args.verbose:
                            logger.warning(f"OCR failed for {timestamp}, using base64")

                        b64 = image_to_base64(frame_path)
                        if b64:
                            section += f'<img src="data:image/png;base64,{b64}" />\n'

                    markdown_sections.append(section)

                slides_md = os.path.join(output_dir, f"{base_name}_slides.md")
                with open(slides_md, "w", encoding="utf-8") as f:
                    f.write("".join(markdown_sections))

                if args.verbose:
                    logger.debug(f"OCR completed: {slides_md}")

            # Step 3: Rewrite audio
            if args.verbose:
                logger.debug("Step 3: Processing audio...")

            params = {
                "output_dir": output_dir,
                "llm": args.llm,
                "chunk_length": args.chunk_length,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
                "temperature": args.temperature,
                "lang": args.lang,
                "transcribe_model": args.transcribe_model,
                "multi_language": args.multi_language,
                "transcribe_lang": args.transcribe_lang,
                "cite_timestamps": args.cite_timestamps,
                "verbose": args.verbose,
                "subcommand": "rewrite"
            }

            result = process_input(
                file_path=video_path,
                url="",
                **params
            )

            audio_markdown = result[0]
            if args.verbose:
                logger.debug(f"Audio processing completed")

            # Step 4: Combine
            if args.verbose:
                logger.debug("Step 4: Combining slide and audio markdown...")

            with open(slides_md, "r", encoding="utf-8") as f:
                slides_content = f.read()

            combined_markdown = combine_speech_and_slides(
                speech_markdown=audio_markdown,
                slides_markdown=slides_content,
                logger=logger,
                verbose=args.verbose
            )

            combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
            with open(combine_md, "w", encoding="utf-8") as f:
                f.write(combined_markdown)

            if args.verbose:
                logger.debug(f"Combined markdown: {combine_md}")

            # Step 5: Clean (if not --no-clean)
            if args.no_clean:
                combine_clean_md = None
                if args.verbose:
                    logger.debug("--no-clean: Skipping clean phase")
            else:
                combine_clean_md = clean_combined_markdown(
                    combine_md, output_dir, base_name, logger, args.verbose
                )

        # Print results
        print("✓ PPT processing completed!")
        print(f"  Combined: {combine_md}")
        if combine_clean_md:
            print(f"  Cleaned: {combine_clean_md}")

    except Exception as e:
        print(f"Error during PPT processing: {e}")
        if args.verbose:
            logger.exception("Detailed error trace:")
        sys.exit(1)


def _old_handle_ppt_command_backup(args):
    """Old implementation kept for reference during migration"""
    import subprocess
    import sys

    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting PPT workflow")
        logger.debug(f"Input: {args.input}")

    # Validate input
    if not args.input:
        print("Error: Video file or URL is required")
        sys.exit(1)

    output_dir = args.output_dir or os.getcwd()
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    is_url = args.input.startswith(("http://", "https://", "www."))

    # Input validation
    if not is_url:
        from wenbi.video_slides import (
            validate_video_input,
            extract_all_frames_from_video,
            deduplicate_slides_by_image,
            ocr_slide_image,
        )

        if not validate_video_input(args.input, logger, args.verbose):
            print(f"Error: Invalid video file: {args.input}")
            sys.exit(1)

    video_path = args.input

    try:
        # Step 1: Download video if URL input
        if is_url:
            if args.verbose:
                logger.debug("Step 1: Downloading video from URL")

            from wenbi.utils import download_audio

            try:
                download_result = download_audio(
                    args.input, output_dir=output_dir, verbose=args.verbose
                )
                if download_result:
                    video_path = download_result
                else:
                    print("Error: Failed to download video from URL")
                    sys.exit(1)
            except Exception as e:
                print(f"Error downloading video: {e}")
                sys.exit(1)

            if args.verbose:
                logger.debug(f"Video downloaded to: {video_path}")

        # Configure time range for slide extraction
        slides_start = getattr(args, "start_time", "00:00:10") or "00:00:10"
        slides_end = getattr(args, "end_time", "01:00:00") or "01:00:00"
        video_for_slides = video_path

        if args.verbose:
            logger.debug("Starting PPT workflow with frame extraction and deduplication")
        
        # Phase 1: Extract ALL frames from video at regular intervals
        from wenbi.video_slides import (
            extract_all_frames_from_video,
            detect_slide_rectangles_in_frames,
            crop_and_save_slides,
            deduplicate_slides_by_image,
        )
        
        all_frames = extract_all_frames_from_video(
            video_for_slides,
            output_dir=output_dir,
            start_time=slides_start,
            end_time=slides_end,
            frame_interval=getattr(args, "frame_interval", 60),
            logger=logger,
            verbose=args.verbose,
        )
        
        if not all_frames:
            print("⚠ Warning: No frames extracted from video. Proceeding with speech transcription only.")
            slides_data = []
        else:
            print(f"✓ Phase 1: Extracted {len(all_frames)} frames")
            
            # Phase 2: Deduplicate frames early (SSIM + histogram)
            print(f"→ Phase 2: Deduplicating frames...")
            unique_frames = deduplicate_slides_by_image(
                all_frames,
                ssim_threshold=getattr(args, "ssim_threshold", 0.98),
                hist_threshold=getattr(args, "hist_threshold", 0.15),
                logger=logger,
                verbose=args.verbose,
            )
            duplicates_removed = len(all_frames) - len(unique_frames)
            print(f"✓ Phase 2: Deduplication - Removed {duplicates_removed} duplicates, {len(unique_frames)} unique frames remain")
            
            # Phase 3: OCR on full frames
            print(f"→ Phase 3: Running OCR on {len(unique_frames)} unique frames...")
            slides_data = []
            
            for frame_idx, frame_data in enumerate(unique_frames, 1):
                frame_path = frame_data.get('frame_path')
                timestamp = frame_data.get('timestamp', '')
                
                if not frame_path or not os.path.exists(frame_path):
                    continue
                
                try:
                    ocr_result = ocr_slide_image(
                        frame_path, output_dir=output_dir, logger=logger, verbose=args.verbose
                    )
                    
                    slide_content = ocr_result.get("text", "").strip()
                    image_data = None
                    
                    # Embed image if OCR failed to extract text
                    if not slide_content or ocr_result.get("failed", False):
                        try:
                            with open(frame_path, "rb") as img_file:
                                image_data = img_file.read()
                            if args.verbose:
                                logger.debug(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: OCR failed, embedded image ({len(image_data)} bytes)")
                            else:
                                print(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: No text extracted (image embedded)")
                        except Exception as e:
                            if args.verbose:
                                logger.warning(f"Failed to read image for embedding at {timestamp}: {e}")
                            print(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: Error embedding image")
                    else:
                        text_preview = slide_content[:50].replace('\n', ' ') if slide_content else "(empty)"
                        if args.verbose:
                            logger.debug(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: OCR succeeded ({len(slide_content)} chars)")
                        else:
                            print(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: {text_preview}...")
                    
                    slides_data.append({
                        "timestamp": timestamp,
                        "content": slide_content,
                        "image_data": image_data,
                        "frame_path": frame_path,
                    })
                        
                except Exception as e:
                    if args.verbose:
                        logger.warning(f"Error during OCR for frame at {timestamp}: {e}")
                    else:
                        print(f"  Frame {frame_idx}/{len(unique_frames)} at {timestamp}: OCR error - {str(e)[:50]}")
                    continue
            
            print(f"✓ Phase 3: OCR completed on {len(slides_data)} frames")

        if not slides_data:
            print("⚠ Warning: No slides extracted from video. Proceeding with speech transcription only.")
            slides_file = None
        else:
            if args.verbose:
                logger.debug(f"Extracted {len(slides_data)} slides. Saving to markdown.")
            slides_file = save_slides_to_markdown(
                slides_data, output_dir, base_name, logger, args.verbose
            )

        # Step 3 (New Step 2): Run RW command to generate speech.md
        if args.verbose:
            logger.debug(
                "Step 3: Running rewrite subcommand to generate speech transcription (_rewritten.md)"
            )

        script_path = os.path.join(os.path.dirname(__file__), "cli.py")
        rw_cmd = [sys.executable, script_path, "rw", args.input, "--cite-timestamps"]

        if args.output_dir:
            rw_cmd.extend(["--output-dir", args.output_dir])
        if args.llm:
            rw_cmd.extend(["--llm", args.llm])
        if args.lang:
            rw_cmd.extend(["--lang", args.lang])
        if args.start_time:
            rw_cmd.extend(["--start-time", args.start_time])
        # Note: --end-time only applies to slides extraction, not speech transcription
        # RW subcommand processes the entire video/audio for speech
        if args.verbose:
            rw_cmd.append("--verbose")

        try:
            if args.verbose:
                logger.debug(f"Running rewrite command: {' '.join(rw_cmd)}")

            process = subprocess.Popen(
                rw_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                cwd=os.getcwd(),
            )

            if args.verbose and process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line.strip())
            process.wait()

            if process.returncode != 0:
                print(
                    f"Error: Rewrite command failed with return code {process.returncode}"
                )
                sys.exit(1)

            speech_file_pattern = f"{base_name}_rewritten.md"
            speech_file = os.path.join(output_dir, speech_file_pattern)

            if not os.path.exists(speech_file):
                # Fallback for different naming conventions if any
                for file in os.listdir(output_dir):
                    if file.startswith(base_name) and file.endswith("_rewritten.md"):
                        speech_file = os.path.join(output_dir, file)
                        break

            if not os.path.exists(speech_file):
                print(
                    f"Error: Could not find generated speech file: {speech_file_pattern}"
                )
                sys.exit(1)

            if args.verbose:
                logger.debug(f"Speech transcription successful. Output: {speech_file}")

        except Exception as e:
            print(f"Error running rewrite command: {e}")
            sys.exit(1)

        # Step 4 (New Step 3): Insert slides into speech.md
        if slides_file:
            if args.verbose:
                logger.debug("Step 4: Inserting slides into speech content")

            combined_content = insert_slides_into_speech(
                speech_file, slides_file, logger=logger, verbose=args.verbose
            )

            combined_file = os.path.join(output_dir, f"{base_name}_combined.md")
            with open(combined_file, "w", encoding="utf-8") as f:
                f.write(combined_content)

            print("✓ PPT processing completed successfully!")
            print(f"  Slides file: {slides_file}")
            print(f"  Speech file: {speech_file}")
            print(f"  Combined file: {combined_file}")
        else:
            print("✓ PPT processing completed (speech transcription only).")
            print(f"  Speech file: {speech_file}")

    except Exception as e:
        print(f"Error during PPT processing: {e}")
        if args.verbose:
            logger.exception("Detailed error trace:")
        sys.exit(1)


def save_slides_to_markdown(
    slides_data, output_dir, base_name, logger=None, verbose=False
):
    """Saves extracted slide data to markdown with Picture_X.jpeg images embedded as base64."""
    import base64
    import re
    
    slides_file = os.path.join(output_dir, f"{base_name}_slide.md")
    if verbose and logger:
        logger.debug(f"Saving slides data to {slides_file}")

    with open(slides_file, "w", encoding="utf-8") as f:
        f.write("# Extracted Slides\n\n")
        
        for idx, slide in enumerate(slides_data, 1):
            f.write(f"## Slide at {slide['timestamp']}\n\n")
            
            # Get OCR content
            ocr_content = slide.get("content", "").strip()
            image_data = slide.get("image_data")
            
            if ocr_content:
                # OCR succeeded, write the text
                f.write(f"{ocr_content}\n\n")
            
            if image_data:
                # Embed image if OCR failed or image provided
                try:
                    # Encode image to base64
                    img_b64 = base64.b64encode(image_data).decode()
                    
                    # Determine MIME type (default to PNG since frames are PNG)
                    mime_type = "image/png"
                    
                    # Embed as base64 image
                    base64_uri = f"data:{mime_type};base64,{img_b64}"
                    f.write(f"![Slide {idx}]({base64_uri})\n\n")
                    
                    if verbose and logger:
                        logger.debug(f"Embedded image for slide at {slide['timestamp']}")
                
                except Exception as e:
                    if verbose and logger:
                        logger.warning(f"Could not embed image for slide at {slide['timestamp']}: {e}")
                    f.write(f"> Image embedding failed: {e}\n\n")
            
            # If no content and no image, note it
            if not ocr_content and not image_data:
                f.write("> No content available for this slide.\n\n")

    if verbose and logger:
        logger.debug(f"Finished saving slides to markdown with embedded Picture images.")

    return slides_file


def extract_slides_from_video(
    video_path,
    output_dir,
    start_time="00:00:10",
    end_time=None,
    roi=None,
    slide_period="00:01:00",
    scenedetect_threshold=10.0,
    max_slides=20,
    deduplicate=True,
    similarity_threshold=0.85,
    dedup_method="both",
    ssim_threshold=0.98,
    hist_threshold=0.15,
    each_roi=False,
    frame_interval=2,
    clip_trim=True,
    logger=None,
    verbose=False,
):
    """Extract slides from video using computer vision and OCR with marker-pdf

    New parameters:
      - auto_roi: if True, perform automatic ROI detection (default True).
      - roi: optional manual ROI tuple or None. If provided, overrides auto ROI detection.
      - slide_period: minimum slide duration (HH:MM:SS) used to tune detection (default "00:02:00").
      - scenedetect_threshold: threshold value for content detector (default 35.0).

    Note: these options are passed into the slide detection flow so downstream
    detectors can use them (e.g., PySceneDetect ContentDetector).
    """
    import tempfile

    import cv2

    from wenbi.video_slides import (
        detect_slide_changes,
        detect_slide_roi,
        detect_video_resolution,
        extract_frame_at_timestamp,
        # manual_roi_override may be used if manual_roi is True
        manual_roi_override,
        ocr_slide_image,
    )

    if logger is None:
        logger = logging.getLogger(__name__)

    slides_data = []
    temp_dir = tempfile.mkdtemp(prefix="wenbi_slides_")

    try:
        # Compute minimum scene duration (seconds) from slide_period
        min_scene_seconds = parse_time_to_seconds(slide_period)
        
        if verbose:
            logger.debug(f"Extracting slides from {video_path}")
            logger.debug(f"Time range: {start_time} to {end_time or 'end of video'}")
            logger.debug(
                f"Slide detection tuning: slide_period={slide_period}, scenedetect_threshold={scenedetect_threshold}, min_scene_seconds={min_scene_seconds}"
            )

        # Step 1: Detect ROI (Region of Interest) for slides
        # Priority: explicit ROI CLI (--roi) > automatic ROI detection > fallback default ROI
        if roi:
            try:
                # Support both pixel coordinates and percentage coordinates.
                # Format: x0,y0,x1,y1 where each value may be an integer (pixels)
                # or a float between 0.0 and 1.0 (percent of width/height).
                parts_raw = [p.strip() for p in str(roi).split(",")]
                if len(parts_raw) == 4:
                    width, height = detect_video_resolution(video_path)
                    coords = []
                    for i, p in enumerate(parts_raw):
                        # empty values invalid
                        if p == "":
                            raise ValueError("Empty ROI value")
                        # detect float-like
                        if "." in p:
                            valf = float(p)
                            # treat 0.0-1.0 as percentage
                            if 0.0 <= valf <= 1.0:
                                if i % 2 == 0:
                                    coords.append(int(valf * width))
                                else:
                                    coords.append(int(valf * height))
                            else:
                                # treat as absolute pixel value if > 1
                                coords.append(int(valf))
                        else:
                            coords.append(int(p))
                    if len(coords) == 4:
                        roi_coords = (coords[0], coords[1], coords[2], coords[3])
                    else:
                        raise ValueError("Parsed ROI does not contain 4 values")
                else:
                    raise ValueError("ROI must be in format 'x0,y0,x1,y1'")
            except Exception as e:
                print(f"Error parsing --roi: {e}")
                # Fallback to automatic detection if ROI parse fails
                roi_coords = detect_slide_roi(video_path, logger, verbose)
        else:
            # Automatic ROI detection (recommended)
            roi_coords = detect_slide_roi(video_path, logger, verbose)

        if verbose:
            logger.debug(f"Using ROI coordinates: {roi_coords}")

        # Step 2: Detect slide transitions
        print(f"Debug extract_slides_from_video: About to detect slide changes")
        print(
            f"Debug extract_slides_from_video: start_time={start_time}, end_time={end_time}"
        )
        slide_timestamps = detect_slide_changes(
            video_path,
            roi_coords,
            start_time=start_time,
            end_time=end_time,
            scenedetect_threshold=scenedetect_threshold,
            min_scene_seconds=min_scene_seconds,
            logger=logger,
            verbose=verbose,
        )
        print(
            f"Debug extract_slides_from_video: Slide detection completed, found {len(slide_timestamps)} transitions"
        )

        if verbose:
            logger.debug(f"Detected {len(slide_timestamps)} slide transitions")

        if not slide_timestamps:
            print(
                f"Debug extract_slides_from_video: No slide transitions detected, returning empty"
            )
            return []

        # Step 3: Extract frames at each slide transition
        print(
            f"Debug extract_slides_from_video: Starting to extract frames for {len(slide_timestamps)} slides"
        )
        import sys
        import shutil
        import re

        # ensure output dir exists for saving images
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass

        # use max_slides parameter passed to the function
        max_slides = int(max_slides) if max_slides else None

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Phase 1: Extract all frames first
        all_frames_data = []
        for i, slide_info in enumerate(slide_timestamps):
            # enforce max slides
            if max_slides and len(all_frames_data) >= max_slides:
                print(
                    f"Debug extract_slides_from_video: Reached max_slides {max_slides}, stopping further extraction"
                )
                break

            timestamp = slide_info["start_time"]
            print(
                f"Debug extract_slides_from_video: Extracting frame {i + 1}/{len(slide_timestamps)} at {timestamp}"
            )
            sys.stdout.flush()
            if verbose:
                logger.debug(
                    f"Extracting frame {i + 1}/{len(slide_timestamps)} at timestamp {timestamp}"
                )

            try:
                frame_path = extract_frame_at_timestamp(
                    video_path,
                    timestamp,
                    roi_coords=roi_coords,
                    output_dir=temp_dir,
                    logger=logger,
                    verbose=verbose,
                )
                
                # Save a copy of the cropped slide image into the output_dir
                try:
                    clean_ts = re.sub(r"[:.]", "_", timestamp)
                    saved_name = f"{base_name}_slide_{i + 1}_{clean_ts}.png"
                    saved_path = os.path.join(output_dir, saved_name)
                    shutil.copyfile(frame_path, saved_path)
                    if verbose and logger:
                        logger.debug(f"Saved cropped slide image to: {saved_path}")
                except Exception as e:
                    saved_path = frame_path
                    print(
                        f"Debug extract_slides_from_video: Warning saving cropped image to output_dir: {e}"
                    )

                all_frames_data.append(
                    {
                        "timestamp": timestamp,
                        "frame_path": saved_path,
                        "slide_info": slide_info,
                    }
                )
                
            except Exception as e:
                print(
                    f"Debug extract_slides_from_video: Error extracting frame {i + 1}: {e}"
                )
                sys.stdout.flush()

        print(
            f"Debug extract_slides_from_video: Completed frame extraction, extracted {len(all_frames_data)} frames"
        )

        # Phase 2: Apply image-based deduplication if enabled
        if deduplicate and dedup_method in ["image", "both"]:
            from wenbi.video_slides import deduplicate_slides_by_image
            
            if verbose and logger:
                logger.debug(
                    f"Applying image-based deduplication with ssim_threshold={ssim_threshold}, "
                    f"hist_threshold={hist_threshold}"
                )
            
            original_count = len(all_frames_data)
            unique_frames = deduplicate_slides_by_image(
                all_frames_data,
                ssim_threshold=ssim_threshold,
                hist_threshold=hist_threshold,
                logger=logger,
                verbose=verbose
            )
            
            if verbose and logger:
                logger.debug(
                    f"Image deduplication: {original_count} frames -> {len(unique_frames)} unique frames"
                )
            
            print(
                f"Debug extract_slides_from_video: After image deduplication, {len(unique_frames)} unique frames remain"
            )
        else:
            unique_frames = all_frames_data

        # Phase 3: OCR only on unique frames
        print(
            f"Debug extract_slides_from_video: Starting OCR on {len(unique_frames)} unique frames"
        )
        
        for i, frame_data in enumerate(unique_frames):
            timestamp = frame_data["timestamp"]
            print(
                f"Debug extract_slides_from_video: OCR frame {i + 1}/{len(unique_frames)} at {timestamp}"
            )
            sys.stdout.flush()
            if verbose:
                logger.debug(
                    f"OCR processing frame {i + 1}/{len(unique_frames)} at timestamp {timestamp}"
                )

            try:
                # OCR using the saved image path
                ocr_input_path = frame_data["frame_path"]
                ocr_result = ocr_slide_image(
                    ocr_input_path, output_dir=temp_dir, logger=logger, verbose=verbose
                )
                print(f"Debug extract_slides_from_video: OCR completed for {timestamp}")
                sys.stdout.flush()
                slide_content = ocr_result.get("text", "")

                image_data = None
                # Embed image if OCR text is short/unreliable or OCR failed
                if (
                    not slide_content
                    or len(slide_content.strip()) < 15
                    or ocr_result.get("failed", False)
                ):
                    if verbose:
                        logger.debug(
                            "OCR content is short or failed, embedding image as fallback."
                        )
                    try:
                        with open(ocr_input_path, "rb") as img_file:
                            image_data = img_file.read()
                    except Exception as e:
                        print(
                            f"Debug extract_slides_from_video: Failed to read image for embedding: {e}"
                        )

                slides_data.append(
                    {
                        "timestamp": timestamp,
                        "content": slide_content,
                        "image_data": image_data,
                        "frame_path": frame_data["frame_path"],
                    }
                )
            except Exception as e:
                print(
                    f"Debug extract_slides_from_video: Error during OCR for frame {i + 1}: {e}"
                )
                sys.stdout.flush()

        print(
            f"Debug extract_slides_from_video: Completed OCR, processed {len(slides_data)} slides"
        )
        
        # Apply text-based deduplication if enabled (and not already done with image-only)
        if deduplicate and slides_data and dedup_method in ["text", "both"]:
            from wenbi.video_slides import deduplicate_slides
            
            original_count = len(slides_data)
            slides_data = deduplicate_slides(
                slides_data, 
                similarity_threshold=similarity_threshold,
                logger=logger,
                verbose=verbose
            )
            
            if verbose and logger:
                logger.debug(
                    f"Text deduplication: {original_count} slides -> {len(slides_data)} unique slides"
                )
            
            print(
                f"Debug extract_slides_from_video: After text deduplication, returning {len(slides_data)} unique slides"
            )
        elif not deduplicate:
            print(
                f"Debug extract_slides_from_video: Deduplication disabled, returning {len(slides_data)} slides"
            )
        else:
            print(
                f"Debug extract_slides_from_video: Returning {len(slides_data)} slides (image-only deduplication)"
            )
        
        return slides_data

    except Exception as e:
        print(f"Debug extract_slides_from_video: Exception occurred: {e}")
        if logger:
            logger.error(f"Error during slide extraction: {e}")
        import traceback

        traceback.print_exc()
        return []

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        if verbose and logger:
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")


def insert_slides_into_speech(speech_file, slides_file, logger=None, verbose=False):
    """Insert slides from a markdown file into speech content based on timestamp matching."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if verbose:
        logger.debug(f"Reading speech content from: {speech_file}")
        logger.debug(f"Reading slides content from: {slides_file}")

    with open(speech_file, "r", encoding="utf-8") as f:
        speech_content = f.read()

    with open(slides_file, "r", encoding="utf-8") as f:
        slides_content = f.read()

    # Parse slides data - extract everything between slide headers
    slides_data = []
    import re

    # Split by slide headers: ## Slide at HH:MM:SS
    slide_pattern = r"## Slide at ([\d:]+)"
    matches = list(re.finditer(slide_pattern, slides_content))
    
    for idx, match in enumerate(matches):
        timestamp = match.group(1)
        
        # Get content from after this slide header to the next slide header (or end of file)
        content_start = match.end()
        if idx < len(matches) - 1:
            content_end = matches[idx + 1].start()
        else:
            content_end = len(slides_content)
        
        content = slides_content[content_start:content_end].strip()
        slides_data.append({"timestamp": timestamp, "content_md": content})

    if verbose:
        logger.debug(f"Parsed {len(slides_data)} slides from the slides file.")

    # Parse speech timestamps (format: ### **HH:MM:SS - HH:MM:SS**)
    timestamp_pattern = r"###\s+\*\*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\*\*"
    speech_sections = re.split(timestamp_pattern, speech_content)

    combined_content = []
    current_slide_index = 0

    # Reconstruct speech and insert slides
    # The first element is the content before the first timestamp
    combined_content.append(speech_sections[0])

    for i in range(1, len(speech_sections), 3):
        start_time = speech_sections[i]
        end_time = speech_sections[i + 1]
        text_content = speech_sections[i + 2]

        # Convert speech section end time to seconds
        try:
            end_seconds = parse_time_to_seconds(end_time)
        except Exception as e:
            if logger:
                logger.warning(
                    f"Could not parse speech section end time {end_time}. Error: {e}"
                )
            continue

        # Find and insert slides that occur before or at this speech section's end time
        while current_slide_index < len(slides_data):
            slide_timestamp_str = slides_data[current_slide_index]["timestamp"]

            # Convert slide timestamp to seconds for comparison
            try:
                slide_seconds = parse_time_to_seconds(slide_timestamp_str)
            except Exception as e:
                if logger:
                    logger.warning(
                        f"Could not parse slide timestamp: {slide_timestamp_str}. Error: {e}"
                    )
                current_slide_index += 1
                continue

            # If slide timestamp is before or at the end of this speech section,
            # insert it before the speech section
            if slide_seconds <= end_seconds:
                slide_info = slides_data[current_slide_index]
                if verbose:
                    logger.debug(
                        f"Inserting slide at {slide_info['timestamp']} before section [{start_time}-{end_time}]"
                    )

                combined_content.append(
                    f"\n\n---\n\n## Slide at {slide_info['timestamp']}\n\n"
                )
                combined_content.append(slide_info["content_md"])
                current_slide_index += 1
            else:
                # Slide is after this section's end time, will be inserted in a later section
                break

        # Add the original speech section after slides
        combined_content.append(f"\n\n[{start_time}-{end_time}]")
        combined_content.append(text_content)

    if verbose:
        logger.debug("Finished combining speech and slides.")

    return "".join(combined_content)


def parse_time_to_seconds(time_str):
    """Convert HH:MM:SS or HH:MM:SS.ms to seconds"""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])

    total_seconds = hours * 3600 + minutes * 60 + seconds

    # Add milliseconds if present
    if len(seconds_parts) > 1:
        milliseconds = int(seconds_parts[1].ljust(3, "0")[:3])
        total_seconds += milliseconds / 1000

    return total_seconds

    # Original PPT functionality (speech + slides)
    if args.verbose:
        logger.debug(f"Speech input: {args.input}")
        logger.debug(f"Slides file: {args.slides_file}")

    # Validate inputs
    if not args.slides_file:
        print("Error: Slides file is required for ppt subcommand")
        sys.exit(1)

    if not os.path.isfile(args.slides_file):
        print(f"Error: Slides file not found: {args.slides_file}")
        sys.exit(1)

    # Check if slides_file is markdown (no format validation needed for markdown)
    slides_is_markdown = is_markdown_file(args.slides_file)

    # Validate slides file format if not markdown
    if not slides_is_markdown:
        slides_ext = os.path.splitext(args.slides_file)[1].lower()
        if slides_ext not in [".pdf", ".pptx"]:
            print(
                f"Error: Slides file must be PDF, PPTX, or markdown file, got {slides_ext}"
            )
            sys.exit(1)

    if not os.path.isfile(args.slides_file):
        print(f"Error: Slides file not found: {args.slides_file}")
        sys.exit(1)

    # Check if slides_file is markdown (no format validation needed for markdown)
    slides_is_markdown = is_markdown_file(args.slides_file)

    # Validate slides file format if not markdown
    if not slides_is_markdown:
        slides_ext = os.path.splitext(args.slides_file)[1].lower()
        if slides_ext not in [".pdf", ".pptx"]:
            print(
                f"Error: Slides file must be PDF, PPTX, or markdown file, got {slides_ext}"
            )
            sys.exit(1)

    # Validate transcription arguments
    validate_transcription_args(args)

    # Load config if provided
    config = load_config(args.config)

    # Prepare parameters for speech processing (rewrite)
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 130000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "lang": args.lang or config.get("lang", "Chinese"),
        "subcommand": "rewrite",  # Always use rewrite for speech processing
        "transcribe_model": args.transcribe_model
        or config.get("transcribe_model", "large-v3"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "verbose": args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != "verbose":
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params["timestamp"] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(
                f"Processing timestamp segment: {args.start_time} - {args.end_time}"
            )
    else:
        params["timestamp"] = None

    try:
        # Determine if speech_input is markdown
        speech_is_markdown = is_markdown_file(args.input)

        # Step 1: Process speech input
        if speech_is_markdown:
            if args.verbose:
                logger.debug(
                    "Step 1: Reading speech markdown file (skipping rewrite subcommand)"
                )

            try:
                speech_markdown, _ = read_markdown_file(
                    args.input, verbose=args.verbose
                )
                base_name = os.path.splitext(os.path.basename(args.input))[0]
                speech_file = None  # Mark as input, not generated
            except Exception as e:
                print(f"Error reading speech markdown: {e}")
                sys.exit(1)
        else:
            if args.verbose:
                logger.debug("Step 1: Processing speech input with rewrite subcommand")

            is_url = args.input.startswith(("http://", "https://", "www."))
            speech_result = process_input(
                None if is_url else args.input, args.input if is_url else "", **params
            )

            if speech_result[0] and speech_result[0].startswith("Error"):
                print(f"Error processing speech: {speech_result[0]}")
                sys.exit(1)

            speech_markdown = speech_result[0]
            speech_file = speech_result[1]  # Mark as generated
            base_name = speech_result[3] or "output"

        if args.verbose:
            logger.debug(f"Speech processing completed.")
            if speech_file:
                logger.debug(f"Output file: {speech_file}")

        # Step 2: Convert/read slides
        if args.verbose:
            logger.debug(
                f"Step 2: Processing slides {'(reading markdown)' if slides_is_markdown else '(converting to markdown)'}"
            )

        output_dir = params["output_dir"] or os.getcwd()

        if slides_is_markdown:
            try:
                slides_markdown, _ = read_markdown_file(
                    args.slides_file, verbose=args.verbose
                )
                slides_file = None  # Mark as input, not generated
            except Exception as e:
                print(f"Error reading slides markdown: {e}")
                sys.exit(1)
        else:
            slides_markdown, slides_file = convert_slides_to_markdown(
                args.slides_file,
                output_dir=output_dir,
                image_export_mode=args.image_export_mode
                or config.get("image_export_mode", "embedded"),
                verbose=args.verbose,
            )  # Mark as generated

        if args.verbose:
            logger.debug(f"Slides processing completed.")

        # Step 3: Combine speech and slides
        if args.verbose:
            logger.debug("Step 3: Combining speech and slides with alignment")

        # Choose alignment method based on flag
        if args.enhanced_alignment:
            if args.verbose:
                logger.debug("Using enhanced similarity-based alignment")
            combined_markdown = combine_speech_and_slides_enhanced(
                speech_markdown,
                slides_markdown,
                llm=params["llm"],
                output_dir=output_dir,
                cite_timestamps=params["cite_timestamps"],
                max_tokens=params["max_tokens"],
                timeout=params["timeout"],
                temperature=params["temperature"],
                verbose=args.verbose,
            )
        else:
            if args.verbose:
                logger.debug("Using original LLM-based alignment")
            combined_markdown = combine_speech_and_slides(
                speech_markdown,
                slides_markdown,
                llm=params["llm"],
                output_dir=output_dir,
                cite_timestamps=params["cite_timestamps"],
                max_tokens=params["max_tokens"],
                timeout=params["timeout"],
                temperature=params["temperature"],
                verbose=args.verbose,
            )

        # Step 4: Save outputs
        if args.verbose:
            logger.debug("Step 4: Saving output files")

        os.makedirs(output_dir, exist_ok=True)

        # Save generated speech file (only if it was generated, not if it was input)
        if speech_file and args.verbose:
            logger.debug(f"Speech file already saved to: {speech_file}")

        # Save generated slides file (only if it was generated, not if it was input)
        if slides_file and args.verbose:
            logger.debug(f"Slides file already saved to: {slides_file}")

        # Always save combined output
        combined_file = os.path.join(output_dir, f"{base_name}_combined.md")
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write(combined_markdown)

        if args.verbose:
            logger.debug(f"Combined output saved to: {combined_file}")

        print("PPT processing completed successfully!")

        # Show which files were generated vs used as input
        if speech_file:
            print(f"Speech file (generated): {speech_file}")
        else:
            print(f"Speech file (input): {args.input}")

        if slides_file:
            print(f"Slides file (generated): {slides_file}")
        else:
            print(f"Slides file (input): {args.slides_file}")

        print(f"Combined file: {combined_file}")

    except Exception as e:
        print(f"Error during PPT processing: {e}")
        if args.verbose:
            logger.exception("Detailed error trace:")
        sys.exit(1)


def main():
    print("Debug: Starting main function...")
    download_all()
    print("Debug: download_all completed")

    # Check if this is a subcommand
    subcommands = ["rewrite", "rw", "translate", "tr", "academic", "ac", "ppt", "p"]
    is_subcommand = len(sys.argv) > 1 and sys.argv[1] in subcommands
    print(f"Debug: sys.argv = {sys.argv}")
    print(f"Debug: is_subcommand = {is_subcommand}")

    if is_subcommand:
        # Create parser for subcommands only
        parser = argparse.ArgumentParser(
            description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
        )

        # Add subparsers
        subparsers = parser.add_subparsers(
            dest="command", help="Available commands", required=True
        )

        # Rewrite subcommand
        rewrite_parser = subparsers.add_parser(
            "rewrite", aliases=["rw"], help="Rewrite text"
        )
        add_global_args(rewrite_parser)
        rewrite_parser.set_defaults(func=handle_rewrite_command)

        # Translate subcommand
        translate_parser = subparsers.add_parser(
            "translate", aliases=["tr"], help="Translate text"
        )
        add_global_args(translate_parser)
        translate_parser.set_defaults(func=handle_translate_command)

        # Academic subcommand
        academic_parser = subparsers.add_parser(
            "academic", aliases=["ac"], help="Academic rewriting"
        )
        add_global_args(academic_parser)
        academic_parser.set_defaults(func=handle_academic_command)

        # PPT subcommand - extract slides from video and combine with speech
        ppt_parser = subparsers.add_parser(
            "ppt",
            aliases=["p"],
            help="Extract slides from video and combine with speech",
        )
        add_global_args(ppt_parser)
        # Note: Legacy slide timing options removed for new workflow
        # Frame extraction options (for new workflow)
        ppt_parser.add_argument(
            "--frame-interval",
            type=int,
            default=60,
            help="Extract frame every N seconds for initial extraction (default: 60).",
        )
        ppt_parser.add_argument(
            "--each-roi",
            action="store_true",
            default=False,
            help="Enable per-frame ROI detection instead of single ROI for all frames (default: disabled).",
        )
        ppt_parser.add_argument(
            "--roi",
            nargs="?",
            const="",
            default=None,
            help="Manual ROI coordinates as 'x0,y0,x1,y1' in pixels. --roi with no value uses full screen (default: auto-detect). Example: --roi '100,50,1660,850'",
        )
        ppt_parser.add_argument(
            "--max-slides",
            "-ms",
            type=int,
            default=20,
            help="Maximum number of slides to extract (default: 20).",
        )
        ppt_parser.add_argument(
            "--no-deduplicate",
            action="store_true",
            default=False,
            help="Disable duplicate slide removal (default: deduplication enabled).",
        )
        ppt_parser.add_argument(
            "--similarity-threshold",
            "-sim",
            type=float,
            default=0.85,
            help="Text similarity threshold for deduplication, 0.0-1.0 (default: 0.85).",
        )
        ppt_parser.add_argument(
            "--dedup-method",
            choices=["image", "text", "both"],
            default="both",
            help="Deduplication method: image (SSIM), text (content), or both (default: both).",
        )
        ppt_parser.add_argument(
            "--ssim-threshold",
            type=float,
            default=0.98,
            help="SSIM threshold for image deduplication, 0.0-1.0 (default: 0.98).",
        )
        ppt_parser.add_argument(
            "--hist-threshold",
            type=float,
            default=0.15,
            help="Histogram correlation threshold for image deduplication pre-filter, 0.0-1.0 (default: 0.15).",
        )
        ppt_parser.add_argument(
            "--cropped-slide",
            nargs="?",
            const="auto",
            default=None,
            help="Enable cropped slide method. No value = auto-detect ROI with RTDETR, "
                 "or provide manual ROI coordinates as 'x0,y0,x1,y1'",
        )
        ppt_parser.add_argument(
            "--ppt",
            type=str,
            default="",
            help="Path to PPT, PDF, image, or OpenDocument file for PPT method",
        )
        ppt_parser.add_argument(
            "--no-ocr",
            action="store_true",
            default=False,
            help="Skip OCR, embed slide images as base64 instead",
        )
        ppt_parser.add_argument(
            "--no-clean",
            action="store_true",
            default=False,
            help="Keep timestamps and image references in final output",
        )
        ppt_parser.set_defaults(func=handle_ppt_command)

        print("Debug: About to parse arguments...")
        args = parser.parse_args()
        print(f"Debug: Arguments parsed. Command: {args.command}")

        print("Debug: About to execute command function...")
        args.func(args)
        print("Debug: Command function executed.")
        return

    # Main command (direct file processing)
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs.\n\nAvailable subcommands: rewrite (rw), translate (tr), academic (ac), ppt (p)\nUse 'wenbi <subcommand> --help' for subcommand-specific help."
    )
    parser.add_argument(
        "input", nargs="?", default="", help="Path to input file or URL"
    )
    parser.add_argument(
        "--config", "-c", default="", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", "-o", default="", help="Output directory (optional)"
    )
    parser.add_argument("--gui", "-g", action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--llm", default="", help="LLM model identifier (optional)")
    parser.add_argument(
        "--transcribe-lang", "-s", default="", help="Transcribe language (optional)"
    )
    parser.add_argument(
        "--lang", "-l", default="Chinese", help="Target language (default: Chinese)"
    )
    parser.add_argument(
        "--multi-language",
        "-m",
        action="store_true",
        help="Enable multi-language processing",
    )
    parser.add_argument(
        "--chunk-length",
        "-cl",
        type=int,
        default=8,
        help="Number of sentences per paragraph (default: 8)",
    )
    parser.add_argument(
        "--max-tokens",
        "-mt",
        type=int,
        default=130000,
        help="Maximum tokens for LLM output (default: 130000)",
    )
    parser.add_argument(
        "--timeout",
        "-to",
        type=int,
        default=3600,
        help="LLM request timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--temperature",
        "-tm",
        type=float,
        default=0.1,
        help="LLM temperature parameter (default: 0.1)",
    )
    parser.add_argument(
        "--transcribe-model",
        "-tsm",
        default="large-v3-turbo",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "turbo",
        ],
        help="Whisper model size for transcription (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--output_wav",
        "-ow",
        default="",
        help="Filename for saving the segmented WAV (optional)",
    )
    parser.add_argument(
        "--start_time",
        "-st",
        default="",
        help="Start time for extraction (format: HH:MM:SS)",
    )
    parser.add_argument(
        "--end_time",
        "-et",
        default="",
        help="End time for extraction (format: HH:MM:SS)",
    )
    parser.add_argument(
        "--cite-timestamps",
        action="store_true",
        default=False,
        help="Include timestamps as headers in markdown output for traceability",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output showing processing details",
    )

    args = parser.parse_args()

    # Setup logging for main command
    logger = setup_logging(args.verbose)

    # Handle config file processing for main command
    if args.config:
        if args.verbose:
            logger.debug(f"Loading configuration from: {args.config}")

        if not args.config.endswith((".yml", ".yaml")):
            print("Error: Config file must be a YAML file")
            sys.exit(1)

        config = load_config(args.config)
        if not isinstance(config, dict):
            print("Error: Invalid YAML configuration")
            sys.exit(1)

        # Add verbose to config if specified in command line
        if args.verbose:
            config["verbose"] = True

        outputs = process_yaml_config(config, args.verbose)

        if outputs:
            output_dir = config.get("output_dir", "")
            final_output = combine_markdown_files(
                outputs, output_dir, verbose=args.verbose
            )
            print(f"Combined output saved to: {final_output}")
        return

    # Load config file if provided
    config = load_config(args.config)

    # Command line arguments take precedence over config file
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "lang": args.lang or config.get("lang", "Chinese"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 130000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "transcribe_model": args.transcribe_model
        or config.get("transcribe_model", "large-v3-turbo"),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "verbose": args.verbose or config.get("verbose", False),
    }

    if args.verbose:
        logger.debug("Starting main command processing")
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != "verbose":
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params["timestamp"] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(
                f"Processing timestamp segment: {args.start_time} - {args.end_time}"
            )
    else:
        params["timestamp"] = None

    # Handle GUI mode
    if args.gui:
        if args.verbose:
            logger.debug("Launching Gradio GUI")
        launch_gui()
        return

    # Otherwise, run CLI mode (input must be provided)
    if not args.input:
        print("Error: Please specify an input file or URL.")
        sys.exit(1)

    if args.verbose:
        logger.debug(f"Processing input: {args.input}")

    is_url = args.input.startswith(("http://", "https://", "www."))
    if args.verbose:
        logger.debug(f"Input type: {'URL' if is_url else 'File'}")

    result = process_input(
        None if is_url else args.input, args.input if is_url else "", **params
    )

    print("Markdown Output:", result[0])
    print("Markdown File:", result[1])
    print("CSV File:", result[2])
    print("Filename (without extension):", result[3] if result[3] is not None else "")


if __name__ == "__main__":
    main()
