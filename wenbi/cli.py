#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import yaml

from wenbi.download import download_all
from wenbi.main import process_input
from wenbi.model import rewrite, translate
from wenbi.ppt_slide import combine_speech_and_slides_by_timestamp


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
        audio_extensions = (".mp3", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".wav")
        return file_path.lower().endswith(video_extensions + audio_extensions)

    return False


def validate_transcription_args(args):
    """Validate that transcription-related arguments are only used with video/audio/URL inputs"""
    # Check if input is video, audio, or URL
    is_media_input = is_video_audio_or_url(args.input, "")

    # Check for transcription-related arguments
    transcription_args = []
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

    style = getattr(args, "style", "rewrite")

    # Validate transcription arguments
    validate_transcription_args(args)

    # Load config if provided
    config = load_config(args.config)

    # --ppt forces cite_timestamps=True so combine can align slides with speech
    ppt_active = getattr(args, "ppt", None) is not None

    # Prepare parameters
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 64000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "lang": args.lang or config.get("lang", "Chinese"),
        "subcommand": {"academic": "academic", "zh-speaker": "zh-speaker"}.get(style, "rewrite"),
        "enable_speakers": style == "zh-speaker",
        "asr_provider": getattr(args, "asr_provider", "auto"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": True if ppt_active else (args.cite_timestamps or config.get("cite_timestamps", False)),
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
        if style == "academic":
            output_dir = params["output_dir"] or os.getcwd()
            base_name = result[3] or os.path.splitext(os.path.basename(args.input))[0]
            output_file = os.path.join(output_dir, f"{base_name}_academic.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result[0])
            print("Academic rewriting completed successfully!")
            print("Output file:", output_file)
            print("You can find the academic text in:", output_file)
        elif style == "zh-speaker":
            output_dir = params["output_dir"] or os.getcwd()
            base_name = result[3] or os.path.splitext(os.path.basename(args.input))[0]
            output_file = os.path.join(output_dir, f"{base_name}_zh_speaker.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result[0])
            print("Speaker-aware rewriting completed successfully!")
            print("Output file:", output_file)
            print("You can find the speaker-attributed text in:", output_file)
        else:
            print("Rewrite completed successfully!")
            print("Output file:", result[1] if result[1] else "Text output only")
            if result[1]:
                print("You can find the rewritten text in:", result[1])

        # Stage 6: slide-combine (optional, gated by --ppt)
        if ppt_active:
            output_dir = params["output_dir"] or os.getcwd()
            base_name = result[3] or os.path.splitext(os.path.basename(args.input))[0]
            is_url = args.input.startswith(("http://", "https://", "www."))
            combine_md, combine_clean_md = finalize_with_slides(
                args, result[0], args.input, output_dir, base_name, logger, args.verbose
            )
            if combine_md:
                print("Combined:", combine_md)
            if combine_clean_md:
                print("Cleaned:", combine_clean_md)
    else:
        print("Error:", result[0])



def handle_en_zh_command(args):
    """Handle the en-zh bilingual source extraction and translation command"""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting en-zh command")
        logger.debug(f"Input: {args.input}")
        logger.debug(f"ASR provider: {args.asr_provider}")

    from wenbi.bilingual import process_en_zh

    try:
        result = process_en_zh(
            input_path=args.input,
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            asr_provider=args.asr_provider,
            source_lang=args.source_lang,
            interpreter_lang=args.interpreter_lang,
            target_language=args.lang or "Chinese",
            llm=args.llm or "ollama/glm-5.2:cloud",
            chunk_length=args.chunk_length,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            deepl_key=args.deepl_key,
            gladia_key=args.gladia_key,
            speaker_labels=args.speaker_labels,
            save_json=args.save_json,
            verbose=args.verbose,
            use_glossary=getattr(args, "glossary", True),
            glossary_file=getattr(args, "glossary_file", None),
        )
    except Exception as e:
        if args.verbose:
            logger.exception("Unexpected error during en-zh processing")
        print(f"Error: {e}")
        return

    print("English-to-Chinese bilingual processing completed successfully!")
    print("ASR provider:", result.provider)
    print("Kept English segments:", result.kept_segments)
    print("Dropped non-English segments:", result.dropped_segments)
    print("English VTT:", result.english_vtt)
    print("English Markdown:", result.english_md)
    print("Bilingual Markdown:", result.bilingual_md)
    if result.gladia_vtt:
        print("Gladia Raw VTT:", result.gladia_vtt)
    if result.english_rewritten_md:
        print("English Rewritten Markdown:", result.english_rewritten_md)
    if result.diagnostics_json:
        print("Diagnostics JSON:", result.diagnostics_json)

    # Stage 6: slide-combine (optional, gated by --ppt)
    # Timestamps recovered from VTT via fuzzy matching (rapidfuzz).
    # Reads english_rewritten_md as the speech markdown for combine.
    if getattr(args, "ppt", None) is not None and result.english_rewritten_md:
        output_dir = args.output_dir or os.getcwd()
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        try:
            with open(result.english_rewritten_md, "r", encoding="utf-8") as f:
                speech_md = f.read()
            combine_md, combine_clean_md = finalize_with_slides(
                args, speech_md, args.input, output_dir, base_name, logger, args.verbose,
                vtt_path=result.english_vtt,
            )
            if combine_md:
                print("Combined:", combine_md)
            if combine_clean_md:
                print("Cleaned:", combine_clean_md)
        except Exception as e:
            print(f"Warning: --ppt slide-combine failed: {e}")
            if args.verbose:
                logger.exception("--ppt combine error details:")


def handle_speaker_command(args):
    """Handle the speaker subcommand — single-language multi-speaker workflow."""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting speaker command")
        logger.debug(f"Input: {args.input}")
        logger.debug(f"ASR provider: {args.asr_provider}")

    from wenbi.bilingual import process_speaker

    try:
        result = process_speaker(
            input_path=args.input,
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            asr_provider=args.asr_provider,
            source_lang=args.source_lang,
            target_language=args.lang or "Chinese",
            llm=args.llm or "ollama/glm-5.2:cloud",
            chunk_length=args.chunk_length,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            deepl_key=args.deepl_key,
            gladia_key=args.gladia_key,
            speaker_labels=args.speaker_labels,
            speaker_count=getattr(args, "speaker_count", None),
            save_json=args.save_json,
            verbose=args.verbose,
            use_glossary=getattr(args, "glossary", True),
            glossary_file=getattr(args, "glossary_file", None),
        )
    except Exception as e:
        if args.verbose:
            logger.exception("Unexpected error during speaker processing")
        print(f"Error: {e}")
        return

    print("Speaker-aware processing completed successfully!")
    print("ASR provider:", result.provider)
    print("Speakers detected:", result.num_speakers)
    print("Total segments:", result.total_segments)
    print("Transcript VTT:", result.transcript_vtt)
    print("Transcript Markdown:", result.transcript_md)
    print("Rewritten Markdown:", result.rewritten_md)
    if result.bilingual_md:
        print("Bilingual Markdown:", result.bilingual_md)
    if result.gladia_vtt:
        print("Gladia Raw VTT:", result.gladia_vtt)
    if result.diagnostics_json:
        print("Diagnostics JSON:", result.diagnostics_json)

    # Stage 6: slide-combine (optional, gated by --ppt)
    # Timestamps recovered from VTT via fuzzy matching (rapidfuzz).
    if getattr(args, "ppt", None) is not None and result.rewritten_md:
        output_dir = args.output_dir or os.getcwd()
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        try:
            with open(result.rewritten_md, "r", encoding="utf-8") as f:
                speech_md = f.read()
            combine_md, combine_clean_md = finalize_with_slides(
                args, speech_md, args.input, output_dir, base_name, logger, args.verbose,
                vtt_path=result.transcript_vtt,
            )
            if combine_md:
                print("Combined:", combine_md)
            if combine_clean_md:
                print("Cleaned:", combine_clean_md)
        except Exception as e:
            print(f"Warning: --ppt slide-combine failed: {e}")
            if args.verbose:
                logger.exception("--ppt combine error details:")


def handle_zh_zh_command(args):
    """Handle Chinese interview transcription and speaker-preserving rewrite."""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting zh-zh command")
        logger.debug(f"Input: {args.input}")
        logger.debug(f"ASR provider: {args.asr_provider}")

    from wenbi.bilingual import process_speaker

    try:
        result = process_speaker(
            input_path=args.input,
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            asr_provider=args.asr_provider,
            source_lang="zh",
            target_language=args.lang or "Chinese",
            llm=args.llm or "ollama/glm-5.2:cloud",
            chunk_length=args.chunk_length,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            deepl_key=args.deepl_key,
            gladia_key=args.gladia_key,
            speaker_labels=args.speaker_labels,
            speaker_count=args.speaker_count,
            rewrite_mode="zh-interview",
            save_json=args.save_json,
            verbose=args.verbose,
        )
    except Exception as e:
        if args.verbose:
            logger.exception("Unexpected error during zh-zh processing")
        print(f"Error: {e}")
        return

    print("Chinese interview processing completed successfully!")
    print("ASR provider:", result.provider)
    print("Speakers detected:", result.num_speakers)
    print("Expected speakers:", args.speaker_count)
    print("Total segments:", result.total_segments)
    print("Transcript VTT:", result.transcript_vtt)
    print("Transcript Markdown:", result.transcript_md)
    print("Rewritten Markdown:", result.rewritten_md)
    if result.gladia_vtt:
        print("Gladia Raw VTT:", result.gladia_vtt)
    if result.diagnostics_json:
        print("Diagnostics JSON:", result.diagnostics_json)

    # Stage 6: slide-combine (optional, gated by --ppt)
    # Timestamps recovered from VTT via fuzzy matching (rapidfuzz).
    if getattr(args, "ppt", None) is not None and result.rewritten_md:
        output_dir = args.output_dir or os.getcwd()
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        try:
            with open(result.rewritten_md, "r", encoding="utf-8") as f:
                speech_md = f.read()
            combine_md, combine_clean_md = finalize_with_slides(
                args, speech_md, args.input, output_dir, base_name, logger, args.verbose,
                vtt_path=result.transcript_vtt,
            )
            if combine_md:
                print("Combined:", combine_md)
            if combine_clean_md:
                print("Cleaned:", combine_clean_md)
        except Exception as e:
            print(f"Warning: --ppt slide-combine failed: {e}")
            if args.verbose:
                logger.exception("--ppt combine error details:")


def handle_en_en_command(args):
    """Handle English interview transcription and speaker-preserving rewrite."""
    logger = setup_logging(args.verbose)

    if args.verbose:
        logger.debug("Starting en-en command")
        logger.debug(f"Input: {args.input}")
        logger.debug(f"ASR provider: {args.asr_provider}")

    from wenbi.bilingual import process_speaker

    try:
        result = process_speaker(
            input_path=args.input,
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            asr_provider=args.asr_provider,
            source_lang="en",
            target_language=args.lang or "Chinese",
            llm=args.llm or "ollama/glm-5.2:cloud",
            chunk_length=args.chunk_length,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            deepl_key=args.deepl_key,
            gladia_key=args.gladia_key,
            speaker_labels=args.speaker_labels,
            speaker_count=args.speaker_count,
            rewrite_mode="en-interview",
            use_glossary=getattr(args, "glossary", True),
            glossary_file=getattr(args, "glossary_file", None),
            save_json=args.save_json,
            verbose=args.verbose,
        )
    except Exception as e:
        if args.verbose:
            logger.exception("Unexpected error during en-en processing")
        print(f"Error: {e}")
        return

    print("English interview processing completed successfully!")
    print("ASR provider:", result.provider)
    print("Speakers detected:", result.num_speakers)
    print("Expected speakers:", args.speaker_count)
    print("Total segments:", result.total_segments)
    print("Transcript VTT:", result.transcript_vtt)
    print("Transcript Markdown:", result.transcript_md)
    print("Rewritten Markdown:", result.rewritten_md)
    if result.gladia_vtt:
        print("Gladia Raw VTT:", result.gladia_vtt)
    if result.diagnostics_json:
        print("Diagnostics JSON:", result.diagnostics_json)

    # Stage 6: slide-combine (optional, gated by --ppt)
    # Timestamps recovered from VTT via fuzzy matching (rapidfuzz).
    if getattr(args, "ppt", None) is not None and result.rewritten_md:
        output_dir = args.output_dir or os.getcwd()
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        try:
            with open(result.rewritten_md, "r", encoding="utf-8") as f:
                speech_md = f.read()
            combine_md, combine_clean_md = finalize_with_slides(
                args, speech_md, args.input, output_dir, base_name, logger, args.verbose,
                vtt_path=result.transcript_vtt,
            )
            if combine_md:
                print("Combined:", combine_md)
            if combine_clean_md:
                print("Cleaned:", combine_clean_md)
        except Exception as e:
            print(f"Warning: --ppt slide-combine failed: {e}")
            if args.verbose:
                logger.exception("--ppt combine error details:")


def add_global_args(subparser):
    """Add common arguments to subparsers"""
    subparser.add_argument("input", help="Path to input file or URL")
    subparser.add_argument(
        "--config", "-c", default="", help="Path to YAML configuration file"
    )
    subparser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Enable verbose logging"
    )
    subparser.add_argument(
        "--start_time", "-st", default="", help="Start time for extraction (format: HH:MM:SS)"
    )
    subparser.add_argument(
        "--end_time", "-et", default="", help="End time for extraction (format: HH:MM:SS)"
    )
    subparser.add_argument(
        "--deepl-key", default="", help="DeepL API key (uses DEEPL_API_KEY env var if not provided)"
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
        default=64000,
        help="Maximum tokens for LLM output (default: 64000)",
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
    # ASR provider (replaces --transcribe-model; whisper model is hardcoded
    # large-v3-turbo inside asr.py)
    subparser.add_argument(
        "--asr-provider",
        "-asr",
        choices=["auto", "gladia", "funasr", "whisper"],
        default="auto",
        help="ASR backend: auto (gladia if GLADIA_API_KEY else funasr), gladia, funasr (local), or whisper (default: auto)",
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


def add_slide_args(subparser):
    """Add --ppt / -p and frame-extraction options to a subcommand parser.

    --ppt absent      : no slide workflow (normal subcommand behavior)
    --ppt (no value)  : TYPE 1 — embed video frames as base64 with timestamps (no OCR)
    --ppt <file>      : TYPE 3 — OCR the slides file (PPT/PDF/image), match pages
                        to video frames via SSIM, combine
    """
    subparser.add_argument(
        "--ppt",
        "-p",
        nargs="?",
        const="",
        default=None,
        help="Enable slide-combine (stage 6). Bare --ppt / -p: embed video frames "
             "as base64 (TYPE 1, no OCR). --ppt <file>: OCR the given PPT/PDF/image "
             "and match to frames (TYPE 3).",
    )
    subparser.add_argument(
        "--frame-interval",
        type=int,
        default=60,
        help="Extract frame every N seconds for initial extraction (default: 60).",
    )
    subparser.add_argument(
        "--max-slides",
        "-ms",
        type=int,
        default=20,
        help="Maximum number of slides to extract (default: 20).",
    )
    subparser.add_argument(
        "--no-deduplicate",
        action="store_true",
        default=False,
        help="Disable duplicate slide removal (default: deduplication enabled).",
    )
    subparser.add_argument(
        "--similarity-threshold",
        "-sim",
        type=float,
        default=0.85,
        help="Text similarity threshold for deduplication, 0.0-1.0 (default: 0.85).",
    )
    subparser.add_argument(
        "--dedup-method",
        choices=["image", "text", "both"],
        default="both",
        help="Deduplication method: image (SSIM), text (content), or both (default: both).",
    )
    subparser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.98,
        help="SSIM threshold for image deduplication, 0.0-1.0 (default: 0.98).",
    )
    subparser.add_argument(
        "--hist-threshold",
        type=float,
        default=0.15,
        help="Histogram correlation threshold for image deduplication pre-filter, 0.0-1.0 (default: 0.15).",
    )
    subparser.add_argument(
        "--no-clean",
        action="store_true",
        default=False,
        help="Keep timestamps and image references in final combined output",
    )
    subparser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False,
        help="Skip OCR on slide images, embed them as base64 instead (TYPE 3 only; TYPE 1 never OCRs).",
    )


def finalize_with_slides(args, audio_markdown, video_path, output_dir, base_name,
                          logger, verbose, vtt_path=None):
    """Stage 6: when --ppt is set, extract frames, build slides (TYPE 1 or TYPE 3),
    combine with the host command's audio_markdown, clean, and return output paths.

    Returns (combine_md, combine_clean_md) when --ppt is set, else (None, None).
    Only adds stage 6 — the host command has already run stages 1-5 (download, ASR,
    diarize, rewrite, translate) and produced audio_markdown.

    audio_markdown should be timestamped (### **HH:MM:SS - HH:MM:SS** headers) for
    alignment. The `rewrite` subcommand produces this directly (forces
    cite_timestamps=True when --ppt is set). The bilingual subcommands strip
    timestamps — pass vtt_path so we can recover timestamps via fuzzy matching
    against the VTT before combining.
    """
    if getattr(args, "ppt", None) is None:
        return None, None

    from wenbi.ppt_slide import (
        build_slides_markdown,
        combine_speech_and_slides_by_timestamp,
        recover_timestamps_from_vtt,
    )

    is_url = video_path.startswith(("http://", "https://", "www."))

    # Download video if URL (frames need a local file)
    if is_url:
        if verbose:
            logger.debug("--ppt: downloading video for frame extraction...")
        from wenbi.utils import download_video
        try:
            download_result = download_video(video_path, output_dir=output_dir, verbose=verbose)
            if download_result:
                video_path = download_result
            else:
                print("Error: Failed to download video from URL for --ppt")
                sys.exit(1)
        except Exception as e:
            print(f"Error downloading video for --ppt: {e}")
            sys.exit(1)
        if verbose:
            logger.debug(f"--ppt: video downloaded to: {video_path}")

    # Validate local video file
    if not is_url:
        from wenbi.video_slides import validate_video_input
        if not validate_video_input(video_path, logger, verbose):
            print(f"Error: Invalid video file for --ppt: {video_path}")
            sys.exit(1)

    if verbose:
        logger.debug("--ppt: Step 1: Extracting and deduplicating frames...")

    deduplicated_frames = extract_and_deduplicate_frames(
        video_path=video_path,
        start_time=getattr(args, "start_time", ""),
        end_time=getattr(args, "end_time", ""),
        frame_interval=args.frame_interval,
        ssim_threshold=args.ssim_threshold,
        hist_threshold=args.hist_threshold,
        output_dir=output_dir,
        logger=logger,
        verbose=verbose,
    )

    if verbose:
        logger.debug(
            f"--ppt: {len(deduplicated_frames)} deduplicated frames extracted"
        )

    # Build slides markdown
    slides_md = build_slides_markdown(
        deduplicated_frames=deduplicated_frames,
        ppt_path=args.ppt if args.ppt else None,  # "" -> None for TYPE 1
        output_dir=output_dir,
        no_ocr=args.no_ocr,
        base_name=base_name,
        logger=logger,
        verbose=verbose,
        ssim_threshold=args.ssim_threshold,
    )

    if verbose:
        logger.debug("--ppt: Step 6: Combining slides and audio markdown...")

    # Recover timestamps for bilingual output if VTT is available
    speech_md = audio_markdown
    if vtt_path and os.path.exists(vtt_path):
        if verbose:
            logger.debug(f"--ppt: recovering timestamps from VTT: {vtt_path}")
        speech_md = recover_timestamps_from_vtt(
            speech_markdown=audio_markdown,
            vtt_path=vtt_path,
            verbose=verbose,
        )

    with open(slides_md, "r", encoding="utf-8") as f:
        slides_content = f.read()

    combined_markdown = combine_speech_and_slides_by_timestamp(
        speech_markdown=speech_md,
        slides_markdown=slides_content,
        verbose=verbose,
    )

    combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
    with open(combine_md, "w", encoding="utf-8") as f:
        f.write(combined_markdown)

    if verbose:
        logger.debug(f"--ppt: Combined markdown: {combine_md}")

    # Clean (if not --no-clean)
    if args.no_clean:
        combine_clean_md = None
        if verbose:
            logger.debug("--ppt: --no-clean: skipping clean phase")
    else:
        combine_clean_md = clean_combined_markdown(
            combine_md, output_dir, base_name, logger, verbose
        )

    return combine_md, combine_clean_md
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
        "--keep-original-lang",
        "-kol",
        action="store_true",
        default=False,
        help="Keep original language alongside translated text (original above, translation below)",
    )
    subparser.add_argument(
        "--deepl-key",
        default="",
        help="DeepL API key (uses DEEPL_API_KEY env var if not provided)",
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


def extract_and_deduplicate_frames(
    video_path,
    start_time,
    end_time,
    frame_interval,
    ssim_threshold,
    hist_threshold,
    output_dir,
    logger,
    verbose,
):
    """Extract and deduplicate frames from video."""
    from wenbi.video_slides import (
        deduplicate_slides_by_image,
        extract_all_frames_from_video,
    )

    if verbose:
        logger.debug(f"Extracting frames with interval {frame_interval}s...")

    all_frames = extract_all_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        start_time=start_time or "00:00:00",
        end_time=end_time or None,
        frame_interval=frame_interval,
        logger=logger,
        verbose=verbose,
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
        verbose=verbose,
    )

    removed = len(all_frames) - len(deduplicated)
    if verbose:
        logger.debug(
            f"Removed {removed} duplicates, {len(deduplicated)} unique frames remain"
        )

    if not deduplicated:
        print("Error: No frames after deduplication")
        sys.exit(1)

    return deduplicated


def run_marker_pdf_on_image(image_path, output_dir, verbose=False, logger=None):
    """
    Run marker_single shell command on image file.
    Returns dict with text, base64_images, and success status.
    """
    import base64
    import shutil
    import subprocess

    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(image_path):
        return {
            "text": "",
            "base64_images": {},
            "success": False,
            "error": f"Image not found: {image_path}",
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
            "--output_format",
            "markdown",
            "--output_dir",
            temp_marker_dir,
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
                "error": error_msg,
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
                "error": "Marker did not generate markdown output",
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

        return {"text": markdown_text, "base64_images": base64_images, "success": True}

    except subprocess.TimeoutExpired:
        return {
            "text": "",
            "base64_images": {},
            "success": False,
            "error": "marker_single timeout (>5 min)",
        }

    except Exception as e:
        logger.error(f"Error running marker_single: {e}")
        return {"text": "", "base64_images": {}, "success": False, "error": str(e)}

    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_marker_dir)
        except:
            pass


def embed_frames_as_base64(
    frames_with_timestamps, output_dir, base_name, logger, verbose
):
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
    Remove timestamps and image file references, keep embedded base64 images.
    """
    if verbose:
        logger.debug("Cleaning combined markdown...")

    with open(combine_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove timestamp headers (### **HH:MM:SS** or ### **HH:MM:SS - HH:MM:SS**)
    import re

    content = re.sub(
        r"\n### \*\*\d{2}:\d{2}:\d{2}(?:\s*-\s*\d{2}:\d{2}:\d{2})?\*\*\n",
        "\n",
        content,
    )

    # Remove image file references ![slide](/path/to/image.png)
    # but keep <img src="data:image/png;base64,..."/>
    content = re.sub(r"!\[.*?\]\([^)]*\.png\)", "", content)

    clean_path = os.path.join(output_dir, f"{base_name}_combine_clean.md")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(content)

    if verbose:
        logger.debug(f"Cleaned markdown: {clean_path}")

    return clean_path


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
                        logger.debug(
                            f"Embedded image for slide at {slide['timestamp']}"
                        )

                except Exception as e:
                    if verbose and logger:
                        logger.warning(
                            f"Could not embed image for slide at {slide['timestamp']}: {e}"
                        )
                    f.write(f"> Image embedding failed: {e}\n\n")

            # If no content and no image, note it
            if not ocr_content and not image_data:
                f.write("> No content available for this slide.\n\n")

    if verbose and logger:
        logger.debug(
            f"Finished saving slides to markdown with embedded Picture images."
        )

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
        import re
        import shutil
        import sys

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
                verbose=verbose,
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
                verbose=verbose,
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


def main():
    print("Debug: Starting main function...")
    download_all()
    print("Debug: download_all completed")

    # Check if this is a subcommand
    subcommands = ["rewrite", "rw", "en-zh", "enzh", "en-en", "enen", "zh-zh", "zhzh", "speaker", "sp"]
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
        add_slide_args(rewrite_parser)
        rewrite_parser.add_argument(
            "--style",
            choices=["rewrite", "academic", "zh-speaker"],
            default="rewrite",
            help="Rewrite style: rewrite (default), academic, or zh-speaker (Chinese with speaker diarization)",
        )
        rewrite_parser.set_defaults(func=handle_rewrite_command)

        # English-to-Chinese bilingual audio subcommand
        en_zh_parser = subparsers.add_parser(
            "en-zh",
            aliases=["enzh"],
            help="Extract English from English/Chinese bilingual audio and translate it to Chinese",
        )
        add_global_args(en_zh_parser)
        add_slide_args(en_zh_parser)
        en_zh_parser.add_argument(
            "--source-lang",
            default="en",
            help="Source language to keep from the bilingual audio (default: en)",
        )
        en_zh_parser.add_argument(
            "--interpreter-lang",
            default="zh",
            help="Interpreter language to drop from the source transcript (default: zh)",
        )
        en_zh_parser.add_argument(
            "--gladia-key",
            default="",
            help="Gladia API key (uses GLADIA_API_KEY env var if not provided)",
        )
        en_zh_parser.add_argument(
            "--speaker-labels",
            action="store_true",
            default=True,
            help="Request speaker labels when the ASR provider supports them (default: enabled)",
        )
        en_zh_parser.add_argument(
            "--no-speaker-labels",
            dest="speaker_labels",
            action="store_false",
            help="Disable speaker labels",
        )
        en_zh_parser.add_argument(
            "--save-json",
            action="store_true",
            default=False,
            help="Save normalized segment diagnostics and raw provider JSON when available",
        )
        en_zh_parser.add_argument(
            "--glossary",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Apply EN→ZH glossary for term consistency (DeepL glossary API + LLM prompt). No-op when target language is not Chinese (default: enabled).",
        )
        en_zh_parser.add_argument(
            "--glossary-file",
            type=str,
            default=None,
            help="Path to user glossary JSON ({english: chinese} dict). Overrides the built-in patristic glossary.",
        )
        en_zh_parser.set_defaults(
            func=handle_en_zh_command,
        )

        # English interview subcommand
        en_en_parser = subparsers.add_parser(
            "en-en",
            aliases=["enen"],
            help="Transcribe and rewrite English interviews with speaker-separated output",
        )
        add_global_args(en_en_parser)
        add_slide_args(en_en_parser)
        en_en_parser.add_argument(
            "--gladia-key",
            default="",
            help="Gladia API key (uses GLADIA_API_KEY env var if not provided)",
        )
        en_en_parser.add_argument(
            "--speaker-labels",
            action="store_true",
            default=True,
            help="Request speaker labels when the ASR provider supports them (default: enabled)",
        )
        en_en_parser.add_argument(
            "--no-speaker-labels",
            dest="speaker_labels",
            action="store_false",
            help="Disable speaker labels",
        )
        en_en_parser.add_argument(
            "--speaker-count",
            type=int,
            default=2,
            help="Expected number of interview speakers for diarization and rewrite (default: 2)",
        )
        en_en_parser.add_argument(
            "--save-json",
            action="store_true",
            default=False,
            help="Save segment diagnostics and raw provider JSON when available",
        )
        en_en_parser.add_argument(
            "--glossary",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Apply EN→ZH glossary for term consistency (DeepL glossary API + LLM prompt). No-op when target language is not Chinese (default: enabled).",
        )
        en_en_parser.add_argument(
            "--glossary-file",
            default=None,
            help="Path to a custom glossary JSON ({english: chinese} dict). Overrides the built-in patristic glossary.",
        )
        en_en_parser.set_defaults(
            func=handle_en_en_command,
        )

        # Chinese interview subcommand
        zh_zh_parser = subparsers.add_parser(
            "zh-zh",
            aliases=["zhzh"],
            help="Transcribe and rewrite Chinese interviews with speaker-separated output",
        )
        add_global_args(zh_zh_parser)
        add_slide_args(zh_zh_parser)
        zh_zh_parser.add_argument(
            "--gladia-key",
            default="",
            help="Gladia API key (uses GLADIA_API_KEY env var if not provided)",
        )
        zh_zh_parser.add_argument(
            "--speaker-labels",
            action="store_true",
            default=True,
            help="Request speaker labels when the ASR provider supports them (default: enabled)",
        )
        zh_zh_parser.add_argument(
            "--no-speaker-labels",
            dest="speaker_labels",
            action="store_false",
            help="Disable speaker labels",
        )
        zh_zh_parser.add_argument(
            "--speaker-count",
            type=int,
            default=2,
            help="Expected number of interview speakers for diarization and rewrite (default: 2)",
        )
        zh_zh_parser.add_argument(
            "--save-json",
            action="store_true",
            default=False,
            help="Save segment diagnostics and raw provider JSON when available",
        )
        zh_zh_parser.set_defaults(
            func=handle_zh_zh_command,
        )

        # Speaker-aware single-language subcommand
        speaker_parser = subparsers.add_parser(
            "speaker",
            aliases=["sp"],
            help="Transcribe single-language multi-speaker audio with diarization, rewrite, and translate",
        )
        add_global_args(speaker_parser)
        add_slide_args(speaker_parser)
        speaker_parser.add_argument(
            "--source-lang",
            default="en",
            help="Language of the audio (default: en)",
        )
        speaker_parser.add_argument(
            "--gladia-key",
            default="",
            help="Gladia API key (uses GLADIA_API_KEY env var if not provided)",
        )
        speaker_parser.add_argument(
            "--speaker-labels",
            action="store_true",
            default=True,
            help="Request speaker labels when the ASR provider supports them (default: enabled)",
        )
        speaker_parser.add_argument(
            "--no-speaker-labels",
            dest="speaker_labels",
            action="store_false",
            help="Disable speaker labels",
        )
        speaker_parser.add_argument(
            "--speaker-count",
            type=int,
            default=None,
            help="Expected number of speakers for diarization (default: provider decides)",
        )
        speaker_parser.add_argument(
            "--save-json",
            action="store_true",
            default=False,
            help="Save segment diagnostics and raw provider JSON when available",
        )
        speaker_parser.add_argument(
            "--glossary",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Apply EN→ZH glossary for term consistency (DeepL glossary API + LLM prompt). No-op when target language is not Chinese (default: enabled).",
        )
        speaker_parser.add_argument(
            "--glossary-file",
            type=str,
            default=None,
            help="Path to user glossary JSON ({english: chinese} dict). Overrides the built-in patristic glossary.",
        )
        speaker_parser.set_defaults(
            func=handle_speaker_command,
        )

        print("Debug: About to parse arguments...")
        args = parser.parse_args()

        print("Debug: About to execute command function...")
        args.func(args)
        print("Debug: Command function executed.")
        return

    # Main command (direct file processing)
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs.\n\nAvailable subcommands: rewrite (rw), en-zh (enzh), en-en (enen), zh-zh (zhzh), speaker (sp)\nAll subcommands accept --ppt / -p for optional slide-combine.\nUse 'wenbi <subcommand> --help' for subcommand-specific help."
    )
    parser.add_argument(
        "input", nargs="?", default="", help="Path to input file or URL"
    )
    parser.add_argument(
        "--config", "-c", default="", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-dir", "-o", default="", help="Output directory (optional)"
    )
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
        default=64000,
        help="Maximum tokens for LLM output (default: 64000)",
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
        "--asr-provider",
        "-asr",
        choices=["auto", "gladia", "funasr", "whisper"],
        default="auto",
        help="ASR backend: auto (gladia if GLADIA_API_KEY else funasr), gladia, funasr (local), or whisper (default: auto)",
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
        "--keep-original-lang",
        "-kol",
        action="store_true",
        default=False,
        help="Keep original language alongside translated text (original above, translation below)",
    )
    parser.add_argument(
        "--deepl-key",
        default="",
        help="DeepL API key (uses DEEPL_API_KEY env var if not provided)",
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

    # DeepL is always enabled by default
    deepl_key = args.deepl_key if hasattr(args, 'deepl_key') else ""

    # Command line arguments take precedence over config file
    params = {
        "output_dir": args.output_dir or config.get("output_dir", ""),
        "llm": args.llm or config.get("llm", ""),
        "transcribe_lang": args.transcribe_lang or config.get("transcribe_lang", ""),
        "lang": args.lang or config.get("lang", "Chinese"),
        "multi_language": args.multi_language or config.get("multi_language", False),
        "chunk_length": args.chunk_length or config.get("chunk_length", 20),
        "max_tokens": args.max_tokens or config.get("max_tokens", 64000),
        "timeout": args.timeout or config.get("timeout", 3600),
        "temperature": args.temperature or config.get("temperature", 0.1),
        "asr_provider": getattr(args, "asr_provider", "auto"),
        "output_wav": args.output_wav or config.get("output_wav", ""),
        "cite_timestamps": args.cite_timestamps or config.get("cite_timestamps", False),
        "keep_original_lang": args.keep_original_lang or config.get("keep_original_lang", False),
        "use_deepl": True,
        "deepl_key": deepl_key,
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
