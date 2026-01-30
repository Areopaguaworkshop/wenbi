#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import logging
from wenbi.main import process_input
from wenbi.model import rewrite, translate, academic, convert_slides_to_markdown, combine_speech_and_slides, combine_speech_and_slides_enhanced, read_markdown_file
from wenbi.download import download_all
from wenbi.gui import launch_gui


def setup_logging(verbose=False):
    """Setup logging configuration based on verbose flag"""
    if verbose:
        level = logging.DEBUG
        format_str = '[VERBOSE] %(message)s'
    else:
        level = logging.INFO
        format_str = '%(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def combine_markdown_files(outputs, output_dir, final_filename="combined_output.md", verbose=False):
    """Combine multiple markdown outputs into a single file"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Combining {len(outputs)} markdown outputs into {final_filename}")
    
    combined_path = os.path.join(output_dir, final_filename)
    with open(combined_path, 'w', encoding='utf-8') as f:
        for idx, (title, content) in enumerate(outputs):
            if verbose:
                logger.debug(f"Adding section {idx+1}/{len(outputs)}: {title}")
            
            if idx > 0:
                f.write('\n---\n\n')  # Separator between sections
            f.write(f'# {title}\n\n')
            # Read and append content from markdown file
            if os.path.isfile(content):
                with open(content, 'r', encoding='utf-8') as mf:
                    f.write(mf.read())
            else:
                f.write(content)
    
    if verbose:
        logger.debug(f"Combined output saved to: {combined_path}")
    
    return combined_path


def parse_timestamp(start_time, end_time):
    """Parse start and end time strings in the format HH:MM:SS"""
    try:
        return {'start': start_time.strip(), 'end': end_time.strip()}
    except ValueError:
        print("Error: Invalid timestamp format. Use HH:MM:SS for both start and end times.")
        sys.exit(1)


def process_yaml_config(config, verbose=False):
    """Process YAML config supporting both single and multiple input formats"""
    logger = logging.getLogger(__name__)
    outputs = []

    if verbose:
        logger.debug("Processing YAML configuration")
        logger.debug(f"Config keys: {list(config.keys())}")

    # Handle single input with segments
    if 'input' in config and 'segments' in config:
        input_path = config['input']
        params = {**config}
        params.pop('input', None)
        params.pop('segments', None)
        params['verbose'] = verbose

        if verbose:
            logger.debug(f"Processing single input with segments: {input_path}")
            logger.debug(f"Found {len(config['segments'])} segments")

        for idx, segment in enumerate(config['segments'], 1):
            # Make all segment fields optional
            if not isinstance(segment, dict):
                continue

            if verbose:
                logger.debug(f"Processing segment {idx}/{len(config['segments'])}")

            # Get timestamp if provided, otherwise process whole file
            if 'start_time' in segment and 'end_time' in segment:
                params['timestamp'] = parse_timestamp(
                    segment['start_time'],
                    segment['end_time']
                )
                if verbose:
                    logger.debug(f"Segment timestamp: {segment['start_time']} - {segment['end_time']}")
            else:
                params['timestamp'] = None

            # Get output_wav if provided
            params['output_wav'] = segment.get('output_wav', '')

            result = process_input(
                input_path if not input_path.startswith(("http://", "https://", "www.")) else None,
                input_path if input_path.startswith(("http://", "https://", "www.")) else "",
                **params
            )

            if result[0] and result[3]:
                # Use title if provided, otherwise use generated base_name
                title = segment.get('title', f"Segment {idx}" if params['timestamp'] else result[3])
                outputs.append((title, result[1] or result[0]))
                if verbose:
                    logger.debug(f"Segment {idx} processed successfully: {title}")

        # Combine outputs into single file
        if outputs:
            output_dir = config.get('output_dir', '')
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            final_output = combine_markdown_files(outputs, output_dir, f"{base_name}_combined.md", verbose)
            print(f"Combined output saved to: {final_output}")

    # Handle multiple inputs with or without segments
    if 'inputs' in config:
        if verbose:
            logger.debug(f"Processing multiple inputs: {len(config['inputs'])} files")

        for input_idx, input_config in enumerate(config['inputs'], 1):
            input_path = input_config['input']
            
            if verbose:
                logger.debug(f"Processing input {input_idx}/{len(config['inputs'])}: {input_path}")

            # If no segments defined, process the entire file
            if 'segments' not in input_config:
                params = {**config, **input_config}
                params.pop('inputs', None)
                params.pop('input', None)
                params['verbose'] = verbose

                result = process_input(
                    input_path if not input_path.startswith(("http://", "https://", "www.")) else None,
                    input_path if input_path.startswith(("http://", "https://", "www.")) else "",
                    **params
                )

                if result[0] and result[3]:
                    # Use filename as title for full file processing
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    outputs.append((base_name, result[1] or result[0]))
                continue

            # Process segments if they exist
            for idx, segment in enumerate(input_config.get('segments', []), 1):
                if not isinstance(segment, dict):
                    continue

                params = {**config, **input_config}
                params.pop('inputs', None)
                params.pop('input', None)
                params.pop('segments', None)
                params['verbose'] = verbose

                # Make timestamp optional
                if 'start_time' in segment and 'end_time' in segment:
                    params['timestamp'] = parse_timestamp(
                        segment['start_time'],
                        segment['end_time']
                    )
                else:
                    params['timestamp'] = None

                params['output_wav'] = segment.get('output_wav', '')

                result = process_input(
                    input_path if not input_path.startswith(("http://", "https://", "www.")) else None,
                    input_path if input_path.startswith(("http://", "https://", "www.")) else "",
                    **params
                )

                if result[0] and result[3]:
                    title = segment.get('title', f"Segment {idx}" if params['timestamp'] else result[3])
                    outputs.append((title, result[1] or result[0]))

    return outputs


def is_video_audio_or_url(file_path, url):
    """Check if input is video, audio, or URL"""
    if url and url.strip():
        return True
    
    if file_path:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm')
        audio_extensions = ('.mp3', '.flac', '.aac', '.ogg', '.m4a', '.opus')
        return file_path.lower().endswith(video_extensions + audio_extensions)
    
    return False


def validate_transcription_args(args):
    """Validate that transcription-related arguments are only used with video/audio/URL inputs"""
    # Check if input is video, audio, or URL
    is_media_input = is_video_audio_or_url(args.input, "")
    
    # Check for transcription-related arguments
    transcription_args = []
    if hasattr(args, 'transcribe_model') and args.transcribe_model != 'large-v3':
        transcription_args.append('--transcribe-model')
    if hasattr(args, 'multi_language') and args.multi_language:
        transcription_args.append('--multi-language')
    if hasattr(args, 'transcribe_lang') and args.transcribe_lang:
        transcription_args.append('--transcribe-lang')
    
    # If transcription arguments are used with non-media input, show error
    if transcription_args and not is_media_input:
        print(f"Error: The following options can only be used with video, audio, or URL inputs: {', '.join(transcription_args)}")
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
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'llm': args.llm or config.get('llm', ''),
        'chunk_length': args.chunk_length or config.get('chunk_length', 20),
        'max_tokens': args.max_tokens or config.get('max_tokens', 130000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'lang': args.lang or config.get('lang', 'Chinese'),
        'subcommand': 'rewrite',
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'output_wav': args.output_wav or config.get('output_wav', ''),
        'cite_timestamps': args.cite_timestamps or config.get('cite_timestamps', False),
        'verbose': args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != 'verbose':
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params['timestamp'] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(f"Processing timestamp segment: {args.start_time} - {args.end_time}")
    else:
        params['timestamp'] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input,
        args.input if is_url else "",
        **params
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
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'llm': args.llm or config.get('llm', ''),
        'chunk_length': args.chunk_length or config.get('chunk_length', 20),
        'max_tokens': args.max_tokens or config.get('max_tokens', 130000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'lang': args.lang or config.get('lang', 'Chinese'),
        'subcommand': 'translate',
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'output_wav': args.output_wav or config.get('output_wav', ''),
        'cite_timestamps': args.cite_timestamps or config.get('cite_timestamps', False),
        'verbose': args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != 'verbose':
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params['timestamp'] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(f"Processing timestamp segment: {args.start_time} - {args.end_time}")
    else:
        params['timestamp'] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input,
        args.input if is_url else "",
        **params
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
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'llm': args.llm or config.get('llm', ''),
        'chunk_length': args.chunk_length or config.get('chunk_length', 20),
        'max_tokens': args.max_tokens or config.get('max_tokens', 130000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'lang': args.lang or config.get('lang', 'English'),
        'subcommand': 'academic',
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'output_wav': args.output_wav or config.get('output_wav', ''),
        'cite_timestamps': args.cite_timestamps or config.get('cite_timestamps', False),
        'verbose': args.verbose,
    }

    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != 'verbose':
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params['timestamp'] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(f"Processing timestamp segment: {args.start_time} - {args.end_time}")
    else:
        params['timestamp'] = None

    # Use the new process_input function that handles all file types
    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input,
        args.input if is_url else "",
        **params
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
    subparser.add_argument("--config", "-c", default="", help="Path to YAML configuration file")
    subparser.add_argument("--output-dir", "-o", default="", help="Output directory (optional)")
    subparser.add_argument("--llm", default="", help="LLM model identifier (optional)")
    subparser.add_argument("--lang", "-l", default="", help="Target language")
    subparser.add_argument("--chunk-length", "-cl", type=int, default=20,
                         help="Number of sentences per paragraph (default: 20)")
    subparser.add_argument("--max-tokens", "-mt", type=int, default=130000,
                         help="Maximum tokens for LLM output (default: 130000)")
    subparser.add_argument("--timeout", "-to", type=int, default=3600,
                         help="LLM request timeout in seconds (default: 3600)")
    subparser.add_argument("--temperature", "-tm", type=float, default=0.1,
                         help="LLM temperature parameter (default: 0.1)")
    # Transcription-related arguments (only for video/audio/URL inputs)
    subparser.add_argument("--transcribe-model", "-tsm", default="large-v3",
                         choices=["tiny", "base", "small", "medium", "large-v1", "large-v2",
                                 "large-v3", "large-v3-turbo", "turbo"],
                         help="Whisper model size for transcription (default: large-v3)")
    subparser.add_argument("--multi-language", "-m", action="store_true", 
                         help="Enable multi-language processing (video/audio/URL only)")
    subparser.add_argument("--transcribe-lang", "-s", default="", 
                         help="Transcribe language (video/audio/URL only)")
    subparser.add_argument("--output-wav", "-ow", default="", 
                         help="Filename for saving the segmented WAV (optional)")
    subparser.add_argument("--start-time", "-st", default="", 
                         help="Start time for extraction (format: HH:MM:SS)")
    subparser.add_argument("--end-time", "-et", default="", 
                         help="End time for extraction (format: HH:MM:SS)")
    subparser.add_argument("--cite-timestamps", action="store_true", default=False,
                         help="Include timestamps as headers in markdown output for traceability")
    subparser.add_argument("--verbose", "-v", action="store_true", default=False,
                         help="Enable verbose output showing processing details")


def is_markdown_file(file_path):
    """Check if file is markdown format"""
    if not file_path:
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ['.md', '.markdown']


def handle_ppt_command(args):
    """Handle ppt subcommand - process video slides (default) or combine speech with presentation slides (backward compatibility)"""
    logger = setup_logging(args.verbose)
    
    if args.verbose:
        logger.debug("Starting ppt command")
        logger.debug(f"Input: {args.input}")
        
    # Check if using video slides mode
    if not args.video_slides and not args.slides_file:
        print("Error: Either provide a slides file OR use --video-slides option")
        sys.exit(1)
        
    # For video slides mode (new default behavior)
    if getattr(args, 'video_mode', True) and not getattr(args, 'slides_mode', False):
        from wenbi.video_slides import process_video_slides, validate_video_input
        
        if args.verbose:
            logger.debug("Video slides mode enabled")
        
        # Validate video input
        if not args.input:
            print("Error: Video file is required for video slides mode")
            sys.exit(1)
        
        is_url = args.input.startswith(("http://", "https://", "www."))
        if is_url:
            print("Error: Video URLs not supported yet for video slides mode. Please use local video file.")
            sys.exit(1)
        
        if not validate_video_input(args.input, logger, args.verbose):
            print(f"Error: Invalid video file: {args.input}")
            sys.exit(1)
        
        # Prepare parameters
        output_dir = args.output_dir or os.getcwd()
        roi_coords = None
        if args.manual_roi:
            from wenbi.video_slides import manual_roi_override
            roi_coords = manual_roi_override(args.input, logger, args.verbose)
        
        # Parse time range (default to first hour)
        start_time = getattr(args, 'slides_start_time', "00:00:00") or "00:00:00"
        end_time = getattr(args, 'slides_end_time', "01:00:00")
        
        try:
            # Process video slides with time range
            output_file = process_video_workflow(
                input_path=args.input,
                output_dir=output_dir,
                cite_timestamps=args.cite_timestamps,
                start_time=start_time,
                end_time=end_time,
                logger=logger,
                verbose=args.verbose
            )
            
            print("Video slides processing completed successfully!")
            print(f"Output file: {output_file}")
            
        except Exception as e:
            print(f"Error processing video slides: {e}")
            if args.verbose:
                logger.exception("Detailed error trace:")
            sys.exit(1)
            
        return
    
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
        if slides_ext not in ['.pdf', '.pptx']:
            print(f"Error: Slides file must be PDF, PPTX, or markdown file, got {slides_ext}")
            sys.exit(1)
    
    if not os.path.isfile(args.slides_file):
        print(f"Error: Slides file not found: {args.slides_file}")
        sys.exit(1)
    
    # Check if slides_file is markdown (no format validation needed for markdown)
    slides_is_markdown = is_markdown_file(args.slides_file)
    
    # Validate slides file format if not markdown
    if not slides_is_markdown:
        slides_ext = os.path.splitext(args.slides_file)[1].lower()
        if slides_ext not in ['.pdf', '.pptx']:
            print(f"Error: Slides file must be PDF, PPTX, or markdown file, got {slides_ext}")
            sys.exit(1)
    
    # Validate transcription arguments
    validate_transcription_args(args)
    
    # Load config if provided
    config = load_config(args.config)
    
    # Prepare parameters for speech processing (rewrite)
    params = {
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'llm': args.llm or config.get('llm', ''),
        'chunk_length': args.chunk_length or config.get('chunk_length', 20),
        'max_tokens': args.max_tokens or config.get('max_tokens', 130000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'lang': args.lang or config.get('lang', 'Chinese'),
        'subcommand': 'rewrite',  # Always use rewrite for speech processing
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'output_wav': args.output_wav or config.get('output_wav', ''),
        'cite_timestamps': args.cite_timestamps or config.get('cite_timestamps', False),
        'verbose': args.verbose,
    }
    
    if args.verbose:
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != 'verbose':
                logger.debug(f"  {key}: {value}")
    
    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params['timestamp'] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(f"Processing timestamp segment: {args.start_time} - {args.end_time}")
    else:
        params['timestamp'] = None
    
    try:
        # Determine if speech_input is markdown
        speech_is_markdown = is_markdown_file(args.input)
        
        # Step 1: Process speech input
        if speech_is_markdown:
            if args.verbose:
                logger.debug("Step 1: Reading speech markdown file (skipping rewrite subcommand)")
            
            try:
                speech_markdown, _ = read_markdown_file(args.input, verbose=args.verbose)
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
                None if is_url else args.input,
                args.input if is_url else "",
                **params
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
            logger.debug(f"Step 2: Processing slides {'(reading markdown)' if slides_is_markdown else '(converting to markdown)'}")
        
        output_dir = params['output_dir'] or os.getcwd()
        
        if slides_is_markdown:
            try:
                slides_markdown, _ = read_markdown_file(args.slides_file, verbose=args.verbose)
                slides_file = None  # Mark as input, not generated
            except Exception as e:
                print(f"Error reading slides markdown: {e}")
                sys.exit(1)
        else:
            slides_markdown, slides_file = convert_slides_to_markdown(
                args.slides_file,
                output_dir=output_dir,
                image_export_mode=args.image_export_mode or config.get('image_export_mode', 'embedded'),
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
                llm=params['llm'],
                output_dir=output_dir,
                cite_timestamps=params['cite_timestamps'],
                max_tokens=params['max_tokens'],
                timeout=params['timeout'],
                temperature=params['temperature'],
                verbose=args.verbose,
            )
        else:
            if args.verbose:
                logger.debug("Using original LLM-based alignment")
            combined_markdown = combine_speech_and_slides(
                speech_markdown,
                slides_markdown,
                llm=params['llm'],
                output_dir=output_dir,
                cite_timestamps=params['cite_timestamps'],
                max_tokens=params['max_tokens'],
                timeout=params['timeout'],
                temperature=params['temperature'],
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
    download_all()

    # Check if this is a subcommand
    subcommands = ['rewrite', 'rw', 'translate', 'tr', 'academic', 'ac', 'ppt', 'p']
    is_subcommand = len(sys.argv) > 1 and sys.argv[1] in subcommands

    if is_subcommand:
        # Create parser for subcommands only
        parser = argparse.ArgumentParser(
            description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
        )
        
        # Add subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

        # Rewrite subcommand
        rewrite_parser = subparsers.add_parser('rewrite', aliases=['rw'], help='Rewrite text')
        add_global_args(rewrite_parser)
        rewrite_parser.set_defaults(func=handle_rewrite_command)

        # Translate subcommand
        translate_parser = subparsers.add_parser('translate', aliases=['tr'], help='Translate text')
        add_global_args(translate_parser)
        translate_parser.set_defaults(func=handle_translate_command)

        # Academic subcommand
        academic_parser = subparsers.add_parser('academic', aliases=['ac'], help='Academic rewriting')
        add_global_args(academic_parser)
        academic_parser.set_defaults(func=handle_academic_command)

        # PPT subcommand - combine speech and slides, or extract slides from video
        ppt_parser = subparsers.add_parser('ppt', aliases=['p'], help='Combine speech with presentation slides, or extract slides from video')
        add_global_args(ppt_parser)
        
        # Input: video file by default, or slides file for backward compatibility
        ppt_parser.add_argument("input", help="Video file/URL or slides file (PDF or PPTX)")
        ppt_parser.add_argument("--video-mode", "-vm", action="store_true", default=True,
                        help="Process video input (default: enabled)")
        ppt_parser.add_argument("--slides-mode", "-sm", action="store_true", default=False,
                        help="Process slides file (backward compatibility)")
        
        ppt_parser.add_argument("--image-export-mode", "-iem", default="embedded",
                              choices=["none", "embedded", "referenced"],
                              help="Image export mode for slides (default: embedded)")
        ppt_parser.add_argument("--enhanced-alignment", "-ea", action="store_true", default=False,
                              help="Use enhanced similarity-based alignment instead of LLM alignment")
        ppt_parser.add_argument("--manual-roi", "-mroi", action="store_true", default=False,
                              help="Manually specify ROI coordinates instead of automatic detection")
        ppt_parser.add_argument("--slides-start-time", "-sst", default="",
                              help="Start time for slides extraction (format: HH:MM:SS, default: 00:00:00)")
        ppt_parser.add_argument("--slides-end-time", "-set", default="01:00:00",
                              help="End time for slides extraction (format: HH:MM:SS, default: 01:00:00, auto-adjusts if video shorter)")
        ppt_parser.set_defaults(func=handle_ppt_command)

        args = parser.parse_args()
        args.func(args)
        return

    # Main command (direct file processing)
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
    )
    parser.add_argument("input", nargs="?", default="", help="Path to input file or URL")
    parser.add_argument("--config", "-c", default="", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", "-o", default="", help="Output directory (optional)")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--llm", default="", help="LLM model identifier (optional)")
    parser.add_argument("--transcribe-lang", "-s", default="", help="Transcribe language (optional)")
    parser.add_argument("--lang", "-l", default="Chinese", help="Target language (default: Chinese)")
    parser.add_argument("--multi-language", "-m", action="store_true", help="Enable multi-language processing")
    parser.add_argument("--chunk-length", "-cl", type=int, default=8,
                       help="Number of sentences per paragraph (default: 8)")
    parser.add_argument("--max-tokens", "-mt", type=int, default=130000,
                       help="Maximum tokens for LLM output (default: 130000)")
    parser.add_argument("--timeout", "-to", type=int, default=3600,
                       help="LLM request timeout in seconds (default: 3600)")
    parser.add_argument("--temperature", "-tm", type=float, default=0.1,
                       help="LLM temperature parameter (default: 0.1)")
    parser.add_argument("--transcribe-model", "-tsm", default="large-v3-turbo",
                       choices=["tiny", "base", "small", "medium", "large-v1", "large-v2",
                               "large-v3", "large-v3-turbo", "turbo"],
                       help="Whisper model size for transcription (default: large-v3-turbo)")
    parser.add_argument("--output_wav", "-ow", default="", help="Filename for saving the segmented WAV (optional)")
    parser.add_argument("--start_time", "-st", default="", help="Start time for extraction (format: HH:MM:SS)")
    parser.add_argument("--end_time", "-et", default="", help="End time for extraction (format: HH:MM:SS)")
    parser.add_argument("--cite-timestamps", action="store_true", default=False,
                       help="Include timestamps as headers in markdown output for traceability")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                       help="Enable verbose output showing processing details")

    args = parser.parse_args()

    # Setup logging for main command
    logger = setup_logging(args.verbose)

    # Handle config file processing for main command
    if args.config:
        if args.verbose:
            logger.debug(f"Loading configuration from: {args.config}")
        
        if not args.config.endswith(('.yml', '.yaml')):
            print("Error: Config file must be a YAML file")
            sys.exit(1)

        config = load_config(args.config)
        if not isinstance(config, dict):
            print("Error: Invalid YAML configuration")
            sys.exit(1)

        # Add verbose to config if specified in command line
        if args.verbose:
            config['verbose'] = True

        outputs = process_yaml_config(config, args.verbose)

        if outputs:
            output_dir = config.get('output_dir', '')
            final_output = combine_markdown_files(outputs, output_dir, verbose=args.verbose)
            print(f"Combined output saved to: {final_output}")
        return

    # Load config file if provided
    config = load_config(args.config)

    # Command line arguments take precedence over config file
    params = {
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'llm': args.llm or config.get('llm', ''),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'lang': args.lang or config.get('lang', 'Chinese'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'chunk_length': args.chunk_length or config.get('chunk_length', 20),
        'max_tokens': args.max_tokens or config.get('max_tokens', 130000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3-turbo'),
        'output_wav': args.output_wav or config.get('output_wav', ''),
        'cite_timestamps': args.cite_timestamps or config.get('cite_timestamps', False),
        'verbose': args.verbose or config.get('verbose', False),
    }

    if args.verbose:
        logger.debug("Starting main command processing")
        logger.debug("Configuration:")
        for key, value in params.items():
            if key != 'verbose':
                logger.debug(f"  {key}: {value}")

    # Handle timestamp parameters
    if args.start_time and args.end_time:
        params['timestamp'] = parse_timestamp(args.start_time, args.end_time)
        if args.verbose:
            logger.debug(f"Processing timestamp segment: {args.start_time} - {args.end_time}")
    else:
        params['timestamp'] = None

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
        None if is_url else args.input,
        args.input if is_url else "",
        **params
    )
    
    print("Markdown Output:", result[0])
    print("Markdown File:", result[1])
    print("CSV File:", result[2])
    print("Filename (without extension):", result[3] if result[3] is not None else "")


if __name__ == "__main__":
    main()
