#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import yaml
from wenbi.main import process_input
from wenbi.download import download_all


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def combine_markdown_files(outputs, output_dir, final_filename="combined_output.md"):
    """Combine multiple markdown outputs into a single file"""
    combined_path = os.path.join(output_dir, final_filename)
    with open(combined_path, 'w', encoding='utf-8') as f:
        for idx, (title, content) in enumerate(outputs):
            if idx > 0:
                f.write('\n---\n\n')  # Separator between sections
            f.write(f'# {title}\n\n')
            # Read and append content from markdown file
            if os.path.isfile(content):
                with open(content, 'r', encoding='utf-8') as mf:
                    f.write(mf.read())
            else:
                f.write(content)
    return combined_path


def parse_timestamp(start_time, end_time):
    """Parse start and end time strings in the format HH:MM:SS"""
    try:
        return {'start': start_time.strip(), 'end': end_time.strip()}
    except ValueError:
        print("Error: Invalid timestamp format. Use HH:MM:SS for both start and end times.")
        sys.exit(1)


def process_yaml_config(config):
    """Process YAML config supporting both single and multiple input formats"""
    outputs = []
    
    # Handle single input with segments
    if 'input' in config and 'segments' in config:
        input_path = config['input']
        params = {**config}  # Remove hardcoded output_wav
        params.pop('input', None)
        params.pop('segments', None)
        
        for segment in config['segments']:
            # Process each segment
            params['timestamp'] = parse_timestamp(
                segment['start_time'],
                segment['end_time']
            )
            params['output_wav'] = segment.get('output_wav', '')  # Get output_wav from segment
            
            result = process_input(
                input_path if not input_path.startswith(("http://", "https://", "www.")) else None,
                input_path if input_path.startswith(("http://", "https://", "www.")) else "",
                **params
            )
            
            if result[0] and result[3]:
                outputs.append((segment.get('title', result[3]), result[1] or result[0]))
        
        # Combine outputs into single file
        if outputs:
            output_dir = config.get('output_dir', '')
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            final_output = combine_markdown_files(outputs, output_dir, f"{base_name}_combined.md")
            print(f"Combined output saved to: {final_output}")
            
    # Handle multiple inputs with segments
    if 'inputs' in config:
        for input_config in config['inputs']:
            input_path = input_config['input']
            for segment in input_config.get('segments', []):
                params = {**config, **input_config}
                params.pop('inputs', None)
                params.pop('input', None)
                params.pop('segments', None)
                params['timestamp'] = parse_timestamp(
                    segment['start_time'],
                    segment['end_time']
                )
                
                result = process_input(
                    input_path if not input_path.startswith(("http://", "https://", "www.")) else None,
                    input_path if input_path.startswith(("http://", "https://", "www.")) else "",
                    **params
                )
                
                if result[0] and result[3]:
                    outputs.append((segment.get('title', result[3]), result[1] or result[0]))
    
    return outputs


def main():
    download_all()
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
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
    parser.add_argument("--gui", "-g", action="store_true",
                        help="Launch Gradio GUI")
    parser.add_argument(
        "--rewrite_llm",
        default="",
        help="Rewrite LLM model identifier (optional)",
    )
    parser.add_argument(
        "--translate_llm",
        default="",
        help="Translation LLM model identifier (optional)",
    )
    parser.add_argument(
        "--transcribe-lang", "-s", default="", help="Transcribe language (optional)"
    )
    parser.add_argument(
        "--translate-lang",
        "-t",
        default="Chinese",
        help="Target translation language (default: Chinese)",
    )
    parser.add_argument(
        "--rewrite-lang",
        "-r",
        default="Chinese",
        help="Target language for rewriting (default: Chinese)",
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
        help="the chunk of Number of sentences per paragraph for llm to tranlsate or rewrite. (default: 8)",
    )
    parser.add_argument(
        "--max-tokens",
        "-mt",
        type=int,
        default=50000,
        help="Maximum tokens for LLM output (default: 50000)",
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
        "--base-url",
        "-u",
        default="http://localhost:11434",
        help="Base URL for LLM API (default: http://localhost:11434)",
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
        help="Whisper model size for transcription (default: large-v3)",
    )
    parser.add_argument(
        "--output_wav",
        "-ow",
        default="",
        help="Filename for saving the segmented WAV (optional)",
    )
    parser.add_argument(
        "--start_time", "-st",
        default="",
        help="Start time for extraction (format: HH:MM:SS)"
    )
    parser.add_argument(
        "--end_time", "-et",
        default="",
        help="End time for extraction (format: HH:MM:SS)"
    )
    args = parser.parse_args()

    # Handle config file processing
    if args.config:
        if not args.config.endswith(('.yml', '.yaml')):
            print("Error: Config file must be a YAML file")
            sys.exit(1)

        config = load_config(args.config)
        if not isinstance(config, dict):
            print("Error: Invalid YAML configuration")
            sys.exit(1)

        outputs = process_yaml_config(config)
        
        if outputs:
            output_dir = config.get('output_dir', '')
            final_output = combine_markdown_files(outputs, output_dir)
            print(f"Combined output saved to: {final_output}")
        return

    # Load config file if provided
    config = load_config(args.config)

    # Command line arguments take precedence over config file
    params = {
        'output_dir': args.output_dir or config.get('output_dir', ''),
        'rewrite_llm': args.rewrite_llm or config.get('rewrite_llm', ''),
        'translate_llm': args.translate_llm or config.get('translate_llm', ''),
        'transcribe_lang': args.transcribe_lang or config.get('transcribe_lang', ''),
        'translate_lang': args.translate_lang or config.get('translate_lang', 'Chinese'),
        'rewrite_lang': args.rewrite_lang or config.get('rewrite_lang', 'Chinese'),
        'multi_language': args.multi_language or config.get('multi_language', False),
        'chunk_length': args.chunk_length or config.get('chunk_length', 8),
        'max_tokens': args.max_tokens or config.get('max_tokens', 50000),
        'timeout': args.timeout or config.get('timeout', 3600),
        'temperature': args.temperature or config.get('temperature', 0.1),
        'base_url': args.base_url or config.get('base_url', 'http://localhost:11434'),
        'transcribe_model': args.transcribe_model or config.get('transcribe_model', 'large-v3-turbo'),
        'timestamp': parse_timestamp(args.start_time, args.end_time) if args.start_time and args.end_time else None,
        'output_wav': args.output_wav or config.get('output_wav', ''),
    }

    # If --gui is specified, run main.py to launch the GUI
    if args.gui:
        # Compute the absolute path of main.py, assumed to be in the same folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_py = os.path.join(current_dir, "main.py")
        subprocess.run(["python", main_py])
        return

    # Otherwise, run CLI mode (input must be provided)
    if not args.input:
        print("Error: Please specify an input file or URL.")
        sys.exit(1)

    is_url = args.input.startswith(("http://", "https://", "www."))
    result = process_input(
        None if is_url else args.input,
        args.input if is_url else "",
        **params
    )
    print("Markdown Output:", result[0])
    print("Markdown File:", result[1])
    print("CSV File:", result[2])
    print("Filename (without extension):",
          result[3] if result[3] is not None else "")


if __name__ == "__main__":
    main()
