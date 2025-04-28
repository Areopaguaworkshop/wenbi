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
        "--rewrite-llm",
        "-rlm",
        default="",
        help="Rewrite LLM model identifier (optional)",
    )
    parser.add_argument(
        "--translate-llm",
        "-tlm",
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

        output_dir = config.get('output_dir', '')
        inputs = config.get('inputs', [])
        if not inputs:
            print("Error: No inputs specified in config file")
            sys.exit(1)

        # Process each input and collect outputs
        outputs = []
        for input_config in inputs:
            input_path = input_config.get('input', '')
            if not input_path:
                continue

            # Merge global config with input-specific config
            input_params = {**config, **input_config}
            input_params.pop('inputs', None)  # Remove inputs list from params
            input_params.pop('input', None) # Remove input from params
            input_params.pop('title', None) # Remove title from params

            is_url = input_path.startswith(("http://", "https://", "www."))
            result = process_input(
                None if is_url else input_path,
                input_path if is_url else "",
                **input_params
            )
            
            if result[0] and result[3]:  # If we have output and filename
                title = input_config.get('title', result[3])
                outputs.append((title, result[1] or result[0]))

        if outputs:
            # Combine all outputs into a single markdown file
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
