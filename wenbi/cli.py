#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from wenbi.main import process_input
from wenbi.download import download_all

def main():
    download_all()
    # Run download_all() to ensure necessary models are present
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
    )
    parser.add_argument("input", nargs="?", default="", help="Path to input file or URL")
    parser.add_argument("--output-dir", "-o", default="", help="Output directory (optional)")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch Gradio GUI")
    parser.add_argument("--rewrite-llm", "-rlm", default="", help="Rewrite LLM model identifier (optional)")
    parser.add_argument("--translate-llm", "-tlm", default="", help="Translation LLM model identifier (optional)")
    parser.add_argument("--transcribe-lang", "-s", default="", help="Transcribe language (optional)")
    parser.add_argument("--translate-lang", "-t", default="Chinese", help="Target translation language (default: Chinese)")
    parser.add_argument("--rewrite-lang", "-r", default="Chinese", help="Target language for rewriting (default: Chinese)")
    parser.add_argument("--multi-language", "-m", action="store_true", help="Enable multi-language processing")
    parser.add_argument("--chunk-length", "-c", type=int, default=8, help="the chunk of Number of sentences per paragraph for llm to tranlsate or rewrite. (default: 8)")
    # Add new LLM configuration options
    parser.add_argument("--max-tokens","-mt", type=int, default=50000, help="Maximum tokens for LLM output (default: 50000)")
    parser.add_argument("--timeout", "-to", type=int, default=3600, help="LLM request timeout in seconds (default: 3600)")
    parser.add_argument("--temperature", "-tm", type=float, default=0.1, help="LLM temperature parameter (default: 0.1)")
    parser.add_argument("--base-url", "-u", default="http://localhost:11434", help="Base URL for LLM API (default: http://localhost:11434)")
    args = parser.parse_args()

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
    
    is_url = args.input.startswith(('http://', 'https://', 'www.'))
    result = process_input(
        None if is_url else args.input,
        args.input if is_url else "",
        args.transcribe_lang,
        args.rewrite_llm,  # Explicitly pass rewrite model
        args.translate_llm,  # Explicitly pass translate model
        args.multi_language,
        args.translate_lang,
        args.output_dir,
        args.rewrite_lang,  # Add rewrite_lang parameter
        args.chunk_length,  # Add chunk_length parameter
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        temperature=args.temperature,
        base_url=args.base_url  # Add base_url parameter
    )
    print("Markdown Output:", result[0])
    print("Markdown File:", result[1])
    print("CSV File:", result[2])
    print("Filename (without extension):", result[3] if result[3] is not None else "")

if __name__ == "__main__":
    main()
