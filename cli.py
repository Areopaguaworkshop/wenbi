#!/usr/bin/env python3
import argparse
import os
from main import process_input
import dspy

def main():
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, url or subtitle files to CSV and written Markdown outputs."
    )
    parser.add_argument("input", help="Path to input file or URL")
    parser.add_argument("--language", default="", help="Transcribe Language (optional)")
    parser.add_argument("--llm", default="", help="Large Language Model identifier (optional)")
    # Store_true automatically sets the flag to True when provided.
    parser.add_argument("--multi-language", action="store_true", default=False, 
                       help="Enable multi-language processing (default: False)")
    parser.add_argument("--translate-lang", default="Chinese", 
                       help="Target translation language (default: Chinese)")
    parser.add_argument("--output-dir", default="", 
                       help="Output directory (optional)")
    args = parser.parse_args()

    # Debug: print the multi_language flag
    print(f"Multi-language flag is set to: {args.multi_language}")

    # Patch dspy.LM.__init__
    default_model = args.llm.strip() if args.llm.strip() else "ollama/qwen2.5"
    orig_init = dspy.LM.__init__
    def new_init(self, *a, **kw):
        kw["model"] = str(default_model)
        orig_init(self, *a, **kw)
    dspy.LM.__init__ = new_init

    # Detect if input is URL or file
    is_url = args.input.startswith(('http://', 'https://', 'www.'))
    result = process_input(
                None if is_url else args.input,  # file_path
                args.input if is_url else "",    # url
                args.language,
                args.llm,
                args.multi_language,
                args.translate_lang,
                args.output_dir
             )
    print("Markdown Output:", result[0])
    print("Markdown File:", result[1])
    print("CSV File:", result[2])
    print("Filename (without extension):", result[3] if result[3] is not None else "")

if __name__ == "__main__":
    main()