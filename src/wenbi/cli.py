#!/usr/bin/env python3
import argparse
from wenbi.main import process_input  # main.py is in the same directory now

def main():
    parser = argparse.ArgumentParser(
        description="wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs."
    )
    parser.add_argument("input", help="Path or URL to the input file")
    parser.add_argument("--language", default="", help="Transcribe language (optional)")
    parser.add_argument("--rewrite-llm", default="", help="Rewrite LLM model identifier (optional)")
    parser.add_argument("--translate-llm", default="", help="Translation LLM model identifier (optional)")
    parser.add_argument("--multi-language", action="store_true", help="Enable multi-language processing")
    parser.add_argument("--translate-lang", default="Chinese", help="Target translation language (default: Chinese)")
    parser.add_argument("--output-dir", default="", help="Output directory (optional)")
    args = parser.parse_args()

    is_url = args.input.startswith(('http://', 'https://', 'www.'))
    result = process_input(
                None if is_url else args.input,   # file_path
                args.input if is_url else "",     # url
                args.language,
                args.rewrite_llm,
                args.translate_llm,
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
