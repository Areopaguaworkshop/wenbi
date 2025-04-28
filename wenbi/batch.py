import os
from pathlib import Path
from natsort import natsorted
import argparse
from wenbi.cli import main as wenbi_main
import sys
import glob
import yaml

def is_media_file(filepath: Path) -> bool:
    """Check if file is video or audio"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    return filepath.suffix.lower() in video_extensions | audio_extensions

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def combine_markdown_files(output_dir: str, combined_file: str):
    """Combine all markdown files in output directory into one file"""
    md_files = glob.glob(os.path.join(output_dir, "*_rewritten.md")) + \
               glob.glob(os.path.join(output_dir, "*_bilingual.md"))
    md_files = natsorted(md_files, key=lambda x: os.path.basename(x))
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for md_file in md_files:
            filename = os.path.basename(md_file)
            outfile.write(f"## {filename}\n\n")
            with open(md_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write("\n\n---\n\n")

def batch_process(input_dir: str, output_dir: str = "", md_output: str = None, **kwargs):
    """Process all media files in directory using wenbi"""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return

    # Create default output directory based on input folder name
    if not output_dir:
        folder_name = os.path.basename(input_path)
        output_dir = os.path.join(os.getcwd(), f"{folder_name}_output")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all media files and sort them naturally
    media_files = [f for f in input_path.iterdir() if is_media_file(f)]
    sorted_files = natsorted(media_files, key=lambda x: x.name)

    for file in sorted_files:
        print(f"\nProcessing: {file.name}")
        # Prepare wenbi arguments
        sys.argv = [
            "wenbi",
            str(file),
            "--output-dir", output_dir or str(input_path),
        ]
        
        # Add any additional arguments from kwargs
        for key, value in kwargs.items():
            if value is not None and key != 'md_output':
                if isinstance(value, bool):
                    if value:  # Only add flag if True
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])

        try:
            wenbi_main()
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Combine markdown files if md_output is specified
    if md_output:
        print("\nCombining markdown files...")
        combine_markdown_files(output_dir or str(input_path), md_output)
        print(f"Combined markdown file created: {md_output}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch process media files in a directory using wenbi"
    )
    parser.add_argument("input_dir", help="Input directory containing media files")
    parser.add_argument("-c", "--config", help="YAML configuration file")
    # Make all other arguments optional since they might come from config
    parser.add_argument("--output-dir", default="", help="Output directory (optional)")
    parser.add_argument("--rewrite-llm", help="Rewrite LLM model identifier")
    parser.add_argument("--translate-llm", help="Translation LLM model identifier")
    parser.add_argument("--transcribe-lang", help="Transcribe language")
    parser.add_argument("--translate-lang", default="Chinese", help="Target translation language")
    parser.add_argument("--rewrite-lang", default="Chinese", help="Target language for rewriting")
    parser.add_argument("--multi-language", action="store_true", help="Enable multi-language processing")
    parser.add_argument("--chunk-length", type=int, default=8, help="Number of sentences per chunk")
    parser.add_argument("--max-tokens", type=int, default=50000, help="Maximum tokens for LLM")
    parser.add_argument("--timeout", type=int, default=3600, help="LLM timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--md", nargs='?', const='default', help="Output combined markdown file path. If no path provided, uses input folder name.")
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        # Remove config from kwargs to avoid passing it to batch_process
        kwargs.pop('config')
        # Update kwargs with config values, preserving command-line arguments
        for key, value in config.items():
            if key in kwargs and kwargs[key] is None:
                kwargs[key] = value
    
    # Set default markdown filename based on input folder name
    md_output = kwargs.pop('md')
    if md_output == 'default':
        folder_name = os.path.basename(os.path.abspath(args.input_dir))
        md_output = f"{folder_name}.md"
    
    batch_process(**kwargs, md_output=md_output)

if __name__ == "__main__":
    main()
