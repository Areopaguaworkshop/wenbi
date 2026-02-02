import os
from pathlib import Path
from natsort import natsorted
import argparse
import logging
import sys
from wenbi.cli import main as wenbi_main
import glob
import yaml


def setup_logging(verbose=False):
    """Setup logging configuration for batch processing"""
    if verbose:
        level = logging.DEBUG
        format_str = '[BATCH-VERBOSE] %(message)s'
    else:
        level = logging.INFO
        format_str = '%(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def is_media_file(filepath: Path) -> bool:
    """Check if file is video or audio"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    return filepath.suffix.lower() in video_extensions | audio_extensions


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def combine_markdown_files(output_dir: str, combined_file: str, verbose=False):
    """Combine all markdown files in output directory into one file"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Combining markdown files from: {output_dir}")
    
    md_files = glob.glob(os.path.join(output_dir, "*_rewritten.md")) + \
               glob.glob(os.path.join(output_dir, "*_bilingual.md")) + \
               glob.glob(os.path.join(output_dir, "*_translated.md")) + \
               glob.glob(os.path.join(output_dir, "*_academic.md"))
    
    md_files = natsorted(md_files, key=lambda x: os.path.basename(x))
    
    if verbose:
        logger.debug(f"Found {len(md_files)} markdown files to combine")
        for i, file in enumerate(md_files, 1):
            logger.debug(f"  {i}. {os.path.basename(file)}")
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for i, md_file in enumerate(md_files, 1):
            filename = os.path.basename(md_file)
            
            if verbose:
                logger.debug(f"Adding file {i}/{len(md_files)}: {filename}")
            
            outfile.write(f"## {filename}\n\n")
            with open(md_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
            outfile.write("\n\n---\n\n")
    
    if verbose:
        logger.debug(f"Combined markdown file created: {combined_file}")


def batch_process(input_dir: str, output_dir: str = "", md_output: str = None, verbose: bool = False, **kwargs):
    """Process all media files in directory using wenbi"""
    logger = setup_logging(verbose)
    
    if verbose:
        logger.debug("=== Starting Batch Processing ===")
        logger.debug(f"Input directory: {input_dir}")
        logger.debug(f"Verbose mode: {verbose}")
    
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
    
    if verbose:
        logger.debug(f"Created output directory: {output_dir}")

    # Get all media files and sort them naturally
    media_files = [f for f in input_path.iterdir() if is_media_file(f)]
    sorted_files = natsorted(media_files, key=lambda x: x.name)

    if verbose:
        logger.debug(f"Found {len(sorted_files)} media files to process")
        for i, file in enumerate(sorted_files, 1):
            logger.debug(f"  {i}. {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")

    successful_count = 0
    failed_count = 0
    
    for i, file in enumerate(sorted_files, 1):
        print(f"\nProcessing {i}/{len(sorted_files)}: {file.name}")
        
        if verbose:
            logger.debug(f"=== Processing File {i}/{len(sorted_files)} ===")
            logger.debug(f"File: {file.name}")
            logger.debug(f"Size: {file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Prepare wenbi arguments
        sys.argv = [
            "wenbi",
            str(file),
            "--output-dir", output_dir or str(input_path),
        ]
        
        # Add verbose flag if enabled
        if verbose:
            sys.argv.append("--verbose")
        
        # Add any additional arguments from kwargs
        for key, value in kwargs.items():
            if value is not None and key != 'md_output':
                if isinstance(value, bool):
                    if value:  # Only add flag if True
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])

        if verbose:
            logger.debug(f"Command arguments: {' '.join(sys.argv[1:])}")

        try:
            wenbi_main()
            successful_count += 1
            if verbose:
                logger.debug(f"Successfully processed: {file.name}")
        except Exception as e:
            failed_count += 1
            print(f"Error processing {file.name}: {e}")
            if verbose:
                logger.debug(f"Failed to process {file.name}: {e}")

    # Print batch processing summary
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total files: {len(sorted_files)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if verbose:
        logger.debug("=== Batch Processing Summary ===")
        logger.debug(f"Total files processed: {len(sorted_files)}")
        logger.debug(f"Successful: {successful_count}")
        logger.debug(f"Failed: {failed_count}")

    # Combine markdown files if md_output is specified
    if md_output:
        print("\nCombining markdown files...")
        if verbose:
            logger.debug("Starting markdown file combination")
        
        combine_markdown_files(output_dir or str(input_path), md_output, verbose)
        print(f"Combined markdown file created: {md_output}")
        
        if verbose:
            logger.debug("=== Batch Processing Completed ===")


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
    parser.add_argument("--lang", default="Chinese", help="Target translation language")
    parser.add_argument("--lang", default="Chinese", help="Target language for rewriting")
    parser.add_argument("--multi-language", action="store_true", help="Enable multi-language processing")
    parser.add_argument("--chunk-length", type=int, default=20, help="Number of sentences per chunk")
    parser.add_argument("--max-tokens", type=int, default=50000, help="Maximum tokens for LLM")
    parser.add_argument("--timeout", type=int, default=3600, help="LLM timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--md", nargs='?', const='default', help="Output combined markdown file path. If no path provided, uses input folder name.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                       help="Enable verbose output showing processing details")
    
    args = parser.parse_args()
    kwargs = vars(args)
    
    # Load config file if provided
    config = {}
    if args.config:
        if args.verbose:
            print(f"Loading configuration from: {args.config}")
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
