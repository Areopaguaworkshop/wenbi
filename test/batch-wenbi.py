import os
import subprocess
import argparse
from pathlib import Path
from natsort import natsorted  # Add this import
import sys

def is_media_file(filename):
    """Check if file is video or audio"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
    audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    ext = Path(filename).suffix.lower()
    return ext in video_extensions or ext in audio_extensions

def process_media_file(filepath):
    """Process single media file with wenbi"""
    try:
        wenbi_dir = os.path.dirname(os.path.dirname(__file__))
        env = os.environ.copy()
        env["PYTHONPATH"] = wenbi_dir

        # Try importing the module directly
        result = subprocess.run([sys.executable, '-c',
                               'from wenbi.__main__ import main; main(["{0}"])'.format(filepath)],
                              capture_output=True,
                              text=True,
                              cwd=wenbi_dir,
                              env=env)
        if result.returncode != 0:
            print(f"Error processing {filepath}")
            print(f"Exit code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute wenbi: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Batch process media files with wenbi')
    parser.add_argument('input_dir', help='Input directory containing media files')
    parser.add_argument('output_file', help='Output markdown file')
    args = parser.parse_args()

    # Get all media files and sort them naturally
    media_files = natsorted([f for f in os.listdir(args.input_dir) 
                            if is_media_file(f)])

    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        for media_file in media_files:
            filepath = os.path.join(args.input_dir, media_file)
            
            # Write filename as header
            out_file.write(f"## {media_file}\n\n")
            
            # Process file and write content
            content = process_media_file(filepath)
            if content:
                out_file.write(content)
                out_file.write('\n\n')

if __name__ == '__main__':
    main()
