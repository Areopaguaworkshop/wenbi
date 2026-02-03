#!/usr/bin/env python3
"""
Simple script to extract a clip from a video file between start and end times.
Requires ffmpeg to be installed.

Usage:
    python extract_clip.py input.mp4
    python extract_clip.py input.mp4 --start-time 00:01:30 --end-time 00:05:00
    python extract_clip.py input.mp4 --start-time 10 --end-time 60  # in seconds
"""

import argparse
import os
import subprocess
import sys


def extract_time_to_seconds(time_str):
    """Convert time string to seconds. Accepts HH:MM:SS or just seconds."""
    try:
        # Try parsing as integer seconds
        return int(time_str)
    except ValueError:
        pass
    
    # Parse HH:MM:SS format
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    except Exception as e:
        raise ValueError(f"Could not parse time '{time_str}': {e}")


def seconds_to_hhmmss(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_clip(input_file, start_time, end_time, output_file=None):
    """
    Extract a clip from a video file using ffmpeg.
    
    Args:
        input_file: Path to input video file
        start_time: Start time as HH:MM:SS or seconds
        end_time: End time as HH:MM:SS or seconds
        output_file: Optional output file path (defaults to input_clip.mp4)
    
    Returns:
        Path to output file if successful, None otherwise
    """
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return None
    
    # Parse times
    try:
        start_seconds = extract_time_to_seconds(start_time)
        end_seconds = extract_time_to_seconds(end_time)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    # Validate times
    if start_seconds < 0:
        print("Error: Start time cannot be negative")
        return None
    
    if end_seconds <= start_seconds:
        print("Error: End time must be greater than start time")
        return None
    
    duration = end_seconds - start_seconds
    
    # Generate output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        ext = os.path.splitext(input_file)[1]
        start_str = seconds_to_hhmmss(start_seconds).replace(":", "-")
        end_str = seconds_to_hhmmss(end_seconds).replace(":", "-")
        output_file = f"{base_name}_clip_{start_str}_to_{end_str}{ext}"
    
    start_hhmmss = seconds_to_hhmmss(start_seconds)
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ss", start_hhmmss,
        "-t", str(duration),
        "-c", "copy",  # Copy codec without re-encoding for speed
        "-y",  # Overwrite output file
        output_file,
    ]
    
    print(f"Extracting clip from {input_file}...")
    print(f"  Start time: {start_hhmmss} ({start_seconds}s)")
    print(f"  Duration: {seconds_to_hhmmss(duration)} ({duration}s)")
    print(f"  Output: {output_file}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"\n✓ Clip extracted successfully!")
            print(f"  Output file: {output_file} ({file_size:.1f} MB)")
            return output_file
        else:
            print("Error: Output file was not created")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg command failed with return code {e.returncode}")
        print("Make sure ffmpeg is installed: https://ffmpeg.org/download.html")
        return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract a clip from a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract first 10 minutes:
    python extract_clip.py video.mp4

  Extract from 1:30 to 5:00:
    python extract_clip.py video.mp4 --start-time 00:01:30 --end-time 00:05:00

  Extract from 60 to 300 seconds:
    python extract_clip.py video.mp4 --start-time 60 --end-time 300

  Extract with custom output name:
    python extract_clip.py video.mp4 -o my_clip.mp4
        """,
    )
    
    parser.add_argument(
        "input",
        help="Input video file path",
    )
    
    parser.add_argument(
        "--start-time",
        default="00:00:00",
        help="Start time in HH:MM:SS or seconds (default: 00:00:00)",
    )
    
    parser.add_argument(
        "--end-time",
        default="00:10:00",
        help="End time in HH:MM:SS or seconds (default: 00:10:00)",
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (optional)",
    )
    
    args = parser.parse_args()
    
    result = extract_clip(args.input, args.start_time, args.end_time, args.output)
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
