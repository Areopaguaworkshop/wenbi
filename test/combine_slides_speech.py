#!/usr/bin/env python3
"""
Combine _slide.md and _rewritten.md files based on timestamp matching.

Slides are inserted BEFORE their corresponding speech sections.
A slide with timestamp <= speech section end time gets inserted before that section.

Usage:
    python combine_slides_speech.py slides.md rewritten.md -o combined.md
    python combine_slides_speech.py slides.md rewritten.md  # output to stdout
"""

import argparse
import re
import sys


def parse_time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    try:
        parts = time_str.strip().split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        raise ValueError(f"Could not parse time '{time_str}': {e}")


def parse_slides(slides_content):
    """Extract slides from _slide.md"""
    slides_data = []
    # Pattern: ## Slide at HH:MM:SS
    slide_pattern = r"## Slide at ([\d:]+)"
    
    matches = list(re.finditer(slide_pattern, slides_content))
    
    for idx, match in enumerate(matches):
        timestamp = match.group(1)
        
        # Get content between this slide header and the next one (or end of file)
        start_pos = match.end()
        if idx < len(matches) - 1:
            end_pos = matches[idx + 1].start()
        else:
            end_pos = len(slides_content)
        
        content = slides_content[start_pos:end_pos].strip()
        
        try:
            timestamp_seconds = parse_time_to_seconds(timestamp)
            slides_data.append({
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
                "content": content,
            })
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
            continue
    
    return slides_data


def parse_speech(speech_content):
    """Extract speech sections from _rewritten.md"""
    speech_sections = []
    # Pattern: ### **HH:MM:SS - HH:MM:SS**
    section_pattern = r"###\s+\*\*(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\*\*"
    
    matches = list(re.finditer(section_pattern, speech_content))
    
    for idx, match in enumerate(matches):
        start_time = match.group(1)
        end_time = match.group(2)
        
        # Get content between this timestamp and the next one (or end of file)
        content_start = match.end()
        if idx < len(matches) - 1:
            content_end = matches[idx + 1].start()
        else:
            content_end = len(speech_content)
        
        content = speech_content[content_start:content_end].strip()
        
        try:
            end_time_seconds = parse_time_to_seconds(end_time)
            speech_sections.append({
                "start_time": start_time,
                "end_time": end_time,
                "end_time_seconds": end_time_seconds,
                "content": content,
            })
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
            continue
    
    return speech_sections


def combine_slides_and_speech(slides_data, speech_sections):
    """
    Combine slides and speech sections.
    Slides are inserted BEFORE their corresponding speech section.
    """
    combined = []
    current_slide_idx = 0
    
    for speech_section in speech_sections:
        end_seconds = speech_section["end_time_seconds"]
        
        # Insert all slides that occur before or at this section's end time
        while current_slide_idx < len(slides_data):
            slide = slides_data[current_slide_idx]
            slide_seconds = slide["timestamp_seconds"]
            
            if slide_seconds <= end_seconds:
                # Insert slide before this speech section
                combined.append(f"\n\n---\n\n## Slide at {slide['timestamp']}\n\n")
                combined.append(slide["content"])
                current_slide_idx += 1
            else:
                # This slide belongs to a later section
                break
        
        # Add the speech section
        combined.append(f"\n\n### **{speech_section['start_time']} - {speech_section['end_time']}**\n\n")
        combined.append(speech_section["content"])
    
    # Add any remaining slides (if they occur after all speech sections)
    while current_slide_idx < len(slides_data):
        slide = slides_data[current_slide_idx]
        combined.append(f"\n\n---\n\n## Slide at {slide['timestamp']}\n\n")
        combined.append(slide["content"])
        current_slide_idx += 1
    
    return "".join(combined).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Combine slide and speech content based on timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Combine and save to file:
    python combine_slides_speech.py slides.md rewritten.md -o combined.md

  Combine and print to stdout:
    python combine_slides_speech.py slides.md rewritten.md

  With verbose output:
    python combine_slides_speech.py slides.md rewritten.md -v
        """,
    )
    
    parser.add_argument("slides", help="Input slides markdown file (_slide.md)")
    parser.add_argument("speech", help="Input speech markdown file (_rewritten.md)")
    parser.add_argument(
        "-o", "--output",
        help="Output file path (optional, prints to stdout if not provided)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    # Read input files
    try:
        with open(args.slides, "r", encoding="utf-8") as f:
            slides_content = f.read()
        if args.verbose:
            print(f"✓ Read slides file: {args.slides}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Slides file not found: {args.slides}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading slides file: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(args.speech, "r", encoding="utf-8") as f:
            speech_content = f.read()
        if args.verbose:
            print(f"✓ Read speech file: {args.speech}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Speech file not found: {args.speech}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading speech file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse content
    try:
        slides_data = parse_slides(slides_content)
        if args.verbose:
            print(f"✓ Parsed {len(slides_data)} slides", file=sys.stderr)
            for slide in slides_data:
                print(f"  - Slide at {slide['timestamp']} ({slide['timestamp_seconds']}s)", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing slides: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        speech_sections = parse_speech(speech_content)
        if args.verbose:
            print(f"✓ Parsed {len(speech_sections)} speech sections", file=sys.stderr)
            for section in speech_sections:
                print(f"  - Section [{section['start_time']} - {section['end_time']}] ({section['end_time_seconds']}s)", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing speech: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Combine
    try:
        combined_content = combine_slides_and_speech(slides_data, speech_sections)
        if args.verbose:
            print(f"✓ Combined slides and speech", file=sys.stderr)
    except Exception as e:
        print(f"Error combining content: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(combined_content)
            if args.verbose:
                print(f"✓ Wrote combined content to: {args.output}", file=sys.stderr)
            else:
                print(f"Combined file saved to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(combined_content)


if __name__ == "__main__":
    main()
