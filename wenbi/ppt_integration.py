"""
Enhanced PPT Integration Module
Handles video input processing with speech+slides extraction and timestamp integration
"""

import os
import tempfile
import shutil
import logging
from typing import List, Dict, Tuple, Optional
from wenbi.video_slides import process_video_slides, validate_video_input
from wenbi.main import process_input
from wenbi.utils import parse_subtitle
import re


def parse_time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to seconds"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return int(hours * 3600 + minutes * 60 + seconds)
        return 0
    except:
        return 0


def is_video_input(input_path: str) -> bool:
    """Check if input is a video file or URL"""
    if not input_path:
        return False
    
    # Check for URL
    if input_path.startswith(("http://", "https://", "www.")):
        return True
    
    # Check for video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm')
    if input_path.lower().endswith(video_extensions):
        return True
    
    # Try to validate with OpenCV
    return validate_video_input(input_path)


def process_speech_from_video(video_path: str, output_dir: str, 
                            cite_timestamps: bool = True, logger=None, verbose=False) -> Tuple[str, str]:
    """
    Process video to extract speech content using rewrite subcommand
    Returns: (speech_content, speech_file_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug("Processing speech from video using rewrite subcommand")
    
    try:
        # Use process_input with rewrite subcommand
        result = process_input(
            file_path=video_path,
            url="",
            subcommand="rewrite",
            output_dir=output_dir,
            cite_timestamps=cite_timestamps,
            verbose=verbose
        )
        
        speech_content = result[0]
        speech_file = result[1]
        
        if verbose and logger:
            logger.debug(f"Speech processing completed: {speech_file}")
        
        return speech_content, speech_file
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing speech from video: {e}")
        raise


def process_slides_from_video(video_path: str, output_dir: str,
                           cite_timestamps: bool = True, logger=None, verbose=False,
                           start_time: str = "00:00:00", end_time: str = "01:00:00") -> Tuple[str, str]:
    """
    Process video to extract slides using video_slides module
    Returns: (slides_content, slides_file_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug("Processing slides from video using video_slides module")
    
    try:
        # Use video_slides module for slide extraction
        slides_file = process_video_slides(
            video_path=video_path,
            output_dir=output_dir,
            cite_timestamps=cite_timestamps,
            start_time=start_time,
            end_time=end_time,
            logger=logger,
            verbose=verbose
        )
        
        # Read the generated slides content
        with open(slides_file, 'r', encoding='utf-8') as f:
            slides_content = f.read()
        
        if verbose and logger:
            logger.debug(f"Slides processing completed: {slides_file}")
        
        return slides_content, slides_file
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing slides from video: {e}")
        raise


def parse_vtt_timestamps(vtt_content: str) -> List[Dict]:
    """
    Parse VTT content to extract timestamped speech segments
    Returns: List of {'start_time': str, 'end_time': str, 'text': str, 'start_seconds': int}
    """
    segments = []
    lines = vtt_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp lines (HH:MM:SS.mmm --> HH:MM:SS.mmm)
        if '-->' in line:
            try:
                timestamp_line = line
                text_lines = []
                i += 1
                
                # Collect text lines until next timestamp or empty line
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                
                # Parse timestamps
                start_str, end_str = timestamp_line.split(' --> ')
                start_seconds = parse_time_to_seconds(start_str.split('.')[0])
                end_seconds = parse_time_to_seconds(end_str.split('.')[0])
                
                if text_lines:
                    segments.append({
                        'start_time': start_str.split('.')[0],
                        'end_time': end_str.split('.')[0],
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'text': ' '.join(text_lines)
                    })
            except Exception:
                pass  # Skip malformed timestamp lines
        else:
            i += 1
    
    return segments


def parse_slides_timestamps(slides_content: str) -> List[Dict]:
    """
    Parse slides markdown content to extract timestamped slide segments
    Returns: List of {'timestamp': str, 'text': str, 'seconds': int}
    """
    slides = []
    lines = slides_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp headers (### **HH:MM:SS**)
        if line.startswith('### **') and line.endswith('**'):
            try:
                timestamp = line.replace('### **', '').replace('**', '')
                slide_seconds = parse_time_to_seconds(timestamp)
                
                # Collect slide content until next header
                slide_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('### **') and next_line.endswith('**'):
                        break
                    if next_line:  # Skip empty lines
                        slide_lines.append(next_line)
                    i += 1
                
                if slide_lines:
                    slides.append({
                        'timestamp': timestamp,
                        'seconds': slide_seconds,
                        'text': '\n'.join(slide_lines)
                    })
            except Exception:
                pass  # Skip malformed timestamp lines
        else:
            i += 1
    
    return slides


def integrate_speech_and_slides(speech_content: str, slides_content: str, 
                              cite_timestamps: bool = True, logger=None, verbose=False) -> str:
    """
    Integrate speech and slides content with timestamp alignment
    Returns: Combined markdown content
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug("Integrating speech and slides with timestamp alignment")
    
    if not cite_timestamps:
        # Simple concatenation without timestamp processing
        return f"# Speech Content\n\n{speech_content}\n\n# Slides Content\n\n{slides_content}"
    
    try:
        # Parse both contents for timestamps
        speech_segments = parse_vtt_timestamps(speech_content)
        slides_segments = parse_slides_timestamps(slides_content)
        
        if verbose and logger:
            logger.debug(f"Parsed {len(speech_segments)} speech segments and {len(slides_segments)} slide segments")
        
        # Build integrated content
        integrated_lines = ["# Integrated Lecture Content\n"]
        current_slide_idx = 0
        
        for speech_seg in speech_segments:
            # Add speech segment with timestamp header
            integrated_lines.append(f"\n### **{speech_seg['start_time']}**\n")
            integrated_lines.append(speech_seg['text'])
            
            # Find slides that fall within this speech time window
            relevant_slides = []
            while (current_slide_idx < len(slides_segments) and 
                   slides_segments[current_slide_idx]['seconds'] >= speech_seg['start_seconds'] and
                   slides_segments[current_slide_idx]['seconds'] <= speech_seg['end_seconds']):
                relevant_slides.append(slides_segments[current_slide_idx])
                current_slide_idx += 1
            
            # Insert relevant slides
            for slide in relevant_slides:
                integrated_lines.append(f"\n### **{slide['timestamp']}**\n")
                integrated_lines.append(slide['text'])
        
        # Add any remaining slides that didn't fit in speech segments
        while current_slide_idx < len(slides_segments):
            slide = slides_segments[current_slide_idx]
            integrated_lines.append(f"\n### **{slide['timestamp']}**\n")
            integrated_lines.append(slide['text'])
            current_slide_idx += 1
        
        if verbose and logger:
            logger.debug("Integration completed successfully")
        
        return '\n'.join(integrated_lines)
        
    except Exception as e:
        if logger:
            logger.error(f"Error during integration: {e}")
        
        # Fallback to simple concatenation
        if verbose and logger:
            logger.debug("Falling back to simple concatenation")
        
        return f"# Speech Content\n\n{speech_content}\n\n# Slides Content\n\n{slides_content}"


def process_video_workflow(input_path: str, output_dir: str, cite_timestamps: bool = True,
                           start_time: str = "00:00:00", end_time: str = "01:00:00",
                           logger=None, verbose=False) -> str:
    """
    Main workflow for processing video input with speech+slides extraction
    Returns: Path to combined.md file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp(prefix="wenbi_ppt_")
    
    try:
        if verbose and logger:
            logger.debug(f"Starting video workflow for: {input_path}")
        
        # Step 1: Process speech from video
        if verbose and logger:
            logger.debug("Step 1: Processing speech from video")
        
        speech_content, speech_file = process_speech_from_video(
            input_path, temp_dir, cite_timestamps, logger, verbose
        )
        
        # Step 2: Process slides from video
        if verbose and logger:
            logger.debug("Step 2: Processing slides from video")
        
        slides_content, slides_file = process_slides_from_video(
            input_path, temp_dir, cite_timestamps, logger, verbose, start_time, end_time
        )
        
        # Step 3: Integrate content with timestamps
        if verbose and logger:
            logger.debug("Step 3: Integrating speech and slides with timestamps")
        
        combined_content = integrate_speech_and_slides(
            speech_content, slides_content, cite_timestamps, logger, verbose
        )
        
        # Step 4: Save all output files
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual files
        speech_output = os.path.join(output_dir, f"{base_name}_speech.md")
        slides_output = os.path.join(output_dir, f"{base_name}_slides.md")
        combined_output = os.path.join(output_dir, f"{base_name}_combined.md")
        
        with open(speech_output, 'w', encoding='utf-8') as f:
            f.write(speech_content)
        
        with open(slides_output, 'w', encoding='utf-8') as f:
            f.write(slides_content)
        
        with open(combined_output, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        if verbose and logger:
            logger.debug(f"Output files saved:")
            logger.debug(f"  Speech: {speech_output}")
            logger.debug(f"  Slides: {slides_output}")
            logger.debug(f"  Combined: {combined_output}")
        
        return combined_output
        
    except Exception as e:
        if logger:
            logger.error(f"Error in video workflow: {e}")
        raise
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass