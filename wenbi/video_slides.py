"""
Video Slide Detection Module
Handles ROI detection, scene detection, and OCR integration for video presentations.
"""

import cv2
import numpy as np
import os
import tempfile
import shutil
import logging
import time
from typing import List, Dict, Tuple, Optional
from scenedetect import VideoManager, SceneManager, ContentDetector
# Import ffmpeg-python for frame extraction
# Use conditional import to handle missing dependency
try:
    import ffmpeg
    FFmpegAvailable = True
except ImportError:
    FFmpegAvailable = False
    ffmpeg = None
from wenbi.model import convert_single_slide_image
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


def detect_video_resolution(video_path: str) -> Tuple[int, int]:
    """Detect video resolution using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return width, height


def detect_slide_roi(video_path: str, logger=None, verbose=False) -> Tuple[int, int, int, int]:
    """
    Automatically detect slide area in Zoom recording
    Returns: (x0, y0, x1, y1) coordinates
    """
    if verbose and logger:
        logger.debug("Starting automatic ROI detection for slide area")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Analyze first 30 seconds to detect slide area
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_analyze = min(int(fps * 30), 900)  # 30 seconds max
    
    slide_areas = []
    frame_count = 0
    
    while frame_count < frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 30th frame (1 second intervals)
        if frame_count % 30 != 0:
            continue
        
        if verbose and logger:
            logger.debug(f"Analyzing frame {frame_count}/{frames_to_analyze}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for rectangular shapes (slides)
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Typical slide aspect ratios (4:3, 16:9, 16:10)
                if 0.75 <= aspect_ratio <= 2.0:
                    rectangles.append((x, y, w, h, area))
        
        # Sort by area, take the largest rectangle that covers 70-80% of frame
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        
        for rect in sorted(rectangles, key=lambda x: x[4], reverse=True):
            x, y, w, h, area = rect
            coverage = area / frame_area
            
            if 0.7 <= coverage <= 0.85:  # 70-85% coverage
                slide_areas.append((x, y, x + w, y + h))
                break
        
        # If no suitable rectangle found, use default assumption
        if len(slide_areas) == frame_count // 30:
            # Default: assume slides take 75% of frame, excluding top-right corner (lecturer)
            default_w = int(frame_width * 0.75)
            default_h = int(frame_height * 0.85)
            slide_areas.append((0, 0, default_w, default_h))
    
    cap.release()
    
    if not slide_areas:
        # Fallback to default coordinates
        width, height = detect_video_resolution(video_path)
        return (0, 0, int(width * 0.75), int(height * 0.85))
    
    # Return median ROI coordinates to handle variations
    slide_areas = np.array(slide_areas)
    median_roi = np.median(slide_areas, axis=0).astype(int)
    
    if verbose and logger:
        logger.debug(f"Detected ROI: {tuple(median_roi)}")
    
    return tuple(median_roi)


def detect_slide_changes(video_path: str, roi_coords: Tuple[int, int, int, int], 
                    logger=None, verbose=False, start_time: str = "00:00:00", 
                    end_time: str = "01:00:00") -> List[Dict]:
    """
    Detect slide transitions using PySceneDetect with ContentDetector
    Returns list of scene changes with timestamps
    """
    if verbose and logger:
        logger.debug(f"Detecting slide changes with ROI: {roi_coords}")
        logger.debug(f"Time range: {start_time} to {end_time}")
    
    # Create video manager (process full video, then filter scenes by time range)
    video_manager = VideoManager([video_path])
    video_manager.set_downscale_factor(1)  # Keep full resolution for accuracy
    video_manager.start()
    
    # Create scene manager with ContentDetector
    scene_manager = SceneManager()
    
    # ContentDetector is better for slides than AdaptiveDetector
    # Threshold: 27 is good for slide transitions (default is 30)
    # min_scene_len: 2 seconds to avoid detecting animations
    detector = ContentDetector(
        threshold=27,
        min_scene_len=2  # 2 seconds minimum (int type)
    )
    scene_manager.add_detector(detector)
    
    try:
        # Process video
        scene_manager.detect_scenes(video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Convert to list of dictionaries with timestamps
        slides_info = []
        target_start = parse_time_to_seconds(start_time)
        target_end = parse_time_to_seconds(end_time)
        
        for i, scene in enumerate(scene_list):
            scene_start = scene[0].get_timecode()
            scene_end = scene[1].get_timecode()
            scene_start_abs = scene[0].get_seconds()
            scene_end_abs = scene[1].get_seconds()
            
            # Only include slides within our time range
            if (scene_start_abs + target_start) <= target_end:
                slides_info.append({
                    'slide_number': i + 1,
                    'start_time': scene_start,
                    'end_time': scene_end,
                    'start_seconds': scene_start_abs + target_start,
                    'end_seconds': scene_end_abs + target_start,
                    'duration': scene_end_abs - scene_start_abs
                })
        
        if verbose and logger:
            logger.debug(f"Detected {len(slides_info)} slide transitions")
        
        return slides_info
        
    except Exception as e:
        if logger:
            logger.error(f"Error detecting scenes: {e}")
        raise
    finally:
        video_manager.release()


def extract_frame_at_timestamp(video_path: str, timestamp: str, 
                          roi_coords: Optional[Tuple[int, int, int, int]] = None,
                          output_dir: str = "", logger=None, verbose=False) -> str:
    """
    Extract a single frame at given timestamp
    Returns path to saved image
    """
    if output_dir == "":
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean timestamp format
    clean_timestamp = re.sub(r'[:.]', '_', timestamp)
    output_filename = f"slide_{clean_timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        if verbose and logger:
            logger.debug(f"Extracting frame at {timestamp}")
        
        # Use ffmpeg for precise frame extraction
        if FFmpegAvailable:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .filter('crop', 
                        roi_coords[2] - roi_coords[0] if roi_coords else 'iw',
                        roi_coords[3] - roi_coords[1] if roi_coords else 'ih',
                        roi_coords[0] if roi_coords else 0,
                        roi_coords[1] if roi_coords else 0)
                .output(output_path, vframes=1, format='image2', vcodec='png')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # Fallback to OpenCV if ffmpeg not available
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(float(timestamp.split(':')[0]) * 3600 + 
                           float(timestamp.split(':')[1]) * 60 + 
                           float(timestamp.split(':')[2]) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and roi_coords:
                x0, y0, x1, y1 = roi_coords
                frame = frame[y0:y1, x0:x1]
            if ret:
                cv2.imwrite(output_path, frame)
            cap.release()
        
        if verbose and logger:
            logger.debug(f"Frame saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting frame: {e}")
        raise


def ocr_slide_image(image_path: str, output_dir: str = "", 
                  logger=None, verbose=False) -> Dict:
    """
    OCR a slide image using marker-pdf functionality
    Returns dictionary with OCR text and metadata
    """
    if output_dir == "":
        output_dir = tempfile.mkdtemp()
    
    try:
        if verbose and logger:
            logger.debug(f"OCR processing: {image_path}")
        
        # Use marker's convert_single function for OCR
        result = convert_single_slide_image(
            image_path,
            langs=['Chinese', 'English'],
            output_dir=output_dir
        )
        
        ocr_data = {
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'page_id': result.get('page_id', 0),
            'metadata': result.get('metadata', {})
        }
        
        if verbose and logger:
            logger.debug(f"OCR completed, confidence: {ocr_data['confidence']}")
        
        return ocr_data
        
    except Exception as e:
        if logger:
            logger.warning(f"OCR failed for {image_path}: {e}")
        
        # Return data indicating failure
        return {
            'text': '',
            'confidence': 0,
            'page_id': 0,
            'metadata': {'error': str(e)},
            'failed': True
        }


def process_video_slides(video_path: str, output_dir: str, cite_timestamps: bool = True,
                      roi_coords: Optional[Tuple[int, int, int, int]] = None,
                      start_time: str = "00:00:00", end_time: str = "01:00:00",
                      logger=None, verbose=False) -> str:
    """
    Main function to process video slides with ROI detection and OCR
    Returns path to combined markdown file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix="wenbi_slides_")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # Step 1: Detect ROI if not provided
        if roi_coords is None:
            if verbose and logger:
                logger.debug("Detecting slide area automatically")
            
            roi_coords = detect_slide_roi(video_path, logger, verbose)
        
        # Step 2: Detect slide changes with time range
        if verbose and logger:
            logger.debug(f"Detecting slide transitions from {start_time} to {end_time}")
        
        slides_info = detect_slide_changes(video_path, roi_coords, logger, verbose, start_time, end_time)
        
        if not slides_info:
            raise ValueError("No slide transitions detected")
        
        # Step 3: Extract frames at each slide transition
        extracted_slides = []
        for slide in slides_info:
            frame_path = extract_frame_at_timestamp(
                video_path, 
                slide['start_time'], 
                roi_coords, 
                frames_dir, 
                logger, 
                verbose
            )
            
            # OCR extracted frame
            ocr_data = ocr_slide_image(
                frame_path, 
                frames_dir, 
                logger, 
                verbose
            )
            
            extracted_slides.append({
                'slide_number': slide['slide_number'],
                'timestamp': slide['start_time'],
                'frame_path': frame_path,
                'ocr_data': ocr_data,
                'duration': slide['duration']
            })
        
        # Step 4: Generate combined markdown
        if verbose and logger:
            logger.debug("Generating combined markdown output")
        
        markdown_content = generate_video_slides_markdown(
            extracted_slides, cite_timestamps, logger, verbose
        )
        
        # Save output file
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_slides_ppt.md")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        if verbose and logger:
            logger.debug(f"Output saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing video slides: {e}")
        raise
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def generate_video_slides_markdown(slides: List[Dict], cite_timestamps: bool = True,
                               logger=None, verbose=False) -> str:
    """
    Generate markdown content from extracted slides
    """
    markdown_lines = [
        "# Video Slides with OCR\n"
    ]
    
    for slide in slides:
        # Add timestamp header if requested
        if cite_timestamps:
            markdown_lines.append(f"\n### **{slide['timestamp']}**\n")
        
        # Add slide image if OCR failed or confidence is low
        if slide['ocr_data'].get('failed', False) or slide['ocr_data']['confidence'] < 0.5:
            markdown_lines.append(f"\n![Slide {slide['slide_number']}]({slide['frame_path']})\n")
        else:
            # Add OCR text
            ocr_text = slide['ocr_data']['text'].strip()
            if ocr_text:
                markdown_lines.append(f"\n{ocr_text}\n")
            else:
                # Fallback to image if no text
                markdown_lines.append(f"\n![Slide {slide['slide_number']}]({slide['frame_path']})\n")
    
    return "\n".join(markdown_lines)


def validate_video_input(video_path: str, logger=None, verbose=False) -> bool:
    """
    Validate that input is a supported video file
    """
    if not os.path.exists(video_path):
        return False
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm')
    if video_path.lower().endswith(video_extensions):
        return True
    
    # Try to open with OpenCV to confirm it's a video
    try:
        cap = cv2.VideoCapture(video_path)
        is_video = cap.isOpened()
        cap.release()
        return is_video
    except:
        return False


def manual_roi_override(video_path: str, logger=None, verbose=False) -> Tuple[int, int, int, int]:
    """
    Allow user to manually specify ROI coordinates
    For now, return default coordinates, but this can be enhanced with GUI
    """
    width, height = detect_video_resolution(video_path)
    
    if verbose and logger:
        logger.debug("Using default ROI coordinates (75% width, 85% height)")
    
    # Default: 75% width, 85% height, from top-left
    return (0, 0, int(width * 0.75), int(height * 0.85))