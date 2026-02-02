"""
Video Slide Detection Module
Handles ROI detection, scene detection, and OCR integration for video presentations.
"""

import logging
import os
import shutil
import tempfile
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, VideoManager

# Import scikit-image for SSIM calculation
try:
    from skimage.metrics import structural_similarity as ssim
    SSIMAvailable = True
except ImportError:
    SSIMAvailable = False
    ssim = None

# Import ffmpeg-python for frame extraction
# Use conditional import to handle missing dependency
try:
    import ffmpeg

    FFmpegAvailable = True
except ImportError:
    FFmpegAvailable = False
    ffmpeg = None
import re

from wenbi.model import convert_single_slide_image


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using difflib.
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    if not text1 and not text2:
        return 1.0  # Both empty, consider as same
    if not text1 or not text2:
        return 0.0  # One empty, one not, different
    
    # Normalize whitespace for comparison
    normalized1 = " ".join(text1.split())
    normalized2 = " ".join(text2.split())
    
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def deduplicate_slides(
    slides: List[Dict], 
    similarity_threshold: float = 0.85,
    logger=None,
    verbose=False
) -> List[Dict]:
    """
    Remove duplicate slides based on OCR content similarity.
    
    Args:
        slides: List of slide dictionaries with 'content' field
        similarity_threshold: Similarity ratio above which slides are considered duplicates (0.0-1.0)
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        List of unique slides (duplicates removed)
    """
    if not slides:
        return []
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    unique_slides = []
    duplicates_removed = 0
    
    for i, slide in enumerate(slides):
        is_duplicate = False
        slide_content = slide.get('content', '').strip()
        
        # Compare with all previously accepted unique slides
        for existing in unique_slides:
            existing_content = existing.get('content', '').strip()
            
            similarity = calculate_text_similarity(slide_content, existing_content)
            
            if similarity >= similarity_threshold:
                is_duplicate = True
                duplicates_removed += 1
                
                if verbose and logger:
                    logger.debug(
                        f"Slide at {slide.get('timestamp', 'unknown')} is duplicate "
                        f"of slide at {existing.get('timestamp', 'unknown')} "
                        f"(similarity: {similarity:.2%})"
                    )
                break
        
        if not is_duplicate:
            unique_slides.append(slide)
    
    if verbose and logger:
        logger.debug(
            f"Deduplication complete: {len(slides)} slides -> {len(unique_slides)} unique slides "
            f"({duplicates_removed} duplicates removed)"
        )
    
    return unique_slides


def calculate_histogram_similarity(img1, img2) -> float:
    """
    Calculate histogram similarity between two images using correlation.
    Returns a value between -1.0 (inverse correlation) and 1.0 (perfect correlation).
    """
    # Convert to grayscale for histogram calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Calculate correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation


def deduplicate_slides_by_image(
    frames_data: List[Dict], 
    ssim_threshold: float = 0.98,
    hist_threshold: float = 0.15,
    logger=None,
    verbose=False
) -> List[Dict]:
    """
    Remove duplicate slides using OpenCV histogram comparison + SSIM.
    
    Args:
        frames_data: List of frame dictionaries with 'frame_path' and 'timestamp' fields
        ssim_threshold: SSIM similarity threshold (default: 0.98)
        hist_threshold: Histogram correlation threshold (default: 0.15)
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        List of unique frames (duplicates removed)
    """
    if not frames_data:
        return []
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug(
            f"Starting image-based deduplication: ssim_threshold={ssim_threshold}, "
            f"hist_threshold={hist_threshold}"
        )
    
    unique_frames = []
    duplicates_removed = 0
    
    # Pre-load images to avoid repeated I/O
    loaded_images = []
    for frame_data in frames_data:
        frame_path = frame_data.get('frame_path', '')
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                loaded_images.append(img)
            else:
                loaded_images.append(None)
        else:
            loaded_images.append(None)
    
    for i, frame_data in enumerate(frames_data):
        current_img = loaded_images[i]
        
        if current_img is None:
            # Can't compare if image failed to load, keep it
            unique_frames.append(frame_data)
            continue
        
        is_duplicate = False
        
        # Compare with previously accepted unique frames
        for j, existing_frame in enumerate(unique_frames):
            existing_img = loaded_images[frames_data.index(existing_frame)]
            
            if existing_img is None:
                continue
            
            # First, quick histogram comparison to filter obvious non-duplicates
            hist_similarity = calculate_histogram_similarity(current_img, existing_img)
            
            if hist_similarity < hist_threshold:
                # Histogram correlation too low, definitely not a duplicate
                continue
            
            # If histogram passes, apply SSIM for precise comparison
            if SSIMAvailable:
                try:
                    # Convert to grayscale for SSIM
                    gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                    gray_existing = cv2.cvtColor(existing_img, cv2.COLOR_BGR2GRAY)
                    
                    # Check if grayscale conversions succeeded
                    if gray_current is None or gray_existing is None:
                        continue
                    
                    # Ensure images have same dimensions
                    if gray_current.shape != gray_existing.shape:
                        # Resize larger image to match smaller
                        h_min = min(gray_current.shape[0], gray_existing.shape[0])
                        w_min = min(gray_current.shape[1], gray_existing.shape[1])
                        gray_current = gray_current[:h_min, :w_min]
                        gray_existing = gray_existing[:h_min, :w_min]
                    
                    # Calculate SSIM
                    ssim_score = ssim(gray_current, gray_existing)
                    
                    if ssim_score >= ssim_threshold:
                        is_duplicate = True
                        duplicates_removed += 1
                        
                        if verbose and logger:
                            logger.debug(
                                f"Frame at {frame_data.get('timestamp', 'unknown')} is duplicate "
                                f"of frame at {existing_frame.get('timestamp', 'unknown')} "
                                f"(histogram: {hist_similarity:.3f}, ssim: {ssim_score:.3f})"
                            )
                        break
                        
                except Exception as e:
                    if verbose and logger:
                        logger.warning(
                            f"SSIM calculation failed for frames at "
                            f"{frame_data.get('timestamp', 'unknown')} and "
                            f"{existing_frame.get('timestamp', 'unknown')}: {e}"
                        )
                    # Fall back to histogram comparison only
                    if hist_similarity >= (1.0 - hist_threshold):  # Higher threshold for fallback
                        is_duplicate = True
                        duplicates_removed += 1
                        if verbose and logger:
                            logger.debug(
                                f"Frame at {frame_data.get('timestamp', 'unknown')} is duplicate "
                                f"(histogram-only: {hist_similarity:.3f})"
                            )
                        break
            else:
                # SSIM not available, use histogram only
                if hist_similarity >= (1.0 - hist_threshold):  # Higher threshold for histogram-only
                    is_duplicate = True
                    duplicates_removed += 1
                    if verbose and logger:
                        logger.debug(
                            f"Frame at {frame_data.get('timestamp', 'unknown')} is duplicate "
                            f"(histogram-only: {hist_similarity:.3f})"
                        )
                    break
        
        if not is_duplicate:
            unique_frames.append(frame_data)
    
    if verbose and logger:
        logger.debug(
            f"Image deduplication complete: {len(frames_data)} frames -> {len(unique_frames)} unique frames "
            f"({duplicates_removed} duplicates removed)"
        )
    
    return unique_frames


def parse_time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to seconds"""
    try:
        parts = time_str.split(":")
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


def detect_slide_roi(
    video_path: str, logger=None, verbose=False
) -> Tuple[int, int, int, int]:
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
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            # Default: use specific coordinates (0.1, 0.1, 0.9, 0.9)
            default_x0 = int(frame_width * 0.1)
            default_y0 = int(frame_height * 0.1)
            default_x1 = int(frame_width * 0.9)
            default_y1 = int(frame_height * 0.9)
            slide_areas.append((default_x0, default_y0, default_x1, default_y1))

    cap.release()

    # Return median ROI coordinates to handle variations
    if slide_areas:
        slide_areas = np.array(slide_areas)
        median_roi = np.median(slide_areas, axis=0).astype(int)
        
        if verbose and logger:
            logger.debug(f"Detected ROI: {tuple(median_roi)}")
        
        return tuple(median_roi)


def detect_slide_rectangles_in_frames(
    frames_data: List[Dict],
    each_roi: bool = False,
    logger=None,
    verbose=False,
) -> List[Dict]:
    """
    Detect slide rectangles in each frame using OpenCV.
    If each_roi is True, detect rectangles for each frame individually.
    Otherwise, use first detected ROI for all frames.
    
    Args:
        frames_data: List of frame dictionaries with frame_path
        each_roi: Enable per-frame ROI detection
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        Updated frames_data with roi_coords for each frame
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug(
            f"Detecting slide rectangles in {len(frames_data)} frames "
            f"(each_roi={each_roi})"
        )
    
    updated_frames = []
    first_roi = None
    
    for i, frame_data in enumerate(frames_data):
        frame_path = frame_data['frame_path']
        
        if not os.path.exists(frame_path):
            if verbose and logger:
                logger.warning(f"Frame file not found: {frame_path}")
            continue
            
        # Read frame
        img = cv2.imread(frame_path)
        if img is None:
            if verbose and logger:
                logger.warning(f"Could not read frame: {frame_path}")
            continue
            
        if each_roi or (i == 0):  # Each ROI mode OR first frame
            if verbose and logger:
                logger.debug(f"Detecting ROI for frame {frame_data.get('frame_index', i)}")
                
            roi = detect_slide_rectangle_in_frame(img, logger, verbose)
            if roi:
                first_roi = roi if i == 0 else first_roi
                frame_data['roi_coords'] = roi
        else:
            # Use first detected ROI for all subsequent frames
            if first_roi:
                frame_data['roi_coords'] = first_roi
            else:
                # No ROI detected, use full frame
                height, width = img.shape[:2]
                frame_data['roi_coords'] = (0, 0, width, height)
        
        updated_frames.append(frame_data)
    
    if verbose and logger:
        logger.debug(f"ROI detection completed for {len(updated_frames)} frames")
        
    return updated_frames


def detect_slide_changes(
    video_path: str,
    roi_coords: Tuple[int, int, int, int],
    logger=None,
    verbose=False,
    start_time: str = "00:00:00",
    end_time: Optional[str] = None,
    scenedetect_threshold: float = 35.0,
    min_scene_seconds: float = 120.0,
    downscale: float = 2.0,
) -> List[Dict]:
    """
    Detect slide transitions using PySceneDetect's ContentDetector.
    Returns list of scene changes with timestamps.

    Parameters:
      - scenedetect_threshold: numeric threshold forwarded to ContentDetector(threshold=...)
      - min_scene_seconds: minimum scene duration in seconds (converted to frames)
      - downscale: VideoManager downscale factor to speed processing
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if verbose and logger:
        logger.debug(f"Detecting slide changes with ROI: {roi_coords}")
        logger.debug(
            f"Time range: {start_time} to {end_time or 'end of video'}, threshold={scenedetect_threshold}, min_scene_seconds={min_scene_seconds}, downscale={downscale}"
        )

    print(f"Debug detect_slide_changes: Starting PySceneDetect-based detection")
    print(f"Debug detect_slide_changes: video_path={video_path}")
    print(f"Debug detect_slide_changes: roi_coords={roi_coords}")
    print(f"Debug detect_slide_changes: time range {start_time} to {end_time}")
    print(
        f"Debug detect_slide_changes: Detection params threshold={scenedetect_threshold}, min_scene_seconds={min_scene_seconds}, downscale={downscale}"
    )

    # Try to obtain FPS via OpenCV to compute frame-based min_scene length
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    start_seconds = parse_time_to_seconds(start_time)
    end_seconds = parse_time_to_seconds(end_time) if end_time else float("inf")

    # Compute min_scene_len in frames for ContentDetector
    min_scene_frames = max(1, int(min_scene_seconds * fps))

    video_manager = VideoManager([video_path])
    # set_downscale_factor expected to speed detection (1 no downscale, 2 half, etc.)
    try:
        # Ensure downscale is a positive number
        if downscale and downscale > 0:
            video_manager.set_downscale_factor(downscale)
    except Exception:
        # Some versions accept different APIs; ignore if not supported
        if verbose and logger:
            logger.debug(
                "Warning: set_downscale_factor may not be supported by this VideoManager"
            )

    video_manager.start()

    scene_manager = SceneManager()
    detector = ContentDetector(
        threshold=scenedetect_threshold, min_scene_len=min_scene_frames
    )
    scene_manager.add_detector(detector)

    try:
        print(
            "Debug detect_slide_changes: Calling scene_manager.detect_scenes (this may take some time)..."
        )
        scene_manager.detect_scenes(video_manager)
        print("Debug detect_slide_changes: Scene detection completed")
        scene_list = scene_manager.get_scene_list()
        print(f"Debug detect_slide_changes: Got {len(scene_list)} scenes")

        slides_info = []

        for i, scene in enumerate(scene_list):
            # scene is a tuple of (start, end) as SceneTimecode objects
            scene_start_abs = scene[0].get_seconds()

            if start_seconds <= scene_start_abs < end_seconds:
                slides_info.append(
                    {
                        "slide_number": i + 1,
                        "start_time": scene[0].get_timecode(),
                        "end_time": scene[1].get_timecode(),
                        "start_seconds": scene_start_abs,
                        "end_seconds": scene[1].get_seconds(),
                        "duration": scene[1].get_seconds() - scene_start_abs,
                    }
                )

        if verbose and logger:
            logger.debug(
                f"Detected {len(slides_info)} slide transitions within the specified time range."
            )

        return slides_info

    except Exception as e:
        if logger:
            logger.error(f"Error detecting scenes: {e}")
        raise
    finally:
        try:
            video_manager.release()
        except Exception:
            pass


def format_seconds_to_timecode(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def extract_frame_at_timestamp(
    video_path: str,
    timestamp: str,
    roi_coords: Optional[Tuple[int, int, int, int]] = None,
    output_dir: str = "",
    logger=None,
    verbose=False,
) -> str:
    """
    Extract a single frame at given timestamp
    Returns path to saved image
    """
    if output_dir == "":
        output_dir = tempfile.mkdtemp()

    os.makedirs(output_dir, exist_ok=True)

    # Clean timestamp format
    clean_timestamp = re.sub(r"[:.]", "_", timestamp)
    output_filename = f"slide_{clean_timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    try:
        if verbose and logger:
            logger.debug(f"Extracting frame at {timestamp}")

        # Use ffmpeg for precise frame extraction
        if FFmpegAvailable:
            if roi_coords and len(roi_coords) == 4:
                crop_width = roi_coords[2] - roi_coords[0]
                crop_height = roi_coords[3] - roi_coords[1]
                crop_x = roi_coords[0]
                crop_y = roi_coords[1]
            else:
                crop_width = "iw"
                crop_height = "ih" 
                crop_x = 0
                crop_y = 0
                
            (
                ffmpeg.input(video_path, ss=timestamp)
                .filter(
                    "crop",
                    crop_width,
                    crop_height,
                    crop_x,
                    crop_y,
                )
                .output(output_path, vframes=1, format="image2", vcodec="png")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # Fallback to OpenCV if ffmpeg not available
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(
                float(timestamp.split(":")[0]) * 3600
                + float(timestamp.split(":")[1]) * 60
                + float(timestamp.split(":")[2]) * fps
            )
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


def ocr_slide_image(
    image_path: str, output_dir: str = "", logger=None, verbose=False
) -> Dict:
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
            image_path, langs=["Chinese", "English"], output_dir=output_dir
        )

        ocr_data = {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
            "page_id": result.get("page_id", 0),
            "metadata": result.get("metadata", {}),
        }

        if verbose and logger:
            logger.debug(f"OCR completed, confidence: {ocr_data['confidence']}")

        return ocr_data

    except Exception as e:
        if logger:
            logger.warning(f"OCR failed for {image_path}: {e}")

        # Return data indicating failure
        return {
            "text": "",
            "confidence": 0,
            "page_id": 0,
            "metadata": {"error": str(e)},
            "failed": True,
        }


def process_video_slides(
    video_path: str,
    output_dir: str,
    cite_timestamps: bool = True,
    roi_coords: Optional[Tuple[int, int, int, int]] = None,
    start_time: str = "00:00:00",
    end_time: str = "01:00:00",
    logger=None,
    verbose=False,
) -> str:
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

        slides_info = detect_slide_changes(
            video_path, roi_coords, logger, verbose, start_time, end_time
        )

        if not slides_info:
            raise ValueError("No slide transitions detected")

        # Step 3: Extract frames at each slide transition
        extracted_slides = []
        for slide in slides_info:
            frame_path = extract_frame_at_timestamp(
                video_path, slide["start_time"], roi_coords, frames_dir, logger, verbose
            )

            # OCR extracted frame
            ocr_data = ocr_slide_image(frame_path, frames_dir, logger, verbose)

            extracted_slides.append(
                {
                    "slide_number": slide["slide_number"],
                    "timestamp": slide["start_time"],
                    "frame_path": frame_path,
                    "ocr_data": ocr_data,
                    "duration": slide["duration"],
                }
            )

        # Step 4: Generate combined markdown
        if verbose and logger:
            logger.debug("Generating combined markdown output")

        markdown_content = generate_video_slides_markdown(
            extracted_slides, cite_timestamps, logger, verbose
        )

        # Save output file
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_slides_ppt.md")

        with open(output_file, "w", encoding="utf-8") as f:
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


def generate_video_slides_markdown(
    slides: List[Dict], cite_timestamps: bool = True, logger=None, verbose=False
) -> str:
    """
    Generate markdown content from extracted slides
    """
    markdown_lines = ["# Video Slides with OCR\n"]

    for slide in slides:
        # Add timestamp header if requested
        if cite_timestamps:
            markdown_lines.append(f"\n### **{slide['timestamp']}**\n")

        # Add slide image if OCR failed or confidence is low
        if (
            slide["ocr_data"].get("failed", False)
            or slide["ocr_data"]["confidence"] < 0.5
        ):
            markdown_lines.append(
                f"\n![Slide {slide['slide_number']}]({slide['frame_path']})\n"
            )
        else:
            # Add OCR text
            ocr_text = slide["ocr_data"]["text"].strip()
            if ocr_text:
                markdown_lines.append(f"\n{ocr_text}\n")
            else:
                # Fallback to image if no text
                markdown_lines.append(
                    f"\n![Slide {slide['slide_number']}]({slide['frame_path']})\n"
                )

    return "\n".join(markdown_lines)


def validate_video_input(video_path: str, logger=None, verbose=False) -> bool:
    """
    Validate that input is a supported video file
    """
    if not os.path.exists(video_path):
        return False

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm")
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


def manual_roi_override(
    video_path: str, logger=None, verbose=False
) -> Tuple[int, int, int, int]:
    """
    Allow user to manually specify ROI coordinates
    For now, return default coordinates, but this can be enhanced with GUI
    """
    width, height = detect_video_resolution(video_path)

    if verbose and logger:
        logger.debug("Using manual ROI override based on typical presentation layout")

    # New coordinates based on user feedback: 85% width, 90% height
    roi_width = int(width * 0.85)
    roi_height = int(height * 0.90)

    if verbose and logger:
        logger.debug(f"Applying manual ROI: (0, 0, {roi_width}, {roi_height})")

    return (0, 0, roi_width, roi_height)


def extract_all_frames_from_video(
    video_path: str,
    output_dir: str,
    start_time: str = "00:00:10",
    end_time: Optional[str] = None,
    frame_interval: int = 60,
    logger=None,
    verbose=False,
) -> List[Dict]:
    """
    Extract frames from video at regular time intervals using ffmpeg.
    Returns list of frame information with paths and timestamps.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        start_time: Start time for extraction (HH:MM:SS)
        end_time: End time for extraction (HH:MM:SS)
        frame_interval: Extract frame every N seconds
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        List of frame dictionaries with timestamp and frame_path
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug(
            f"Extracting frames from {video_path} with {frame_interval}s interval, "
            f"time range: {start_time} to {end_time or 'end'}"
        )
    
    frames_data = []
    frames_dir = os.path.join(output_dir, "_extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # Get video metadata using ffprobe
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Parse time range
        start_seconds = parse_time_to_seconds(start_time)
        if end_time:
            end_seconds = parse_time_to_seconds(end_time)
        else:
            end_seconds = total_frames / fps
        
        if verbose and logger:
            logger.debug(
                f"Video: {total_frames} frames @ {fps:.2f} FPS, duration {total_frames/fps:.1f}s"
            )
        
        # Extract frames at regular intervals
        current_seconds = max(0, start_seconds)
        frame_count = 0
        
        while current_seconds < end_seconds:
            hours = int(current_seconds // 3600)
            minutes = int((current_seconds % 3600) // 60)
            seconds = int(current_seconds % 60)
            timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Extract frame using ffmpeg
            output_filename = f"frame_{frame_count:06d}_{timestamp_str.replace(':', '_')}.png"
            output_path = os.path.join(frames_dir, output_filename)
            
            try:
                if FFmpegAvailable:
                    (
                        ffmpeg.input(video_path, ss=timestamp_str)
                        .output(output_path, vframes=1, format="image2", vcodec="png")
                        .overwrite_output()
                        .run(quiet=True, overwrite_output=True)
                    )
                else:
                    # Fallback to OpenCV
                    cap = cv2.VideoCapture(video_path)
                    frame_num = int(current_seconds * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(output_path, frame)
                    cap.release()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    frames_data.append({
                        'timestamp': timestamp_str,
                        'frame_path': output_path,
                        'frame_index': frame_count
                    })
                    if verbose and logger:
                        logger.debug(f"Extracted frame {frame_count} at {timestamp_str}")
                    frame_count += 1
                else:
                    if verbose and logger:
                        logger.warning(f"Frame extraction failed at {timestamp_str}")
                    
            except Exception as e:
                if verbose and logger:
                    logger.warning(f"Error extracting frame at {timestamp_str}: {e}")
            
            current_seconds += frame_interval
            
            # Safety limit
            if frame_count >= 100:
                if verbose and logger:
                    logger.warning("Reached frame limit (100), stopping extraction")
                break
        
        if verbose and logger:
            logger.debug(f"Extracted {len(frames_data)} frames from video")
            
        return frames_data
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting frames: {e}")
        return []


def detect_slide_rectangles_in_frames(
    frames_data: List[Dict],
    each_roi: bool = False,
    logger=None,
    verbose=False,
) -> List[Dict]:
    """
    Detect slide rectangles in each frame using OpenCV.
    If each_roi is True, detect rectangles for each frame individually.
    Otherwise, use first detected ROI for all frames.
    
    Args:
        frames_data: List of frame dictionaries with frame_path
        each_roi: Enable per-frame ROI detection
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        Updated frames_data with roi_coords for each frame
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug(
            f"Detecting slide rectangles in {len(frames_data)} frames "
            f"(each_roi={each_roi})"
        )
    
    updated_frames = []
    first_roi = None
    
    for i, frame_data in enumerate(frames_data):
        frame_path = frame_data['frame_path']
        
        if not os.path.exists(frame_path):
            if verbose and logger:
                logger.warning(f"Frame file not found: {frame_path}")
            continue
            
        # Read frame
        img = cv2.imread(frame_path)
        if img is None:
            if verbose and logger:
                logger.warning(f"Could not read frame: {frame_path}")
            continue
            
        if each_roi or (i == 0):  # Each ROI mode OR first frame
            if verbose and logger:
                logger.debug(f"Detecting ROI for frame {frame_data.get('frame_index', i)}")
                
            roi = detect_slide_rectangle_in_frame(img, logger, verbose)
            if roi:
                first_roi = roi if i == 0 else first_roi
                frame_data['roi_coords'] = roi
        else:
            # Use first detected ROI for all subsequent frames
            if first_roi:
                frame_data['roi_coords'] = first_roi
            else:
                # No ROI detected, use full frame
                height, width = img.shape[:2]
                frame_data['roi_coords'] = (0, 0, width, height)
        
        updated_frames.append(frame_data)
    
    if verbose and logger:
        logger.debug(f"ROI detection completed for {len(updated_frames)} frames")
        
    return updated_frames


def detect_slide_rectangle_in_frame(
    img: np.ndarray,
    logger=None,
    verbose=False,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect slide rectangle in a single frame using hybrid OpenCV methods.
    Combines: Canny edges, Hough lines, color thresholding, morphological ops.
    Returns (x0, y0, x1, y1) coordinates.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    height, width = img.shape[:2]
    frame_area = width * height
    
    # Method 1: Canny Edge Detection + Contours (original method)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Method 2: Morphological Operations to clean edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Method 3: Color-based thresholding (separate foreground/background)
    # Slides often have distinct colors from background
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create mask for non-black regions (typical slide content)
    lower = np.array([0, 0, 50])
    upper = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower, upper)
    
    # Combine edge-based and color-based detection
    combined = cv2.bitwise_or(closed, color_mask)
    
    # Dilate to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(combined, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if verbose and logger:
        logger.debug(f"Found {len(contours)} contours in frame")
    
    # Filter for rectangular shapes (slides)
    rectangles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Lower threshold to catch more candidates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small or too large
            if w < 100 or h < 100:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            # Typical slide aspect ratios (4:3≈1.33, 16:9≈1.78, 16:10≈1.6)
            # Expanded range to handle various presentations
            if 0.5 <= aspect_ratio <= 2.2:
                coverage = area / frame_area
                if 0.2 <= coverage <= 0.98:  # Expanded coverage range
                    rectangles.append((x, y, w, h, area, aspect_ratio))
    
    if verbose and logger:
        logger.debug(f"Found {len(rectangles)} candidate rectangles after filtering")
    
    # Sort by area (prefer larger rectangles)
    rectangles.sort(key=lambda x: x[4], reverse=True)
    
    # Strategy 1: Take largest rectangle (most likely the slide)
    if rectangles:
        x, y, w, h, area, aspect = rectangles[0]
        coverage = area / frame_area
        
        if verbose and logger:
            logger.debug(
                f"Selected rectangle: x={x}, y={y}, w={w}, h={h}, "
                f"aspect={aspect:.2f}, coverage={coverage:.2%}"
            )
        
        return (x, y, x + w, y + h)
    
    # Strategy 2: If no rectangles found, try Hough Line Detection
    if verbose and logger:
        logger.debug("No contours found, attempting Hough line detection")
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )
    
    if lines is not None:
        # Extract horizontal and vertical lines
        h_lines = []  # Horizontal lines
        v_lines = []  # Vertical lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx > dy:  # More horizontal
                h_lines.append((min(y1, y2), max(y1, y2)))
            else:  # More vertical
                v_lines.append((min(x1, x2), max(x1, x2)))
        
        if h_lines and v_lines:
            # Get bounds from lines
            h_lines.sort()
            v_lines.sort()
            
            y_top = h_lines[0][0]
            y_bottom = h_lines[-1][1]
            x_left = v_lines[0][0]
            x_right = v_lines[-1][1]
            
            rect_area = (x_right - x_left) * (y_bottom - y_top)
            coverage = rect_area / frame_area
            
            if 0.2 <= coverage <= 0.98:
                if verbose and logger:
                    logger.debug(
                        f"Hough detection: x={x_left}, y={y_top}, "
                        f"w={x_right-x_left}, h={y_bottom-y_top}, coverage={coverage:.2%}"
                    )
                return (x_left, y_top, x_right, y_bottom)
    
    # Strategy 3: Fallback - assume slide takes up center 80% of frame
    if verbose and logger:
        logger.warning(
            "Could not detect slide rectangle with edge/line detection. "
            "Using default centered region."
        )
    
    margin_w = int(width * 0.1)
    margin_h = int(height * 0.1)
    
    return (margin_w, margin_h, width - margin_w, height - margin_h)


def crop_and_save_slides(
    frames_data: List[Dict],
    output_dir: str,
    base_name: str,
    logger=None,
    verbose=False,
) -> List[Dict]:
    """
    Crop slide rectangles from frames and save as slide images.
    Returns updated frames_data with cropped slide paths.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if verbose and logger:
        logger.debug(f"Cropping and saving slides for {len(frames_data)} frames")
    
    # Create slides subdirectory
    slides_dir = os.path.join(output_dir, f"{base_name}_slides")
    try:
        os.makedirs(slides_dir, exist_ok=True)
    except Exception as e:
        if logger:
            logger.error(f"Failed to create slides directory: {e}")
        return frames_data
    
    cropped_frames = []
    
    for frame_data in frames_data:
        roi_coords = frame_data.get('roi_coords')
        if not roi_coords:
            if verbose and logger:
                logger.warning(f"No ROI coordinates for frame {frame_data.get('frame_index', 'unknown')}")
            continue
            
        frame_path = frame_data['frame_path']
        timestamp = frame_data['timestamp']
        
        # Clean timestamp for filename
        clean_ts = re.sub(r"[:.]", "_", timestamp)
        slide_filename = f"slide_{clean_ts}.png"
        slide_path = os.path.join(slides_dir, slide_filename)
        
        try:
            # Crop frame using ROI
            if FFmpegAvailable:
                crop_w = roi_coords[2] - roi_coords[0]
                crop_h = roi_coords[3] - roi_coords[1]
                crop_x = roi_coords[0]
                crop_y = roi_coords[1]
                
                (
                    ffmpeg.input(frame_path)
                    .filter(
                        "crop",
                        crop_w,
                        crop_h,
                        crop_x,
                        crop_y,
                    )
                    .output(slide_path, vframes=1, format="image2", vcodec="png")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                # Fallback to OpenCV
                img = cv2.imread(frame_path)
                if img is not None:
                    x0, y0, x1, y1 = roi_coords
                    cropped_img = img[y0:y1, x0:x1]
                    cv2.imwrite(slide_path, cropped_img)
            
            # Update frame data with cropped slide path
            cropped_frame_data = frame_data.copy()
            cropped_frame_data['cropped_slide_path'] = slide_path
            cropped_frames.append(cropped_frame_data)
            
            if verbose and logger:
                logger.debug(f"Cropped slide: {slide_path}")
                
        except Exception as e:
            if verbose and logger:
                logger.warning(f"Failed to crop slide for {timestamp}: {e}")
            cropped_frame_data = frame_data.copy()
            cropped_frame_data['cropped_slide_path'] = None
            cropped_frames.append(cropped_frame_data)
    
    if verbose and logger:
        logger.debug(f"Created {len(cropped_frames)} cropped slide images")
        
    return cropped_frames
