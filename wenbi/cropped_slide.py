"""
CROPPED-SLIDE Method Implementation (Type 2)
Detect ROI → Crop Slides → OCR → Combine with Audio
"""

import os
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

# Import ultralytics for RT-DETR slide detection
try:
    from ultralytics import RTDETR
    RTDETRAvailable = True
except ImportError:
    RTDETRAvailable = False
    RTDETR = None


def detect_slide_roi(video_path, roi_string, output_dir, logger, verbose):
    """
    Detect or parse ROI coordinates.
    
    Args:
        roi_string: None (auto-detect) or "x0,y0,x1,y1" (manual)
    
    Returns:
        (x0, y0, x1, y1) tuple
    """
    
    if roi_string:
        # Manual ROI provided
        if verbose:
            logger.debug(f"Parsing manual ROI: {roi_string}")
        
        try:
            coords = [int(x.strip()) for x in roi_string.split(",")]
            if len(coords) != 4:
                raise ValueError(f"Invalid ROI format: expected 4 values, got {len(coords)}")
            
            x0, y0, x1, y1 = coords
            if x0 >= x1 or y0 >= y1:
                raise ValueError(f"Invalid ROI: x0 must be < x1, y0 must be < y1")
            
            if verbose:
                logger.debug(f"ROI parsed: ({x0}, {y0}, {x1}, {y1})")
            
            return (x0, y0, x1, y1)
        
        except ValueError as e:
            print(f"Error: ROI parse error: {e}")
            raise SystemExit(1)
    
    else:
        # Auto-detect with RT-DETR using the working algorithm from video_slides.py
        if verbose:
            logger.debug("Auto-detecting ROI with RTDETR...")
        
        # Extract first frame for detection
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Cannot read first frame from video")
            raise SystemExit(1)
        
        # Save first frame temporarily for detection
        temp_frame_path = os.path.join(output_dir, "_temp_frame_for_roi.jpg")
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # Use working detect_slide_roi_with_yolo function
            roi = detect_slide_roi_with_yolo(temp_frame_path, logger, verbose)
            
            if verbose:
                logger.debug(f"ROI detection result: {roi}")
            
            if roi is None:
                if verbose:
                    logger.debug("ROI detection failed, using full frame fallback")
                h, w = frame.shape[:2]
                return (0, 0, w, h)
            
            return roi
        
        except Exception as e:
            if verbose:
                logger.warning(f"RTDETR detection failed: {e}, falling back to full frame")
            h, w = frame.shape[:2]
            return (0, 0, w, h)
            
            return roi
        
        except Exception as e:
            if verbose:
                logger.warning(f"RTDETR detection failed: {e}, falling back to full frame")
            h, w = frame.shape[:2]
            return (0, 0, w, h)
        
        finally:
            # Clean up temp frame
            try:
                os.remove(temp_frame_path)
            except:
                pass


def detect_slide_roi_with_yolo(
    frame_path: str, logger=None, verbose=False
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect slide area using RT-DETR-v2 object detection model.
    Detects speaker box in corners and creates mask to remove it,
    isolating the slide area. Fast and efficient detection.
    
    Args:
        frame_path: Path to frame image file
        logger: Optional logger instance
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (x0, y0, x1, y1) coordinates, or None if no detection (fallback to full-frame)
    """
    if not RTDETRAvailable:
        if logger:
            logger.warning("RTDETR not available. Install with: rye add ultralytics")
        return None
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        img = cv2.imread(frame_path)
        if img is None:
            if logger:
                logger.warning(f"Could not read frame: {frame_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        
        if verbose and logger:
            logger.debug(f"Loading RT-DETR model for slide detection")
        
        # Load pre-trained RT-DETR model (fast object detection)
        if not RTDETRAvailable:
            if logger:
                logger.warning("RTDETR not available. Install with: rye add ultralytics")
            return None
        
        model = RTDETR("rtdetr-l.pt")
        
        if verbose and logger:
            logger.debug(f"Running RT-DETR detection on frame: {frame_path}")
        
        # Run detection (no prompts needed, model auto-detects objects)
        results = model(frame_path, verbose=False)
        
        if not results or len(results) == 0:
            if logger and verbose:
                logger.debug("RT-DETR detection returned no results, using full-frame fallback")
            return None
        
        result = results[0]
        
        # Check if bounding boxes exist
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            if verbose and logger:
                logger.debug("No objects detected by RT-DETR, using full-frame fallback")
            return None
        
        if verbose and logger:
            logger.debug(f"RT-DETR detected {len(result.boxes)} objects")
        
        # Define corner regions to identify speaker box
        corner_regions = {
            "top-right": (int(img_width * 0.6), 0, img_width, int(img_height * 0.4)),
            "top-left": (0, 0, int(img_width * 0.4), int(img_height * 0.4)),
            "bottom-right": (int(img_width * 0.6), int(img_height * 0.6), img_width, img_height),
            "bottom-left": (0, int(img_height * 0.6), int(img_width * 0.4), img_height),
        }
        
        speaker_box = None
        corner_location = None
        
        # Find largest object in corner regions (likely speaker)
        for box in result.boxes:
            x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().astype(int)
            box_area = (x1 - x0) * (y1 - y0)
            
            # Check if box is in any corner region
            for corner_name, (cx0, cy0, cx1, cy1) in corner_regions.items():
                # Check if box centroid or significant overlap is in corner
                box_cx = (x0 + x1) / 2
                box_cy = (y0 + y1) / 2
                
                if cx0 <= box_cx <= cx1 and cy0 <= box_cy <= cy1:
                    # This box is in a corner, likely speaker
                    if speaker_box is None or box_area > (speaker_box[2] - speaker_box[0]) * (speaker_box[3] - speaker_box[1]):
                        speaker_box = (x0, y0, x1, y1)
                        corner_location = corner_name
                        
                        if verbose and logger:
                            logger.debug(f"RT-DETR detected object in {corner_name} corner: {speaker_box} (area: {box_area})")
                    break
        
        # New approach: Use geometric inference based on speaker box detection
        # If we detect a speaker box in a corner, infer slide area as the remaining area
        
        if speaker_box is not None and corner_location:
            # Define slide area based on speaker box location
            sx0, sy0, sx1, sy1 = speaker_box
            
            if corner_location == "top-right":
                # Slide area: everything except top-right corner
                slide_roi = (0, 0, int(img_width * 0.85), img_height)
            elif corner_location == "top-left":
                # Slide area: everything except top-left corner  
                slide_roi = (int(img_width * 0.15), 0, img_width, img_height)
            elif corner_location == "bottom-right":
                # Slide area: everything except bottom-right corner
                slide_roi = (0, 0, int(img_width * 0.85), int(img_height * 0.85))
            elif corner_location == "bottom-left":
                # Slide area: everything except bottom-left corner
                slide_roi = (int(img_width * 0.15), 0, img_width, int(img_height * 0.85))
            else:
                # Default to 85% width centered
                margin_w = int(img_width * 0.075)
                slide_roi = (margin_w, 0, img_width - margin_w, img_height)
            
            if verbose and logger:
                logger.debug(f"Inferred slide ROI from speaker box at {corner_location}: {slide_roi}")
            
            return slide_roi
        else:
            # No speaker box detected, try to find largest object
            
            # Find the largest detected object overall
            all_boxes = []
            for box in result.boxes:
                x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().astype(int)
                box_area = (x1 - x0) * (y1 - y0)
                aspect_ratio = (x1 - x0) / (y1 - y0) if (y1 - y0) > 0 else 0
                
                # Prefer objects with slide-like aspect ratios (4:3 to 16:9)
                aspect_score = 1.0
                if 1.2 <= aspect_ratio <= 2.0:
                    aspect_score = 2.0  # Boost slide-like aspect ratios
                
                all_boxes.append({
                    'box': (x0, y0, x1, y1),
                    'area': box_area,
                    'aspect_score': aspect_score,
                    'center': ((x0 + x1) / 2, (y0 + y1) / 2)
                })
            
            if all_boxes:
                # Score by area * aspect_score, prefer centered objects
                best_box = max(all_boxes, key=lambda x: (
                    x['area'] * x['aspect_score'],
                    -abs(x['center'][0] - img_width / 2),  # Prefer horizontally centered
                    -abs(x['center'][1] - img_height / 2)   # Prefer vertically centered
                ))
                
                slide_roi = tuple(int(coord) for coord in best_box['box'])
                
                if verbose and logger:
                    logger.debug(f"Selected best slide ROI: {slide_roi}")
                
                return slide_roi
        
        # Final fallback: use 85% width centered area
        margin_w = int(img_width * 0.075)  # 7.5% margin on each side
        slide_roi = (margin_w, 0, img_width - margin_w, img_height)
        
        if verbose and logger:
            logger.debug(f"Fallback slide ROI: {slide_roi}")
        
        return slide_roi
        
    except Exception as e:
        if logger:
            logger.error(f"Error in RT-DETR slide detection: {e}")
        return None


def crop_slides_from_frames(deduplicated_frames, roi_coords, output_dir, 
                            base_name, logger, verbose):
     """Crop frames to ROI region and save."""
     
     if verbose:
         logger.debug(f"Cropping frames to ROI {roi_coords}...")
     
     x0, y0, x1, y1 = roi_coords
     crop_dir = os.path.join(output_dir, f"{base_name}_cropped-slide")
     os.makedirs(crop_dir, exist_ok=True)
     
     cropped_frames = []
     
     for frame_dict in deduplicated_frames:
         timestamp = frame_dict["timestamp"]
         original_path = frame_dict["frame_path"]
         
         try:
             # Load and crop
             img = cv2.imread(original_path)
             if img is None:
                 logger.warning(f"Cannot read frame: {original_path}")
                 continue
             
             cropped = img[y0:y1, x0:x1]
             
             # Save cropped frame
             filename = os.path.basename(original_path)
             cropped_path = os.path.join(crop_dir, f"cropped_{filename}")
             cv2.imwrite(cropped_path, cropped)
             
             cropped_frames.append({
                 "timestamp": timestamp,
                 "original_frame": original_path,
                 "cropped_frame": cropped_path
             })
             
             if verbose:
                 logger.debug(f"Cropped {timestamp}: {cropped_path}")
         
         except Exception as e:
             logger.warning(f"Failed to crop {timestamp}: {e}")
     
     if not cropped_frames:
         print("Error: No frames successfully cropped")
         raise SystemExit(1)
     
     if verbose:
         logger.debug(f"Cropped {len(cropped_frames)} frames to {crop_dir}")
     
     return cropped_frames


def execute_cropped_slide_method(video_path, deduplicated_frames, roi_string, 
                                 output_dir, no_ocr, no_clean, base_name,
                                 cite_timestamps, llm, chunk_length, max_tokens,
                                 timeout, temperature, lang, transcribe_model,
                                 multi_language, transcribe_lang, logger, verbose):
    """
    Execute CROPPED-SLIDE method workflow.
    Returns: (combine_md_path, combine_clean_md_path)
    """
    from wenbi.cli import (
        run_marker_pdf_on_image, image_to_base64, embed_frames_as_base64,
        clean_combined_markdown
    )
    from wenbi.main import process_input
    from wenbi.model import combine_speech_and_slides
    
    if verbose:
        logger.debug("=== TYPE 2: CROPPED-SLIDE Method ===")
    
    # Step 1: Detect ROI with RTDETR (finds largest rectangle = slide area)
    roi_coords = detect_slide_roi(video_path, roi_string, output_dir, logger, verbose)
    
    # Step 2: Crop slides to ROI
    cropped_frames = crop_slides_from_frames(
        deduplicated_frames, roi_coords, output_dir, base_name, logger, verbose
    )
    
    # Step 3: OCR cropped slides
    if verbose:
        logger.debug("Step 3: Running OCR on cropped slides...")
    
    if no_ocr:
        if verbose:
            logger.debug("--no-ocr: Embedding cropped slides as base64...")
        
        # Convert cropped frames for embedding
        embed_frames = [
            {"timestamp": f["timestamp"], "frame_path": f["cropped_frame"]}
            for f in cropped_frames
        ]
        
        slides_md = embed_frames_as_base64(
            embed_frames, output_dir, f"{base_name}_cropped-slide", logger, verbose
        )
    else:
        # OCR each cropped slide
        markdown_sections = []
        
        for idx, frame_dict in enumerate(cropped_frames, 1):
            timestamp = frame_dict["timestamp"]
            cropped_path = frame_dict["cropped_frame"]
            
            if verbose:
                logger.debug(f"OCR cropped slide {idx}/{len(cropped_frames)}: {timestamp}")
            
            ocr_result = run_marker_pdf_on_image(
                cropped_path, output_dir, verbose, logger
            )
            
            section = f"\n### **{timestamp}**\n"
            
            if ocr_result["success"]:
                section += ocr_result["text"]
                
                # Add base64 images if any
                for filename, b64 in ocr_result["base64_images"].items():
                    section += f'\n<img src="data:image/png;base64,{b64}" />\n'
            else:
                # OCR failed, embed as base64
                if verbose:
                    logger.warning(f"OCR failed for {timestamp}, using base64")
                
                b64 = image_to_base64(cropped_path)
                if b64:
                    section += f'<img src="data:image/png;base64,{b64}" />\n'
            
            markdown_sections.append(section)
        
        slides_md = os.path.join(output_dir, f"{base_name}_cropped-slide.md")
        with open(slides_md, "w", encoding="utf-8") as f:
            f.write("".join(markdown_sections))
        
        if verbose:
            logger.debug(f"Cropped slide OCR completed: {slides_md}")
    
    # Step 4: Rewrite audio
    if verbose:
        logger.debug("Step 3: Processing audio...")
    
    params = {
        "output_dir": output_dir,
        "llm": llm,
        "chunk_length": chunk_length,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "temperature": temperature,
        "lang": lang,
        "transcribe_model": transcribe_model,
        "multi_language": multi_language,
        "transcribe_lang": transcribe_lang,
        "cite_timestamps": cite_timestamps,
        "verbose": verbose,
        "subcommand": "rewrite"
    }
    
    result = process_input(
        file_path=video_path,
        url="",
        **params
    )
    
    audio_markdown = result[0]
    if verbose:
        logger.debug("Audio processing completed")
    
    # Step 5: Combine
    if verbose:
        logger.debug("Step 4: Combining cropped slide and audio markdown...")
    
    with open(slides_md, "r", encoding="utf-8") as f:
        slides_content = f.read()
    
    combined_markdown = combine_speech_and_slides(
        speech_markdown=audio_markdown,
        slides_markdown=slides_content,
        verbose=verbose
    )
    
    combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
    with open(combine_md, "w", encoding="utf-8") as f:
        f.write(combined_markdown)
    
    if verbose:
        logger.debug(f"Combined markdown: {combine_md}")
    
    # Step 6: Clean (if not --no-clean)
    if no_clean:
        combine_clean_md = None
        if verbose:
            logger.debug("--no-clean: Skipping clean phase")
    else:
        combine_clean_md = clean_combined_markdown(
            combine_md, output_dir, base_name, logger, verbose
        )
    
    return combine_md, combine_clean_md
