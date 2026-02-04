"""
CROPPED-SLIDE Method Implementation (Type 2)
Detect ROI → Crop Slides → OCR → Combine with Audio
"""

import os
import logging
import cv2
from typing import List, Dict, Tuple, Optional


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
        # Auto-detect with RTDETR
        if verbose:
            logger.debug("Auto-detecting ROI with RTDETR...")
        
        try:
            from ultralytics import RTDETR
            
            # Load model (auto-download if not present)
            if verbose:
                logger.debug("Loading RTDETR model (rtdetr-l.pt)...")
            
            model = RTDETR("rtdetr-l.pt")
            
            # Extract first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Cannot read first frame from video")
                raise SystemExit(1)
            
            # Run detection
            if verbose:
                logger.debug("Running RTDETR detection on first frame...")
            
            results = model(frame)
            
            # Parse detections for slide rectangle (largest box)
            detections = results[0]
            if len(detections.boxes) == 0:
                logger.warning("No objects detected, falling back to full frame")
                h, w = frame.shape[:2]
                return (0, 0, w, h)
            
            # Get largest bounding box (assuming it's the slide)
            boxes = detections.boxes.xyxy.cpu().numpy()
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            max_idx = areas.index(max(areas))
            x0, y0, x1, y1 = boxes[max_idx]
            
            roi = (int(x0), int(y0), int(x1), int(y1))
            if verbose:
                logger.debug(f"Detected ROI: {roi}")
            
            return roi
        
        except ImportError:
            print("Error: ultralytics not installed. Run: rye add ultralytics")
            raise SystemExit(1)
        except Exception as e:
            logger.warning(f"RTDETR detection failed: {e}, falling back to full frame")
            cap = cv2.VideoCapture(video_path)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            return (0, 0, w, h)


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
    
    # Step 1: Detect ROI
    roi_coords = detect_slide_roi(video_path, roi_string, output_dir, logger, verbose)
    
    # Step 2: Crop slides
    cropped_frames = crop_slides_from_frames(
        deduplicated_frames, roi_coords, output_dir, base_name, logger, verbose
    )
    
    # Step 3: OCR cropped slides
    if verbose:
        logger.debug("Step 2: Running OCR on cropped slides...")
    
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
        logger=logger,
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
