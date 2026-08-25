"""
PPT Method Implementation (Type 3)
Load/Convert PDF/PPT/Images → Extract Timestamps → OCR → Combine with Audio
"""

import os
import logging
import subprocess
from typing import List, Tuple, Dict
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_and_convert_pdf(ppt_pdf_path, output_dir, logger, verbose):
    """
    Load PPT, PDF, or image file. Convert PPT to PDF if needed.
    For images, create a simple workflow to process them as slides.
    Returns path to PDF file or the original path for images.
    """
    
    if not os.path.exists(ppt_pdf_path):
        print(f"Error: File not found: {ppt_pdf_path}")
        raise SystemExit(1)
    
    file_ext = os.path.splitext(ppt_pdf_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(ppt_pdf_path))[0]
    
    logger.debug(f"Loading: {ppt_pdf_path}")
    
    if file_ext == ".pptx":
        logger.debug("Converting PPTX to PDF with LibreOffice...")
        
        try:
            # Validate PPTX file using python-pptx
            from pptx import Presentation
            
            prs = Presentation(ppt_pdf_path)
            num_slides = len(prs.slides)
            
            if verbose:
                logger.debug(f"PPTX file has {num_slides} slides")
            
            # Convert PPTX to PDF using LibreOffice headless
            pdf_path = os.path.join(output_dir, f"{base_name}_converted.pdf")
            
            cmd = [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", output_dir,
                ppt_pdf_path
            ]
            
            if verbose:
                logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error: LibreOffice conversion failed: {result.stderr}")
                raise SystemExit(1)
            
            # LibreOffice outputs with original name, rename if needed
            generated_pdf = os.path.join(output_dir, f"{base_name}.pdf")
            if os.path.exists(generated_pdf) and generated_pdf != pdf_path:
                os.rename(generated_pdf, pdf_path)
            
            if not os.path.exists(pdf_path):
                print("Error: PDF conversion failed: output file not found")
                raise SystemExit(1)
            
            logger.debug(f"PDF converted: {pdf_path}")
            return pdf_path
        
        except ImportError:
            print("Error: python-pptx not installed. Run: rye add python-pptx")
            raise SystemExit(1)
        except subprocess.TimeoutExpired:
            print("Error: LibreOffice conversion timeout (>5 min)")
            raise SystemExit(1)
        except Exception as e:
            print(f"Error: Failed to convert PPTX to PDF: {e}")
            raise SystemExit(1)
    
    elif file_ext == ".pdf":
        logger.debug("PDF file detected, using directly")
        return ppt_pdf_path
    
    elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        logger.debug(f"Image file detected: {file_ext}")
        # For image inputs, return the image path directly
        return ppt_pdf_path
    
    elif file_ext == ".odp":
        logger.debug("OpenDocument Presentation (.odp) file detected")
        return ppt_pdf_path  # Will try to process as-is for now
    
    else:
        print(f"Error: Unsupported file format: {file_ext} (expected .pdf, .pptx, .png, .jpg, .jpeg, .bmp, .tiff, .webp, .odp)")
        raise SystemExit(1)


def _detect_yolo11n(frame_path, logger, verbose):
    """YOLO11n via ultralytics, CPU-only. Returns (x0,y0,x1,y1) or None.
    Lazy-imports ultralytics. Auto-downloads yolo11n.pt on first call.
    Filters COCO classes 62 (tv) + 63 (laptop), conf>=0.25, largest box."""
    try:
        from ultralytics import YOLO
    except ImportError:
        if verbose:
            logger.debug("slide-crop: ultralytics not installed, skipping YOLO11n")
        return None
    try:
        model = YOLO("yolo11n.pt")
        results = model(frame_path, device="cpu", classes=[62, 63], verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        # filter conf >= 0.25, pick largest area
        best = None
        best_area = 0
        for (x0, y0, x1, y1), c in zip(boxes, confs):
            if c < 0.25:
                continue
            area = (x1 - x0) * (y1 - y0)
            if area > best_area:
                best_area = area
                best = (int(x0), int(y0), int(x1), int(y1))
        if verbose and best:
            logger.debug(f"slide-crop: YOLO11n found box {best} (conf filter, area={best_area:.0f})")
        return best
    except Exception as e:
        if verbose:
            logger.debug(f"slide-crop: YOLO11n failed: {e}")
        return None


def _detect_paddle(frame_path, logger, verbose):
    """PaddleDetection PP-YOLOE, CPU-only. Returns (x0,y0,x1,y1) or None.
    Lazy-imports. Only used as secondary fallback when YOLO11n unavailable.
    NOTE: PaddleDetection's python infer API requires a model dir; this is a
    best-effort path. If paddlepaddle/paddledet not importable, return None."""
    try:
        import paddle
        # ponytail: PaddleDetection needs a model dir + deploy.python.infer;
        # without a configured model dir this path returns None gracefully.
        # Users wanting Paddle fallback must set WENBI_PADDLE_MODEL_DIR env var.
        model_dir = os.environ.get("WENBI_PADDLE_MODEL_DIR")
        if not model_dir or not os.path.isdir(model_dir):
            return None
        import sys
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from deploy.python.infer import Detector
        detector = Detector(
            model_dir=model_dir, device="CPU", run_mode="paddle", cpu_threads=4
        )
        results = detector.predict_image([frame_path], visual=False)
        # filter classes 62 (tv) + 63 (laptop) — PaddleDetection COCO ids match
        best = None
        best_area = 0
        for item in (results or []):
            boxes = item.get("boxes", item.get("bbox", []))
            classes = item.get("classes", item.get("category_id", []))
            for box, cls in zip(boxes, classes):
                if int(cls) not in (62, 63):
                    continue
                x0, y0, x1, y1 = [int(v) for v in box[:4]]
                area = (x1 - x0) * (y1 - y0)
                if area > best_area:
                    best_area = area
                    best = (x0, y0, x1, y1)
        if verbose and best:
            logger.debug(f"slide-crop: PaddleDetection found box {best}")
        return best
    except Exception as e:
        if verbose:
            logger.debug(f"slide-crop: PaddleDetection failed: {e}")
        return None


def _detect_heuristic(frame_path, logger, verbose):
    """Largest-bright-rectangle heuristic. cv2-only, zero deps. Always available.
    Returns (x0,y0,x1,y1) or None."""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Otsu threshold to isolate bright slide region
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        min_area = 0.10 * h * w  # at least 10% of frame
        for c in contours:
            x0, y0, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < min_area:
                continue
            aspect = cw / max(ch, 1)
            if aspect < 1.2:  # slides wider than tall; skip the lecturer square
                continue
            if area > best_area:
                best_area = area
                best = (x0, y0, x0 + cw, y0 + ch)
        if verbose and best:
            logger.debug(f"slide-crop: heuristic found box {best} (area={best_area:.0f})")
        return best
    except Exception as e:
        if verbose:
            logger.debug(f"slide-crop: heuristic failed: {e}")
        return None


def detect_slide_region(frame_path, logger, verbose):
    """Detect the projected-slide rectangle in one frame.
    Engine chain: YOLO11n (primary) -> PaddleDetection (secondary) -> heuristic (fallback).
    Returns (x0, y0, x1, y1) or None (=> use full frame)."""
    box = _detect_yolo11n(frame_path, logger, verbose)
    if box is None:
        box = _detect_paddle(frame_path, logger, verbose)
    if box is None:
        box = _detect_heuristic(frame_path, logger, verbose)
    return box


def crop_slides_region(deduplicated_frames, output_dir, base_name, logger, verbose):
    """Crop the slide region out of each deduplicated frame.
    Returns a NEW list of frame dicts with same shape:
        {"frame_path": <cropped_path_or_original>, "timestamp": str, "original_frame_path": str}
    Cropped images saved to output_dir/<base_name>_cropped/.
    Frames where detection fails keep their original frame_path."""
    crop_dir = os.path.join(output_dir, f"{base_name}_cropped")
    os.makedirs(crop_dir, exist_ok=True)

    cropped_frames = []
    cropped_count = 0
    for frame_data in deduplicated_frames:
        ts = frame_data["timestamp"]
        src = frame_data["frame_path"]
        box = detect_slide_region(src, logger, verbose)

        if box is None:
            if verbose:
                logger.debug(f"slide-crop: {ts}: no box, using full frame")
            cropped_frames.append({
                "frame_path": src,
                "timestamp": ts,
                "original_frame_path": src,
            })
            continue

        x0, y0, x1, y1 = box
        try:
            img = cv2.imread(src)
            if img is None:
                if verbose:
                    logger.debug(f"slide-crop: {ts}: cannot reread, using full frame")
                cropped_frames.append({
                    "frame_path": src,
                    "timestamp": ts,
                    "original_frame_path": src,
                })
                continue
            cropped = img[y0:y1, x0:x1]
            out_name = f"crop_{os.path.splitext(os.path.basename(src))[0]}.png"
            out_path = os.path.join(crop_dir, out_name)
            cv2.imwrite(out_path, cropped)
            cropped_count += 1
            if verbose:
                logger.debug(f"slide-crop: {ts}: cropped to {out_path} ({x1-x0}x{y1-y0})")
            cropped_frames.append({
                "frame_path": out_path,
                "timestamp": ts,
                "original_frame_path": src,
            })
        except Exception as e:
            if verbose:
                logger.debug(f"slide-crop: {ts}: crop failed {e}, using full frame")
            cropped_frames.append({
                "frame_path": src,
                "timestamp": ts,
                "original_frame_path": src,
            })

    logger.debug(f"slide-crop: cropped {cropped_count}/{len(deduplicated_frames)} frames")
    return cropped_frames


def process_images_as_slides(ppt_path, deduplicated_frames, output_dir, no_ocr, base_name, cite_timestamps, logger, verbose):
    """
    Process image files directly as slides.
    Map each image to timestamps from deduplicated frames.
    """
    from wenbi.cli import image_to_base64, run_marker_pdf_on_image
    
    if verbose:
        logger.debug(f"=== Processing image input: {ppt_path} ===")
    
    markdown_sections = []
    
    for frame_idx, frame_data in enumerate(deduplicated_frames):
        timestamp = frame_data["timestamp"]
        
        if verbose:
            logger.debug(f"Processing timestamp {frame_idx + 1}/{len(deduplicated_frames)}: {timestamp}")
        
        section = f"\n### **{timestamp}**\n"
        
        if no_ocr:
            # Embed image as base64
            b64 = image_to_base64(ppt_path)
            if b64:
                section += f'<img src="data:image/png;base64,{b64}" />\n'
        else:
            # Run OCR on image
            ocr_result = run_marker_pdf_on_image(
                ppt_path, output_dir, verbose, logger
            )
            
            if ocr_result["success"]:
                section += ocr_result["text"]
                
                # Add base64 images if any
                for filename, b64 in ocr_result["base64_images"].items():
                    section += f'\n<img src="data:image/png;base64,{b64}" />\n'
            else:
                # OCR failed, fallback to base64
                if verbose:
                    logger.warning(f"OCR failed for {timestamp}, using base64")
                
                b64 = image_to_base64(ppt_path)
                if b64:
                    section += f'<img src="data:image/png;base64,{b64}" />\n'
        
        markdown_sections.append(section)
    
    return markdown_sections


def extract_timestamps_for_pdf_pages(deduplicated_frames):
    """
    Extract timestamps from deduplicated frames in order.
    Returns list of timestamps.
    """
    timestamps = [frame["timestamp"] for frame in deduplicated_frames]
    return timestamps


def match_pdf_pages_to_frames(pdf_path, deduplicated_frames, logger, verbose, ssim_threshold=0.8):
    """
    Match PDF pages to deduplicated frames using OpenCV SSIM comparison.
    Uses sequential matching with skip logic - moves to next if no match found.
    
    Returns: List of tuples (pdf_page_idx, frame_data) for matched pairs
    """
    import PyPDF2
    
    logger.debug(f"Step 4: Starting PDF-to-Frame matching (SSIM threshold: {ssim_threshold})")
    
    try:
        # Get PDF page count
        with open(pdf_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            num_pages = len(pdf.pages)
        
        num_frames = len(deduplicated_frames)
        logger.debug(f"PDF has {num_pages} pages, Found {num_frames} deduplicated frames")
        
        matched_pairs = []
        pdf_idx = 0
        frame_idx = 0
        
        while pdf_idx < num_pages and frame_idx < num_frames:
            logger.debug(f"Matching PDF page {pdf_idx + 1}/{num_pages} with frame {frame_idx + 1}/{num_frames}")
            
            # Convert current PDF page to image
            try:
                pdf_page_img = convert_pdf_page_to_image(pdf_path, pdf_idx)
                if verbose:
                    logger.debug(f"  Converted PDF page {pdf_idx + 1} to image")
            except Exception as e:
                logger.warning(f"  Failed to convert PDF page {pdf_idx + 1}: {e}, skipping")
                pdf_idx += 1
                continue
            
            # Get current frame image
            frame_data = deduplicated_frames[frame_idx]
            frame_path = frame_data["frame_path"]
            
            try:
                frame_img = cv2.imread(frame_path)
                if frame_img is None:
                    logger.warning(f"  Failed to read frame {frame_idx + 1}: {frame_path}, skipping")
                    frame_idx += 1
                    continue
                if verbose:
                    logger.debug(f"  Read frame {frame_idx + 1}: {frame_path}")
            except Exception as e:
                logger.warning(f"  Failed to read frame {frame_idx + 1}: {e}, skipping")
                frame_idx += 1
                continue
            
            # Convert images to grayscale and compute SSIM
            pdf_gray = cv2.cvtColor(np.array(pdf_page_img), cv2.COLOR_RGB2GRAY)
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            
            # Resize both images to same size for comparison
            h, w = pdf_gray.shape
            frame_resized = cv2.resize(frame_gray, (w, h))
            
            # Calculate SSIM
            try:
                similarity = ssim(pdf_gray, frame_resized, data_range=255)
                logger.debug(f"  SSIM score: {similarity:.4f} (threshold: {ssim_threshold})")
                
                if similarity >= ssim_threshold:
                    # Match found!
                    logger.debug(f"  ✓ MATCH FOUND: PDF page {pdf_idx + 1} ↔ Frame {frame_idx + 1} ({frame_data['timestamp']})")
                    matched_pairs.append((pdf_idx, frame_data))
                    pdf_idx += 1
                    frame_idx += 1
                else:
                    # No match, try next frame
                    if verbose:
                        logger.debug(f"  ✗ No match, trying next frame")
                    frame_idx += 1
            
            except Exception as e:
                logger.warning(f"  Error computing SSIM: {e}, skipping frame")
                frame_idx += 1
        
        logger.debug(f"Matching complete: {len(matched_pairs)} pairs matched out of {num_pages} pages and {num_frames} frames")
        
        if len(matched_pairs) == 0:
            logger.warning("No PDF pages matched with any frames!")
        
        return matched_pairs
    
    except ImportError:
        print("Error: PyPDF2 not installed. Run: rye add PyPDF2")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: Failed to match PDF pages to frames: {e}")
        raise SystemExit(1)


def convert_pdf_page_to_image(pdf_path, page_idx):
    """Convert single PDF page to PIL Image."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        pix = page.get_pixmap()
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    
    except ImportError:
        # Fallback: use pdf2image
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, first_page=page_idx+1, last_page=page_idx+1)
            return images[0]
        except ImportError:
            print("Error: PyMuPDF or pdf2image not installed. Run: rye add PyMuPDF")
            raise SystemExit(1)


def build_slides_markdown(deduplicated_frames, ppt_path, output_dir, no_ocr,
                           base_name, logger, verbose, ssim_threshold=0.8):
    """
    Build slides markdown from deduplicated frames + optional slides file.
    - ppt_path None/empty  -> TYPE 1: embed frames as base64 (no OCR)
    - ppt_path <file>      -> TYPE 3: load/convert, SSIM-match PDF pages to
                              frames, OCR each matched page (or base64 if no_ocr)
    Returns: path to the slides markdown file.
    """
    from wenbi.cli import run_marker_pdf_on_image, image_to_base64, embed_frames_as_base64
    import tempfile
    import os

    # TYPE 1: frame base64-embed, no OCR
    if not ppt_path:
        if verbose:
            logger.debug("TYPE 1: embedding frames as base64 (no OCR)")
        return embed_frames_as_base64(
            deduplicated_frames, output_dir, base_name, logger, verbose
        )

    # TYPE 3: OCR slides file, match to frames
    if verbose:
        logger.debug("=== TYPE 3: PPT Method (slides file) ===")
        logger.debug(f"PPT/PDF: {ppt_path}")
        logger.debug(f"Deduplicated frames count: {len(deduplicated_frames)}")

    file_ext = os.path.splitext(ppt_path)[1].lower()

    if file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        if verbose:
            logger.debug("Processing single image file as slide")
        markdown_sections = process_images_as_slides(
            ppt_path, deduplicated_frames, output_dir, no_ocr, base_name,
            True, logger, verbose  # cite_timestamps forced True for combine
        )
        ppt_md = os.path.join(output_dir, f"{base_name}_ppt.md")
        with open(ppt_md, "w", encoding="utf-8") as f:
            f.write("".join(markdown_sections))
        logger.debug(f"Image processing completed: {ppt_md}")
        return ppt_md

    # PDF/PPT input
    logger.debug("Step 1: Loading and converting PDF/PPT file...")
    pdf_path = load_and_convert_pdf(ppt_path, output_dir, logger, verbose)
    logger.debug(f"Loaded PDF: {pdf_path}")

    logger.debug("Step 2: Matching PDF pages to video frames using OpenCV SSIM...")
    matched_pairs = match_pdf_pages_to_frames(
        pdf_path, deduplicated_frames, logger, verbose,
        ssim_threshold=ssim_threshold,
    )
    if not matched_pairs:
        print("Error: No PDF pages matched with video frames. Check your inputs.")
        raise SystemExit(1)
    logger.debug(f"Successfully matched {len(matched_pairs)} PDF pages to frames")

    logger.debug("Step 3: Running OCR on matched PDF pages...")
    temp_img_dir = tempfile.mkdtemp(prefix="ppt_images_")
    try:
        markdown_sections = []
        for pair_idx, (pdf_page_idx, frame_data) in enumerate(matched_pairs):
            timestamp = frame_data["timestamp"]
            if verbose:
                logger.debug(f"Processing matched pair {pair_idx + 1}/{len(matched_pairs)}: PDF page {pdf_page_idx + 1} ↔ {timestamp}")
            try:
                page_image = convert_pdf_page_to_image(pdf_path, pdf_page_idx)
                temp_img_path = os.path.join(temp_img_dir, f"pdf_page_{pdf_page_idx}.png")
                page_image.save(temp_img_path)

                section = f"\n### **{timestamp}**\n"
                if no_ocr:
                    logger.debug(f"  Embedding page {pdf_page_idx + 1} as base64 (--no-ocr)")
                    b64 = image_to_base64(temp_img_path)
                    if b64:
                        section += f'<img src="data:image/png;base64,{b64}" />\n'
                else:
                    logger.debug(f"  Running marker OCR on page {pdf_page_idx + 1}...")
                    ocr_result = run_marker_pdf_on_image(
                        temp_img_path, output_dir, verbose, logger
                    )
                    if ocr_result["success"]:
                        logger.debug(f"  OCR successful for page {pdf_page_idx + 1}")
                        section += ocr_result["text"]
                        for filename, b64 in ocr_result["base64_images"].items():
                            section += f'\n<img src="data:image/png;base64,{b64}" />\n'
                    else:
                        logger.warning(f"  OCR failed for page {pdf_page_idx + 1}, using base64")
                        b64 = image_to_base64(temp_img_path)
                        if b64:
                            section += f'<img src="data:image/png;base64,{b64}" />\n'
                markdown_sections.append(section)
                try:
                    os.remove(temp_img_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Error processing page {pdf_page_idx + 1}: {e}")
                raise SystemExit(1)
        ppt_md = os.path.join(output_dir, f"{base_name}_ppt.md")
        with open(ppt_md, "w", encoding="utf-8") as f:
            f.write("".join(markdown_sections))
        logger.debug(f"PDF OCR completed: {ppt_md}")
        return ppt_md
    finally:
        import shutil
        try:
            shutil.rmtree(temp_img_dir)
        except:
            pass


def parse_time_to_seconds(time_str: str) -> int:
    """
    Convert HH:MM:SS to seconds
    """
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return int(hours * 3600 + minutes * 60 + seconds)
        return 0
    except:
        return 0


def _seconds_to_display(seconds: float) -> str:
    """Format seconds as HH:MM:SS (no millis), matching bilingual.py format."""
    seconds = max(float(seconds or 0), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_vtt_to_segments(vtt_path: str) -> list[dict]:
    """Parse a WebVTT file into [{'start': float, 'end': float, 'text': str}, ...].

    Strips <v Speaker> tags from text. Handles both plain VTT and speaker-tagged VTT.
    """
    segments: list[dict] = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            try:
                start_str, end_str = line.split("-->")
                start_str = start_str.strip().split(".")[0]  # drop millis
                end_str = end_str.strip().split(".")[0]
                start_sec = parse_time_to_seconds(start_str)
                end_sec = parse_time_to_seconds(end_str)
                text_lines: list[str] = []
                i += 1
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    raw = lines[i].strip()
                    # strip <v Speaker>...</v> wrapper
                    if raw.startswith("<v ") and ">" in raw:
                        raw = raw.split(">", 1)[1]
                    if raw.endswith("</v>"):
                        raw = raw[:-4]
                    text_lines.append(raw)
                    i += 1
                text = " ".join(text_lines).strip()
                if text:
                    segments.append({"start": float(start_sec), "end": float(end_sec), "text": text})
            except Exception:
                pass
        else:
            i += 1
    return segments


def recover_timestamps_from_vtt(
    speech_markdown: str,
    vtt_path: str,
    verbose: bool = False,
) -> str:
    """Recover ### **HH:MM:SS - HH:MM:SS** headers for timestamp-stripped markdown.

    Bilingual subcommands (en-zh, en-en, zh-zh, speaker) produce rewritten markdown
    with no timestamps (--- separated paragraphs). The VTT from stage 2 has timestamps
    + raw transcript text. Since group_into_topics preserves 100% of text (only groups
    adjacent segments) and rewrite_english keeps 97% wording, we can fuzzy-match each
    rewritten paragraph against consecutive VTT segment windows to recover timestamps.

    Algorithm:
      1. Parse VTT → segments[(start, end, text)]
      2. Split speech_markdown on '---' → paragraphs (in transcript order)
      3. For each paragraph, try windows of 1..max_window consecutive VTT segments
         starting from the current pointer; pick the window with highest
         rapidfuzz.fuzz.partial_ratio score.
      4. Assign that window's start..end as the paragraph timestamp range,
         advance the pointer past the matched window.
      5. Rebuild markdown with ### **HH:MM:SS - HH:MM:SS** headers.

    Args:
        speech_markdown: --- separated rewritten paragraphs (no timestamps)
        vtt_path: path to the VTT file from stage 2 ASR
        verbose: enable debug logging

    Returns:
        Markdown with ### **HH:MM:SS - HH:MM:SS** headers, ready for
        combine_speech_and_slides_by_timestamp().
    """
    import logging

    logger = logging.getLogger(__name__)

    # Check if speech_markdown already has timestamp headers — no recovery needed
    if "### **" in speech_markdown and " - " in speech_markdown:
        if verbose:
            logger.debug("recover_timestamps: speech_markdown already has timestamps, no recovery needed")
        return speech_markdown

    segments = _parse_vtt_to_segments(vtt_path)
    if not segments:
        if verbose:
            logger.warning("recover_timestamps: no segments parsed from VTT, returning unchanged")
        return speech_markdown

    # Split on '---' separators (the format write_rewritten_markdown uses)
    paragraphs = [p.strip() for p in speech_markdown.split("---") if p.strip()]
    if not paragraphs:
        if verbose:
            logger.warning("recover_timestamps: no paragraphs found in speech_markdown")
        return speech_markdown

    try:
        from rapidfuzz import fuzz
    except ImportError:
        # ponytail: difflib fallback if rapidfuzz not available
        from difflib import SequenceMatcher

        class _Fuzz:
            @staticmethod
            def ratio(a: str, b: str) -> float:
                return SequenceMatcher(None, a, b).ratio() * 100

        fuzz = _Fuzz()

    max_window = min(20, len(segments))  # cap window size
    seg_ptr = 0  # forward-only pointer into VTT segments

    result_parts: list[str] = []
    matched_count = 0

    for para_idx, paragraph in enumerate(paragraphs):
        # Strip bilingual labels like **[English]** / **[中文]** for matching
        match_text = paragraph
        for label in ("**[English]**", "**[中文]**", "**[EN]**", "**[ZH]**"):
            match_text = match_text.replace(label, "")
        match_text = match_text.strip()

        best_score = -1.0
        best_window_end = seg_ptr + 1  # default: single segment
        best_start = segments[seg_ptr]["start"] if seg_ptr < len(segments) else 0.0
        best_end = segments[seg_ptr]["end"] if seg_ptr < len(segments) else 0.0

        # Try windows of increasing size from current pointer.
        # Use fuzz.ratio (not partial_ratio) — ratio compares full strings and
        # penalizes length mismatch, so the correct-size window scores highest.
        # partial_ratio finds best substring → bigger windows always win.
        max_end = min(seg_ptr + max_window, len(segments))
        for window_end in range(seg_ptr + 1, max_end + 1):
            window_text = " ".join(s["text"] for s in segments[seg_ptr:window_end])
            score = fuzz.ratio(match_text, window_text)
            if score > best_score:
                best_score = score
                best_window_end = window_end
                best_start = segments[seg_ptr]["start"]
                best_end = segments[window_end - 1]["end"]

        # Assign timestamp range
        start_ts = _seconds_to_display(best_start)
        end_ts = _seconds_to_display(best_end)
        header = f"### **{start_ts} - {end_ts}**"
        result_parts.append(f"{header}\n\n{paragraph}")

        # Advance pointer past matched window
        seg_ptr = best_window_end
        matched_count += 1

        if verbose:
            logger.debug(
                "recover_timestamps: para %d/%d → %s - %s (score=%.1f, %d segments)",
                para_idx + 1,
                len(paragraphs),
                start_ts,
                end_ts,
                best_score,
                best_window_end - (best_window_end - max(0, seg_ptr - (best_window_end - seg_ptr))),
            )

    if verbose:
        logger.debug(
            "recover_timestamps: matched %d/%d paragraphs, %d/%d VTT segments consumed",
            matched_count,
            len(paragraphs),
            seg_ptr,
            len(segments),
        )

    return "\n\n".join(result_parts)


def combine_speech_and_slides_by_timestamp(speech_markdown: str, slides_markdown: str, verbose: bool = False) -> str:
    """
    Combine speech and slides markdown based on timestamp alignment.
    Preserves all timestamps exactly as they appear.
    
    Args:
        speech_markdown: Content from _rewritten.md with headers like "### **00:00:00 - 00:00:41**"
        slides_markdown: Content from _slides.md with headers like "### **00:00:00**"
        verbose: Enable verbose logging
    
    Returns:
        Combined markdown with slides inserted before matching speech sections
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("=== Starting Timestamp-Based Speech and Slides Combination ===")
    
    # Parse speech sections
    speech_sections = []
    lines = speech_markdown.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for speech section headers
        if line.startswith('### **') and line.endswith('**') and ' - ' in line:
            header = line
            # Extract start time from header
            time_range = line.replace('### **', '').replace('**', '')
            if ' - ' in time_range:
                start_time_str = time_range.split(' - ')[0]
                start_time_seconds = parse_time_to_seconds(start_time_str)
                
                # Collect content until next header or end
                content_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith('### **') and next_line.endswith('**') and ' - ' in next_line:
                        break
                    content_lines.append(lines[i])
                    i += 1
                
                content = '\n'.join(content_lines)
                
                speech_sections.append({
                    'header': header,
                    'start_time': start_time_seconds,
                    'content': content
                })
                continue
        
        i += 1
    
    # Parse slides sections
    slide_sections = []
    lines = slides_markdown.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for slide headers
        if line.startswith('### **') and line.endswith('**'):
            start_timestamp = line.replace('### **', '').replace('**', '')
            
            # Find next slide's timestamp to determine range
            next_timestamp = None
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith('### **') and next_line.endswith('**'):
                    next_timestamp = next_line.replace('### **', '').replace('**', '')
                    break
                j += 1
            
            # Collect content from this slide until next slide header
            content_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.startswith('### **') and next_line.endswith('**'):
                    break
                content_lines.append(lines[i])
                i += 1
            
            content = '\n'.join(content_lines)
            
            slide_sections.append({
                'start_timestamp': start_timestamp,
                'start_seconds': parse_time_to_seconds(start_timestamp),
                'end_timestamp': next_timestamp,
                'content': content
            })
            continue
        
        i += 1
    
    if verbose:
        logger.debug(f"Parsed {len(speech_sections)} speech sections and {len(slide_sections)} slide sections")
    
    # Build combined content
    combined_lines = []
    speech_idx = 0
    used_slides = set()
    
    # For each speech section, find and insert slides before it
    for speech_idx, speech_section in enumerate(speech_sections):
        speech_start_time = speech_section['start_time']
        speech_end_time = speech_sections[speech_idx + 1]['start_time'] if speech_idx + 1 < len(speech_sections) else float('inf')
        
        # Find slides that should be placed before this speech section
        slides_to_insert = []
        for slide_idx, slide in enumerate(slide_sections):
            if slide_idx in used_slides:
                continue
                
            slide_time = slide['start_seconds']
            
            # Insert slide if its timestamp falls within this speech section's time range
            # OR if it falls exactly at the boundary of this section
            if (slide_time >= speech_start_time and 
                (slide_time < speech_end_time or (speech_idx == len(speech_sections) - 1))):
                slides_to_insert.append(slide)
                used_slides.add(slide_idx)
        
        # Sort slides by timestamp
        slides_to_insert.sort(key=lambda x: x['start_seconds'])
        
        # Insert slides before this speech section
        for slide in slides_to_insert:
            if verbose:
                logger.debug(f"Inserting slide {slide['start_timestamp']} before speech section {speech_section['header']}")
            
            combined_lines.append(f"\n### **{slide['start_timestamp']}**\n")
            combined_lines.append(slide['content'])
            combined_lines.append("")  # Blank line separator
        
        # Insert speech section
        combined_lines.append(f"\n{speech_section['header']}\n")
        combined_lines.append(speech_section['content'])
        combined_lines.append("")  # Blank line separator
    
    # Add any remaining slides that weren't inserted (edge case: slides after last speech section)
    remaining_slides = [slide for i, slide in enumerate(slide_sections) if i not in used_slides]
    if remaining_slides:
        if verbose:
            logger.debug(f"Adding {len(remaining_slides)} remaining slides after all speech sections")
        
        for slide in remaining_slides:
            combined_lines.append(f"\n### **{slide['start_timestamp']}**\n")
            combined_lines.append(slide['content'])
            combined_lines.append("")
    
    result = '\n'.join(combined_lines).strip()
    
    if verbose:
        logger.debug("=== Timestamp-Based Combination Completed ===")
    
    return result