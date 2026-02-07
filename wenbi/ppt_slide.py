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


def execute_ppt_method(video_path, deduplicated_frames, ppt_path, output_dir,
                       no_ocr, no_clean, base_name, cite_timestamps, llm,
                       chunk_length, max_tokens, timeout, temperature, lang,
                       transcribe_model, multi_language, transcribe_lang,
                       logger, verbose, ssim_threshold=0.8):
    """
    Execute PPT method workflow with OpenCV-based PDF-to-Frame matching.
    Returns: (combine_md_path, combine_clean_md_path)
    """
    from wenbi.cli import (
        run_marker_pdf_on_image, image_to_base64, clean_combined_markdown
    )
    from wenbi.main import process_input
    from wenbi.model import combine_speech_and_slides
    import tempfile
    import os
    
    if verbose:
        logger.debug("=== TYPE 3: PPT Method ===")
        logger.debug(f"Video: {video_path}")
        logger.debug(f"PPT/PDF: {ppt_path}")
        logger.debug(f"Output dir: {output_dir}")
        logger.debug(f"Deduplicated frames count: {len(deduplicated_frames)}")
    
    # Check if input is an image file
    file_ext = os.path.splitext(ppt_path)[1].lower()
    
    if file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        # Process as image input
        if verbose:
            logger.debug("Processing single image file as slide")
        
        markdown_sections = process_images_as_slides(
            ppt_path, deduplicated_frames, output_dir, no_ocr, base_name, cite_timestamps, logger, verbose
        )
        
        ppt_md = os.path.join(output_dir, f"{base_name}_ppt.md")
        with open(ppt_md, "w", encoding="utf-8") as f:
            f.write("".join(markdown_sections))
        
        logger.debug(f"Image processing completed: {ppt_md}")
    else:
        # Process as PDF/PPT input
        logger.debug("Step 1: Loading and converting PDF/PPT file...")
        pdf_path = load_and_convert_pdf(ppt_path, output_dir, logger, verbose)
        logger.debug(f"Loaded PDF: {pdf_path}")
        
        # Step 4: Use OpenCV SSIM to match PDF pages to frames
        logger.debug("Step 2: Matching PDF pages to video frames using OpenCV SSIM...")
        matched_pairs = match_pdf_pages_to_frames(
            pdf_path, 
            deduplicated_frames, 
            logger, 
            verbose, 
            ssim_threshold=ssim_threshold
        )
        
        if not matched_pairs:
            print("Error: No PDF pages matched with video frames. Check your inputs.")
            raise SystemExit(1)
        
        logger.debug(f"Successfully matched {len(matched_pairs)} PDF pages to frames")
        
        # Step 5: OCR matched PDF pages with their timestamps
        logger.debug("Step 3: Running OCR on matched PDF pages...")
        
        temp_img_dir = tempfile.mkdtemp(prefix="ppt_images_")
        
        try:
            markdown_sections = []
            
            for pair_idx, (pdf_page_idx, frame_data) in enumerate(matched_pairs):
                timestamp = frame_data["timestamp"]
                
                if verbose:
                    logger.debug(f"Processing matched pair {pair_idx + 1}/{len(matched_pairs)}: PDF page {pdf_page_idx + 1} ↔ {timestamp}")
                
                try:
                    # Convert PDF page to image
                    page_image = convert_pdf_page_to_image(pdf_path, pdf_page_idx)
                    
                    # Save temp image
                    temp_img_path = os.path.join(temp_img_dir, f"pdf_page_{pdf_page_idx}.png")
                    page_image.save(temp_img_path)
                    
                    section = f"\n### **{timestamp}**\n"
                    
                    if no_ocr:
                        # Embed as base64
                        logger.debug(f"  Embedding page {pdf_page_idx + 1} as base64 (--no-ocr)")
                        b64 = image_to_base64(temp_img_path)
                        if b64:
                            section += f'<img src="data:image/png;base64,{b64}" />\n'
                    else:
                        # OCR with marker
                        logger.debug(f"  Running marker OCR on page {pdf_page_idx + 1}...")
                        ocr_result = run_marker_pdf_on_image(
                            temp_img_path, output_dir, verbose, logger
                        )
                        
                        if ocr_result["success"]:
                            logger.debug(f"  OCR successful for page {pdf_page_idx + 1}")
                            section += ocr_result["text"]
                            
                            # Add base64 images if any
                            for filename, b64 in ocr_result["base64_images"].items():
                                section += f'\n<img src="data:image/png;base64,{b64}" />\n'
                        else:
                            # OCR failed, fallback to base64
                            logger.warning(f"  OCR failed for page {pdf_page_idx + 1}, using base64")
                            
                            b64 = image_to_base64(temp_img_path)
                            if b64:
                                section += f'<img src="data:image/png;base64,{b64}" />\n'
                    
                    markdown_sections.append(section)
                    
                    # Clean temp image
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
        
        finally:
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(temp_img_dir)
            except:
                pass
    
    # Step 6: Process audio
    logger.debug("Step 4: Processing audio from video...")
    
    # Determine if input is URL or file path
    is_url = video_path.startswith(("http://", "https://", "www."))
    
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
    
    logger.debug("Calling process_input for audio processing...")
    result = process_input(
        file_path=video_path if not is_url else None,
        url=video_path if is_url else "",
        **params
    )
    
    audio_markdown = result[0]
    logger.debug("Audio processing completed")
    
    # Step 7: Combine
    logger.debug("Step 5: Combining PDF and audio markdown...")
    
    with open(ppt_md, "r", encoding="utf-8") as f:
        ppt_content = f.read()
    
    combined_markdown = combine_speech_and_slides(
        speech_markdown=audio_markdown,
        slides_markdown=ppt_content,
        cite_timestamps=cite_timestamps,
        verbose=verbose
    )
    
    combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
    with open(combine_md, "w", encoding="utf-8") as f:
        f.write(combined_markdown)
    
    logger.debug(f"Combined markdown created: {combine_md} (timestamps preserved)")
    
    # Step 8: Clean (if not --no-clean)
    if no_clean:
        combine_clean_md = None
        logger.debug("--no-clean flag set: Skipping clean phase")
    else:
        logger.debug("Step 6: Cleaning combined markdown...")
        combine_clean_md = clean_combined_markdown(
            combine_md, output_dir, base_name, logger, verbose
        )
        logger.debug(f"Cleaned markdown created: {combine_clean_md}")
    
    logger.debug("=== PPT Method Complete ===")
    return combine_md, combine_clean_md