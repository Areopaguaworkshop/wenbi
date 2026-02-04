"""
PPT Method Implementation (Type 3)
Load/Convert PDF/PPT → Extract Timestamps → OCR → Combine with Audio
"""

import os
import logging
import subprocess
from typing import List, Tuple


def load_and_convert_pdf(ppt_pdf_path, output_dir, logger, verbose):
    """
    Load PPT or PDF file. Convert PPT to PDF if needed.
    Returns path to PDF file.
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
    
    else:
        print(f"Error: Unsupported file format: {file_ext} (expected .pdf or .pptx)")
        raise SystemExit(1)


def extract_timestamps_for_pdf_pages(deduplicated_frames):
    """
    Extract timestamps from deduplicated frames in order.
    Returns list of timestamps.
    """
    timestamps = [frame["timestamp"] for frame in deduplicated_frames]
    return timestamps


def validate_pdf_frame_mapping(pdf_path, timestamps, logger):
    """
    Validate that number of PDF pages matches number of timestamps.
    """
    try:
        import PyPDF2
        
        with open(pdf_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            num_pages = len(pdf.pages)
        
        num_timestamps = len(timestamps)
        
        if num_pages != num_timestamps:
            print(
                f"Error: PDF page count ({num_pages}) does not match "
                f"deduplicated frames count ({num_timestamps}). "
                f"Please check your inputs."
            )
            raise SystemExit(1)
        
        logger.debug(f"Mapping validated: {num_pages} pages ↔ {num_timestamps} timestamps")
    
    except ImportError:
        print("Error: PyPDF2 not installed. Run: rye add PyPDF2")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: Failed to validate PDF: {e}")
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
                      logger, verbose):
    """
    Execute PPT method workflow.
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
    
    # Step 1: Load and convert PDF/PPT
    pdf_path = load_and_convert_pdf(ppt_path, output_dir, logger, verbose)
    
    # Step 2: Extract timestamps
    timestamps = extract_timestamps_for_pdf_pages(deduplicated_frames)
    
    # Step 3: Validate 1-to-1 mapping
    validate_pdf_frame_mapping(pdf_path, timestamps, logger)
    
    # Step 4: OCR PDF pages
    if verbose:
        logger.debug("Step 2: Running OCR on PDF pages...")
    
    temp_img_dir = tempfile.mkdtemp(prefix="ppt_images_")
    
    try:
        markdown_sections = []
        
        for page_idx, timestamp in enumerate(timestamps):
            if verbose:
                logger.debug(f"Processing page {page_idx + 1}/{len(timestamps)}: {timestamp}")
            
            # Convert PDF page to image
            try:
                page_image = convert_pdf_page_to_image(pdf_path, page_idx)
                
                # Save temp image
                temp_img_path = os.path.join(temp_img_dir, f"pdf_page_{page_idx}.png")
                page_image.save(temp_img_path)
                
                section = f"\n### **{timestamp}**\n"
                
                if no_ocr:
                    # Embed as base64
                    b64 = image_to_base64(temp_img_path)
                    if b64:
                        section += f'<img src="data:image/png;base64,{b64}" />\n'
                else:
                    # OCR with marker
                    ocr_result = run_marker_pdf_on_image(
                        temp_img_path, output_dir, verbose, logger
                    )
                    
                    if ocr_result["success"]:
                        section += ocr_result["text"]
                        
                        # Add base64 images if any
                        for filename, b64 in ocr_result["base64_images"].items():
                            section += f'\n<img src="data:image/png;base64,{b64}" />\n'
                    else:
                        # OCR failed, fallback to base64
                        if verbose:
                            logger.warning(f"OCR failed for page {page_idx + 1}, using base64")
                        
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
                print(f"Error: Failed to process page {page_idx + 1}: {e}")
                raise SystemExit(1)
        
        ppt_md = os.path.join(output_dir, f"{base_name}_ppt.md")
        with open(ppt_md, "w", encoding="utf-8") as f:
            f.write("".join(markdown_sections))
        
        if verbose:
            logger.debug(f"PDF OCR completed: {ppt_md}")
    
    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_img_dir)
        except:
            pass
    
    # Step 5: Rewrite audio
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
    
    # Step 6: Combine
    if verbose:
        logger.debug("Step 4: Combining PPT and audio markdown...")
    
    with open(ppt_md, "r", encoding="utf-8") as f:
        ppt_content = f.read()
    
    combined_markdown = combine_speech_and_slides(
        speech_markdown=audio_markdown,
        slides_markdown=ppt_content,
        logger=logger,
        verbose=verbose
    )
    
    combine_md = os.path.join(output_dir, f"{base_name}_combine.md")
    with open(combine_md, "w", encoding="utf-8") as f:
        f.write(combined_markdown)
    
    if verbose:
        logger.debug(f"Combined markdown: {combine_md}")
    
    # Step 7: Clean (if not --no-clean)
    if no_clean:
        combine_clean_md = None
        if verbose:
            logger.debug("--no-clean: Skipping clean phase")
    else:
        combine_clean_md = clean_combined_markdown(
            combine_md, output_dir, base_name, logger, verbose
        )
    
    return combine_md, combine_clean_md
