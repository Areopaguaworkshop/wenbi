import logging
import os

try:
    import litellm
except ImportError:
    litellm = None
from wenbi.utils import segment


DEFAULT_LLM = "ollama/glm-5.2:cloud"
DEFAULT_OPENAI_LLM = "openai/gpt-5.6-terra"


def resolve_translation_engine(engine: str = "auto", llm: str = "") -> tuple[str, bool]:
    """Return the LLM and whether translations should use DeepL first."""
    engine = (engine or "auto").lower()
    if engine not in {"auto", "deepl", "ollama", "openai"}:
        raise ValueError(f"Unknown translation engine: {engine}")
    if engine == "openai":
        return llm or DEFAULT_OPENAI_LLM, False
    if engine == "ollama":
        return llm or DEFAULT_LLM, False
    return llm or DEFAULT_LLM, True


def _import_dspy():
    """Import dspy lazily so DeepL-first flows don't fail at module import time."""
    try:
        if "DSPY_CACHEDIR" not in os.environ:
            cache_dir = os.path.join(os.getcwd(), ".dspy_cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["DSPY_CACHEDIR"] = cache_dir
        import dspy

        return dspy
    except Exception as e:
        raise RuntimeError(f"Failed to import dspy: {e}") from e


def configure_lm(model_string, verbose=False, **kwargs):
    """Configure the Language Model with verbose logging support"""
    logger = logging.getLogger(__name__)
    dspy = _import_dspy()

    if not model_string:
        model_string = DEFAULT_LLM

    if verbose:
        logger.debug(f"Configuring LLM: {model_string}")

    parts = model_string.strip().split("/")
    provider = parts[0].lower() if parts else ""

    if verbose:
        logger.debug(f"LLM provider: {provider}")

    config = kwargs
    if provider == "ollama":
        # Check if Ollama is running before trying to configure
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama returned status code {response.status_code}"
                )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at http://localhost:11434. "
                f"Please ensure Ollama is running. Error: {e}"
            )

        config.update(
            {
                "base_url": "http://localhost:11434",
                "model": model_string,
            }
        )
        if verbose:
            logger.debug(f"Ollama configuration: {config}")
        lm = dspy.LM(**config)
    elif provider == "openai":
        from wenbi.llm.openai import get_openai_lm

        if verbose:
            logger.debug(f"OpenAI configuration: model={model_string}")
        lm = get_openai_lm(model_name=model_string, **config)
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_JSON")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GOOGLE_API_KEY_JSON environment variable not set."
            )

        # Extract the actual model name (e.g., "gemini-2.5-flash" from "gemini/gemini-2.5-flash")
        model_name = (
            model_string.split("/", 1)[1] if "/" in model_string else model_string
        )

        # Use the correct format for LiteLLM Gemini integration
        config.update(
            {
                "model": f"gemini/{model_name}",
                "api_key": api_key,
            }
        )
        if verbose:
            logger.debug(f"Gemini configuration: model={config['model']}")
        lm = dspy.LM(**config)
    else:
        config.update({"model": model_string})
        if verbose:
            logger.debug(f"Generic LM configuration: {config}")
        lm = dspy.LM(**config)

    dspy.configure(lm=lm)

    if verbose:
        logger.debug("LLM configuration completed successfully")

    return lm


def translate(
    input_file,
    output_dir="",
    translate_language="Chinese",
    llm="ollama/glm-5.2:cloud",
    chunk_length=20,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    cite_timestamps=False,
    verbose=False,
    use_deepl=True,
    deepl_key=None,
    keep_original_lang=False,
    use_glossary=True,
    glossary_file=None,
):
    """
    Translate text content using DeepL (primary) with LLM fallback.

    Args:
        use_deepl: Always True - DeepL API is always attempted first
        deepl_key: Optional DeepL API key (uses DEEPL_API_KEY env var if not provided)
        keep_original_lang: If True, output both original and translated text side-by-side
        use_glossary: If True (default), apply EN→ZH glossary for term consistency.
            No-op when target language is not Chinese.
        glossary_file: Optional path to a user glossary JSON ({english: chinese}).
            Overrides the built-in patristic glossary.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Translation Process ===")
        logger.debug(f"Input file: {input_file}")
        logger.debug(f"Target language: {translate_language}")
        logger.debug(f"LLM model: {llm}")
        logger.debug(f"Use DeepL: {use_deepl}")
        logger.debug(f"Chunk length: {chunk_length}")
        logger.debug(f"Max tokens: {max_tokens}")
        logger.debug(f"Include timestamps: {cite_timestamps}")
        logger.debug(f"Keep original language: {keep_original_lang}")

    # Try to initialize DeepL if requested
    deepl_translator = None
    deepl_available = False
    if use_deepl:
        try:
            from wenbi.llm.deepl import configure_deepl, is_deepl_available

            if is_deepl_available(deepl_key, verbose=verbose):
                deepl_translator = configure_deepl(deepl_key, verbose=verbose)
                deepl_available = True
                if verbose:
                    logger.debug("DeepL translator initialized successfully")
            else:
                if verbose:
                    logger.debug("DeepL not available, will use LLM only")
        except Exception as e:
            if verbose:
                logger.debug(f"Failed to initialize DeepL: {e}")
            deepl_available = False

    # Glossary text for the LLM fallback path (only for Chinese target).
    _is_chinese_target = (translate_language or "").lower() in ("chinese", "zh")
    glossary_text = ""
    if use_glossary and _is_chinese_target:
        try:
            if glossary_file:
                import json

                with open(glossary_file, encoding="utf-8") as f:
                    pairs = json.load(f)
                glossary_text = "\n".join(f"{en}: {zh}" for en, zh in pairs.items())
            else:
                from wenbi.patristic_glossary import get_glossary_for_dspy

                glossary_text = get_glossary_for_dspy()
        except Exception as e:
            if verbose:
                logger.debug(f"Glossary load failed: {e}")
            glossary_text = ""

    # Configure LLM lazily: only initialize if DeepL fails for a chunk.
    translate_module = None
    llm_init_error = None

    def get_translate_module():
        nonlocal translate_module, llm_init_error
        if translate_module is not None:
            return translate_module
        if llm_init_error is not None:
            raise llm_init_error

        try:
            lm = configure_lm(
                llm,
                verbose=verbose,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
            )
            dspy = _import_dspy()
        except Exception as e:
            llm_init_error = e
            raise

        class TranslateSignature(dspy.Signature):
            """Translate the given text to the target language while preserving the original meaning and style."""

            text_to_translate = dspy.InputField(desc="Text content to be translated")
            target_language = dspy.InputField(desc="Target language for translation")
            glossary = dspy.InputField(desc="Optional term glossary; honor it for consistency", required=False)
            translated_text = dspy.OutputField(
                desc="Translated text in the target language"
            )

        translate_module = dspy.Predict(TranslateSignature)
        if verbose:
            logger.debug("LLM fallback module initialized")
        return translate_module

    if verbose:
        logger.debug(
            "Translation modules initialized (DeepL primary, LLM lazy fallback)"
        )

    # Read and segment the input text
    segmented_text = segment(input_file, chunk_length, cite_timestamps, verbose=verbose)

    # Split into chunks for processing
    chunks = segmented_text.split("\n\n")
    if verbose:
        logger.debug(f"Text divided into {len(chunks)} chunks for processing")

    translated_chunks = []
    deepl_count = 0
    llm_count = 0

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue

        if verbose:
            logger.debug(
                f"Translating chunk {i}/{len(chunks)} ({len(chunk)} characters)"
            )

        try:
            # Skip timestamp headers when cite_timestamps is True
            is_timestamp_header = chunk.strip().startswith(
                "### **"
            ) and chunk.strip().endswith("**")

            if cite_timestamps and is_timestamp_header:
                if verbose:
                    logger.debug(
                        f"Preserving timestamp header: {chunk.strip()[:50]}..."
                    )
                translated_chunks.append(chunk)
            else:
                translated_text = None

                # Try DeepL first if available
                if deepl_available:
                    try:
                        from wenbi.llm.deepl import translate_with_deepl

                        translated_text = translate_with_deepl(
                            deepl_translator, chunk, translate_language, verbose=verbose,
                            use_glossary=use_glossary, glossary_file=glossary_file,
                        )
                        deepl_count += 1
                        if verbose:
                            logger.debug(f"Chunk {i} translated with DeepL")
                    except Exception as e:
                        if verbose:
                            logger.debug(
                                f"DeepL failed for chunk {i}: {e}. Falling back to LLM."
                            )
                        translated_text = None

                # Fallback to LLM if DeepL failed or not available
                if translated_text is None:
                    try:
                        translate_module = get_translate_module()
                        llm_kwargs = {
                            "text_to_translate": chunk,
                            "target_language": translate_language,
                        }
                        if glossary_text:
                            llm_kwargs["glossary"] = glossary_text
                        result = translate_module(**llm_kwargs)
                        translated_text = result.translated_text
                        llm_count += 1
                        if verbose:
                            logger.debug(f"Chunk {i} translated with LLM (fallback)")
                    except Exception as e:
                        error_msg = f"Error translating chunk {i}: No translation service available (DeepL failed, LLM unavailable: {e})"
                        if verbose:
                            logger.debug(error_msg)
                        print(error_msg)
                        translated_text = f"[Translation Error: {chunk}]"

                # Format output based on keep_original_lang flag
                if keep_original_lang:
                    chunk_output = f"**[Original]**\n{chunk}\n\n**[{translate_language}]**\n{translated_text}"
                    translated_chunks.append(chunk_output)
                else:
                    translated_chunks.append(translated_text)

        except Exception as e:
            error_msg = f"Error translating chunk {i}: {e}"
            if verbose:
                logger.debug(error_msg)
            print(error_msg)
            translated_chunks.append(f"[Translation Error: {chunk}]")

    final_translation = "\n\n".join(translated_chunks)

    if verbose:
        logger.debug(
            f"Translation completed. Final text length: {len(final_translation)} characters"
        )
        logger.debug(f"DeepL chunks: {deepl_count}, LLM chunks: {llm_count}")
        logger.debug("=== Translation Process Completed ===")

    return final_translation


def rewrite(
    input_file,
    output_dir="",
    llm="ollama/glm-5.2:cloud",
    rewrite_language="Chinese",
    chunk_length=20,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    cite_timestamps=False,
    verbose=False,
    style=None,
):
    """
    Rewrite oral language to written form using LLM with verbose logging support.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Rewrite Process ===")
        logger.debug(f"Input file: {input_file}")
        logger.debug(f"Target language: {rewrite_language}")
        logger.debug(f"LLM model: {llm}")
        logger.debug(f"Chunk length: {chunk_length}")
        logger.debug(f"Max tokens: {max_tokens}")
        logger.debug(f"Include timestamps: {cite_timestamps}")
        if style:
            logger.debug(f"Style: {style}")

    # Configure LLM
    lm = configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )

    dspy = _import_dspy()

    # Select signature based on style
    if style == "zh-speaker":
        class SpeakerRewriteSignature(dspy.Signature):
            """
            Rewrite this multi-speaker transcript into formal written prose in {target_language}, suitable for publication. Preserve speaker distinctions. Follow these rules strictly:
            1. Keep speaker labels — note who said what. Use 【主持人】for the host/moderator and 【嘉宾】for the guest, or maintain the original Speaker0/Speaker1 labels if speaker roles are unclear.
            2. Remove oral artifacts per speaker: filler words (嗯, 啊, 对吧, 是吧 and equivalents), vague completions (什么的/之类), self-corrections, half-sentences, and meta-commentary about the conversation.
            3. Rewrite each speaker's conversational patterns into formal written prose, but maintain the turn-taking structure so readers can follow who said what.
            4. Correct grammar, punctuation, and usage errors within each speaker's turn.
            5. IMPORTANT: Preserve the original meaning and scholarly content faithfully. Do not add ideas that were not stated or alter any speaker's intended arguments.
            6. The resulting text should read like a polished interview transcript or dialogue-based academic text.
            """

            speaker_text = dspy.InputField(desc="Multi-speaker transcript with speaker labels (e.g., 【Speaker0】, 【Speaker1】)")
            target_language = dspy.InputField(desc="Target language for the rewriting")
            written_text = dspy.OutputField(desc="Formal written version preserving speaker attribution and turn structure")

        rewrite_module = dspy.Predict(SpeakerRewriteSignature)
    else:
        class RewriteSignature(dspy.Signature):
            """
            Rewrite this oral/spoken text into formal written prose in {target_language}, suitable for publication as an academic transcript. Follow these rules strictly:
            1. Remove all oral artifacts: filler words, rhetorical confirmations (对吧/是吧 and equivalents), vague completions (什么的/之类的 and equivalents), self-corrections, half-sentences, and meta-commentary about the conversation itself.
            2. Rewrite conversational patterns into formal academic prose. Restructure fragmented speech, run-on sentences, and repetitions into clear, concise written sentences.
            3. Correct any grammar, punctuation, or usage errors.
            4. IMPORTANT: Preserve the original meaning and scholarly content faithfully. Do not add ideas that were not stated or alter the speaker's intended arguments.
            5. The resulting text may be 85-100% of the original length, since removing oral artifacts naturally shortens the text.
            """

            oral_text = dspy.InputField(desc="Oral or spoken text to be rewritten")
            target_language = dspy.InputField(desc="Target language for the rewriting")
            written_text = dspy.OutputField(desc="Formal written version suitable for publication")

        rewrite_module = dspy.Predict(RewriteSignature)

    if verbose:
        logger.debug(f"LLM rewrite module initialized (style={style or 'default'})")

    # Read and segment the input text
    segmented_text = segment(input_file, chunk_length, cite_timestamps, verbose=verbose, style=style)

    # Split into chunks for processing
    chunks = segmented_text.split("\n\n")
    if verbose:
        logger.debug(f"Text divided into {len(chunks)} chunks for processing")

    rewritten_chunks = []

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            rewritten_chunks.append(chunk)
            continue

        if verbose:
            logger.debug(f"Rewriting chunk {i}/{len(chunks)} ({len(chunk)} characters)")

        try:
            # Check if chunk contains a timestamp header when cite_timestamps is True
            timestamp_header = None
            content_to_rewrite = chunk

            if cite_timestamps:
                # Extract timestamp header if present (format: ### **timestamp**)
                lines = chunk.split("\n", 1)
                first_line = lines[0].strip()

                if first_line.startswith("### **") and first_line.endswith("**"):
                    timestamp_header = first_line
                    # Content is everything after the first line
                    content_to_rewrite = lines[1] if len(lines) > 1 else ""

                    if verbose:
                        logger.debug(f"Extracted timestamp header: {timestamp_header}")

            # Skip rewriting if only timestamp header exists with no content
            if not content_to_rewrite.strip():
                rewritten_chunks.append(chunk)
                if verbose:
                    logger.debug(
                        f"Chunk {i} contains only timestamp header, skipping rewrite"
                    )
                continue

            # Rewrite the content
            if style == "zh-speaker":
                result = rewrite_module(
                    speaker_text=content_to_rewrite, target_language=rewrite_language
                )
            else:
                result = rewrite_module(
                    oral_text=content_to_rewrite, target_language=rewrite_language
                )
            rewritten_content = result.written_text

            # Reconstruct chunk with timestamp header if it was extracted
            if cite_timestamps and timestamp_header:
                rewritten_chunk = f"{timestamp_header}\n\n{rewritten_content}"
            else:
                rewritten_chunk = rewritten_content

            rewritten_chunks.append(rewritten_chunk)

            if verbose:
                logger.debug(f"Chunk {i} rewritten successfully")

        except Exception as e:
            error_msg = f"Error rewriting chunk {i}: {e}"
            if verbose:
                logger.debug(error_msg)
            print(error_msg)
            rewritten_chunks.append(f"[Rewrite Error: {chunk}]")

    final_rewrite = "\n\n".join(rewritten_chunks)

    if verbose:
        logger.debug(
            f"Rewrite completed. Final text length: {len(final_rewrite)} characters"
        )
        logger.debug("=== Rewrite Process Completed ===")

    # Write output file if output_dir is provided
    output_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_rewritten.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_rewrite)

        if verbose:
            logger.debug(f"Rewrite output saved to: {output_file}")

    return final_rewrite, output_file


def academic(
    input_file,
    output_dir="",
    llm="ollama/glm-5.2:cloud",
    academic_lang="English",
    chunk_length=20,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    cite_timestamps=False,
    verbose=False,
):
    """
    Convert text to academic writing style using LLM with verbose logging support.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Academic Writing Process ===")
        logger.debug(f"Input file: {input_file}")
        logger.debug(f"Academic language: {academic_lang}")
        logger.debug(f"LLM model: {llm}")
        logger.debug(f"Chunk length: {chunk_length}")
        logger.debug(f"Max tokens: {max_tokens}")
        logger.debug(f"Include timestamps: {cite_timestamps}")

    # Configure LLM
    lm = configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )

    dspy = _import_dspy()

    class AcademicSignature(dspy.Signature):
        """
        Rewrite this text into formal academic style in {academic_lang}, suitable for publication. Follow these rules strictly:
        1. Remove all oral artifacts: filler words, rhetorical confirmations (对吧/是吧 and equivalents), vague completions (什么的/之类的 and equivalents), self-corrections, half-sentences, and meta-commentary about the conversation itself.
        2. Rewrite conversational patterns into formal academic prose. Restructure fragmented speech, run-on sentences, and repetitions into clear, concise written sentences.
        3. Correct any grammar, punctuation, or usage errors.
        4. IMPORTANT: Preserve the original meaning and scholarly content faithfully. Do not add ideas that were not stated or alter the speaker's intended arguments.
        5. The resulting text may be 85-100% of the original length, since removing oral artifacts naturally shortens the text.
        6. IMPORTANT: Preserve ALL footnote references (e.g., [^1], [^2]) exactly as they appear.
        """

        input_text = dspy.InputField(
            desc="Original text to be transformed into academic style"
        )
        target_language = dspy.InputField(desc="Target language for academic writing")
        academic_text = dspy.OutputField(desc="Formal academic text suitable for publication")

    academic_module = dspy.Predict(AcademicSignature)

    if verbose:
        logger.debug("LLM academic writing module initialized")

    # Handle different input file types
    if input_file.lower().endswith(".docx"):
        if verbose:
            logger.debug("Processing DOCX file")
        content = process_docx(input_file, verbose=verbose)
        segmented_text = content
    else:
        # Read and segment the input text
        segmented_text = segment(
            input_file, chunk_length, cite_timestamps, verbose=verbose
        )

    # Split into chunks for processing
    chunks = segmented_text.split("\n\n")
    if verbose:
        logger.debug(f"Text divided into {len(chunks)} chunks for processing")

    academic_chunks = []

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            academic_chunks.append(chunk)
            continue

        if verbose:
            logger.debug(f"Converting chunk {i}/{len(chunks)} to academic style ({len(chunk)} characters)")

        try:
            # Skip timestamp headers when cite_timestamps is True
            is_timestamp_header = chunk.strip().startswith(
                "### **"
            ) and chunk.strip().endswith("**")

            if cite_timestamps and is_timestamp_header:
                if verbose:
                    logger.debug(
                        f"Preserving timestamp header: {chunk.strip()[:50]}..."
                    )
                academic_chunks.append(chunk)
            else:
                result = academic_module(
                    input_text=chunk, target_language=academic_lang
                )
                academic_chunks.append(result.academic_text)

                if verbose:
                    logger.debug(f"Chunk {i} converted to academic style successfully")

        except Exception as e:
            error_msg = f"Error converting chunk {i} to academic style: {e}"
            if verbose:
                logger.debug(error_msg)
            print(error_msg)
            academic_chunks.append(f"[Academic Conversion Error: {chunk}]")

    final_academic = "\n\n".join(academic_chunks)

    if verbose:
        logger.debug(
            f"Academic conversion completed. Final text length: {
                len(final_academic)
            } characters"
        )
        logger.debug("=== Academic Writing Process Completed ===")

    return final_academic


def process_docx(input_file, verbose=False):
    """
    Process DOCX files with verbose logging support.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug(f"Processing DOCX file: {input_file}")

    try:
        from docx import Document

        doc = Document(input_file)
        content = []

        paragraph_count = len(doc.paragraphs)
        if verbose:
            logger.debug(f"Found {paragraph_count} paragraphs in DOCX")

        for i, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
                if verbose and i % 50 == 0:  # Log progress every 50 paragraphs
                    logger.debug(f"Processed paragraph {i}/{paragraph_count}")

        result = "\n\n".join(content)

        if verbose:
            logger.debug(
                f"DOCX processing completed. Extracted {len(content)} paragraphs, {
                    len(result)
                } characters total"
            )

        return result

    except ImportError:
        error_msg = "python-docx library not installed. Please install it with: pip install python-docx"
        if verbose:
            logger.debug(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Error processing DOCX file: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def read_markdown_file(file_path, verbose=False):
    """
    Read markdown file content with verbose logging support.

    Returns: (markdown_content, file_path)
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug(f"Reading markdown file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if verbose:
            logger.debug(
                f"Markdown file read successfully. Content length: {len(content)} characters"
            )

        return content, file_path

    except Exception as e:
        error_msg = f"Error reading markdown file: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def convert_slides_to_markdown(
    slides_file,
    output_dir="",
    image_export_mode="embedded",
    verbose=False,
):
    """
    Convert PPTX/PDF to markdown using marker-pdf Python API with verbose logging support.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Slides to Markdown Conversion ===")
        logger.debug(f"Slides file: {slides_file}")
        logger.debug(f"Image export mode: {image_export_mode}")

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        if verbose:
            logger.debug("Loading marker-pdf models...")

        # Create model dict for conversion
        artifact_dict = create_model_dict()

        if verbose:
            logger.debug("Initializing PDF converter...")

        # Convert slides to markdown
        converter = PdfConverter(
            artifact_dict=artifact_dict,
        )

        markdown_output = converter(slides_file)
        markdown_text = markdown_output.markdown

        if verbose:
            logger.debug(
                f"Conversion completed. Markdown length: {len(markdown_text)} characters"
            )

        # Handle image export mode
        if image_export_mode == "none":
            # Remove image references from markdown
            import re

            markdown_text = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_text)
            if verbose:
                logger.debug("Stripped image references from markdown")

        # Save to file if output_dir provided
        output_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(slides_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_slides.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            if verbose:
                logger.debug(f"Slides markdown saved to: {output_file}")

        return markdown_text, output_file

    except ImportError as e:
        error_msg = f"marker-pdf library not installed. Please install it with: pip install marker-pdf"
        if verbose:
            logger.debug(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Error converting slides to markdown: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def convert_single_slide_image(
    image_path,
    langs=["Chinese", "English"],
    output_dir=None,
    verbose=False,
):
    """
    OCR a single image using marker-pdf functionality
    Returns dictionary with OCR text and metadata
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Single Image OCR ===")
        logger.debug(f"Image file: {image_path}")

    try:
        import tempfile

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from PIL import Image

        if verbose:
            logger.debug("Loading marker-pdf models...")

        # Create model dict for conversion
        artifact_dict = create_model_dict()

        if verbose:
            logger.debug("Processing single image...")

        # Use output_dir if provided, otherwise use temp directory
        if output_dir and os.path.exists(output_dir):
            work_dir = output_dir
        else:
            work_dir = tempfile.mkdtemp()

        temp_pdf = os.path.join(work_dir, "temp_ocr.pdf")

        # Convert image to PDF
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save(temp_pdf, "PDF", resolution=150.0)

        # Convert PDF to markdown
        converter = PdfConverter(
            artifact_dict=artifact_dict,
        )

        markdown_output = converter(temp_pdf)

        if verbose:
            logger.debug(
                f"OCR completed. Text length: {len(markdown_output.markdown)} characters"
            )
            logger.debug(f"Images output to: {work_dir}")

        # Cleanup temp PDF only if we created a temp directory
        import shutil

        if not (output_dir and os.path.exists(output_dir)):
            shutil.rmtree(work_dir)
        else:
            # Cleanup just the temp PDF if using output_dir
            try:
                os.remove(temp_pdf)
            except:
                pass

        # Extract confidence and metadata (marker doesn't provide this for images directly)
        result = {
            "text": markdown_output.markdown.strip(),
            "confidence": 0.8,  # Default confidence for image OCR
            "page_id": 1,
            "metadata": {"source": image_path, "languages": langs},
        }

        return result

    except ImportError as e:
        error_msg = f"marker-pdf library not installed. Please install it with: pip install marker-pdf"
        if verbose:
            logger.debug(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Error OCR processing image: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Error during alignment: {e}"
        if verbose:
            logger.debug(error_msg)
        logger.debug("=== Slide-Speech Alignment Completed ===")
        return {
            "aligned_section": "NO_MATCH",
            "confidence": "none",
        }


def combine_speech_and_slides_enhanced(
    speech_markdown,
    slides_markdown,
    llm="ollama/glm-5.2:cloud",
    output_dir="",
    cite_timestamps=False,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    verbose=False,
):
    """
    Enhanced version of combine_speech_and_slides using multi-layered similarity analysis
    instead of LLM-based alignment. Implements keyword extraction, semantic similarity,
    temporal constraints, and content preservation guarantees.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Enhanced Speech and Slides Combination ===")
        logger.debug(f"Speech content length: {len(speech_markdown)} characters")
        logger.debug(f"Slides content length: {len(slides_markdown)} characters")

    # Import enhanced functions
    from wenbi.enhanced_combination import _extract_slides as enhanced_extract_slides
    from wenbi.enhanced_combination import (
        build_enhanced_combined_markdown,
        calculate_combined_similarity,
        create_similarity_matrix,
        distribute_unaligned_slides,
        find_optimal_alignment,
    )

    # Step 1: Extract slides from markdown
    slides = enhanced_extract_slides(slides_markdown, verbose)

    if verbose:
        logger.debug(f"Extracted {len(slides)} slides from presentation")

    # Step 2: Split speech into paragraphs (reuse existing logic)
    strategies = [
        lambda x: [p.strip() for p in x.split("\n\n") if p.strip()],  # Double newlines
        lambda x: [p.strip() for p in x.split("\n") if p.strip()],  # Single newlines
        lambda x: [
            p.strip() for p in x.replace("\n\n", "\n").split("\n") if p.strip()
        ],  # Normalize then split
    ]

    best_paragraphs = []
    best_content_ratio = 0

    for strategy in strategies:
        test_paragraphs = strategy(speech_markdown)
        content_ratio = (
            sum(len(p) for p in test_paragraphs) / len(speech_markdown)
            if speech_markdown
            else 0
        )

        if content_ratio > best_content_ratio and len(test_paragraphs) > 0:
            best_content_ratio = content_ratio
            best_paragraphs = test_paragraphs

    speech_paragraphs = best_paragraphs

    # Fallback: if all strategies fail, use entire content as one paragraph
    if not speech_paragraphs and speech_markdown.strip():
        speech_paragraphs = [speech_markdown.strip()]
        if verbose:
            logger.debug("Used fallback: entire speech as single paragraph")

    if verbose:
        logger.debug(f"Speech split into {len(speech_paragraphs)} paragraphs")
        preserved_ratio = (
            sum(len(p) for p in speech_paragraphs) / len(speech_markdown)
            if speech_markdown
            else 0
        )
        logger.debug(f"Content preservation ratio: {preserved_ratio:.2%}")

    # Step 3: Create similarity matrix for all slide-speech pairs
    similarity_matrix = create_similarity_matrix(slides, speech_paragraphs, verbose)

    # Step 4: Find optimal alignment with hybrid temporal constraints
    aligned_slides = find_optimal_alignment(
        similarity_matrix, len(slides), len(speech_paragraphs), verbose
    )

    # Step 5: Handle unaligned slides with even distribution
    aligned_slides = distribute_unaligned_slides(
        slides, speech_paragraphs, aligned_slides, verbose
    )

    # Step 6: Build enhanced combined markdown with content preservation
    combined_content = build_enhanced_combined_markdown(
        speech_paragraphs, slides, aligned_slides, cite_timestamps, verbose
    )

    # Step 7: Final content preservation check
    if verbose:
        original_speech_length = len(speech_markdown)
        final_combined_length = len(combined_content)
        preservation_ratio = (
            final_combined_length / original_speech_length
            if original_speech_length
            else 0
        )
        logger.debug(f"Final content preservation ratio: {preservation_ratio:.2%}")

        # Count aligned vs unaligned slides
        aligned_count = len(
            [slide for slide, speeches in aligned_slides.items() if speeches]
        )
        unaligned_count = len(slides) - aligned_count
        logger.debug(
            f"Alignment summary: {aligned_count} aligned, {unaligned_count} distributed slides"
        )

        if preservation_ratio < 0.90:
            logger.warning(
                f"Content preservation ratio is low: {preservation_ratio:.2%} - some speech content may be missing!"
            )

        logger.debug("=== Enhanced Speech and Slides Combination Completed ===")

    return combined_content


def _extract_slides(slides_markdown, verbose=False):
    """Extract individual slides from markdown (by splitting on major headings)"""
    logger = logging.getLogger(__name__)

    # Split by ## or # headings (common slide markers)
    import re

    slides = []
    current_slide = []

    lines = slides_markdown.split("\n")
    for line in lines:
        # Check if this is a slide heading (## or #)
        if re.match(r"^#+\s", line) and current_slide:
            # Save current slide and start new one
            slides.append("\n".join(current_slide).strip())
            current_slide = [line]
        else:
            current_slide.append(line)

    # Don't forget last slide
    if current_slide:
        slides.append("\n".join(current_slide).strip())

    if verbose:
        logger.debug(f"Extracted {len(slides)} slides from markdown")

    return [s for s in slides if s.strip()]  # Filter empty slides


def _find_matching_paragraph(matched_section, speech_paragraphs, verbose=False):
    """Find which paragraph index contains the matched section"""
    logger = logging.getLogger(__name__)

    if matched_section == "NO_MATCH":
        return None

    # Try exact match first
    for idx, para in enumerate(speech_paragraphs):
        if matched_section.lower() in para.lower():
            if verbose:
                logger.debug(f"Found exact match at paragraph {idx}")
            return idx

    # Fuzzy match: find most similar paragraph
    best_match_idx = None
    best_similarity = 0

    for idx, para in enumerate(speech_paragraphs):
        # Count overlapping words
        matched_words = set(matched_section.lower().split())
        para_words = set(para.lower().split())
        overlap = len(matched_words & para_words)

        if overlap > best_similarity:
            best_similarity = overlap
            best_match_idx = idx

    if best_match_idx is not None and best_similarity > 2:
        if verbose:
            logger.debug(
                f"Found fuzzy match at paragraph {best_match_idx} with {best_similarity} overlapping words"
            )
        return best_match_idx

    if verbose:
        logger.debug("No matching paragraph found")
    return None


def _build_combined_markdown(
    speech_paragraphs, aligned_results, cite_timestamps=False, verbose=False
):
    """Build combined markdown by inserting slides before matching speech sections"""
    logger = logging.getLogger(__name__)

    # Create a mapping of speech index -> list of slides to insert before it
    slides_before_speech = {}
    unaligned_slides = []

    for result in aligned_results:
        if result["speech_idx"] is not None:
            if result["speech_idx"] not in slides_before_speech:
                slides_before_speech[result["speech_idx"]] = []
            slides_before_speech[result["speech_idx"]].append(result["slide_content"])
        else:
            unaligned_slides.append(result)

    # Build combined content
    combined = []

    for idx, para in enumerate(speech_paragraphs):
        # Insert slides before this paragraph if they align with it
        if idx in slides_before_speech:
            for slide_content in slides_before_speech[idx]:
                combined.append(slide_content)
                combined.append("")  # Blank line separator

        combined.append(para)
        combined.append("")  # Blank line separator

    # Add unaligned slides at the end (in order)
    if unaligned_slides:
        if verbose:
            logger.debug(f"Adding {len(unaligned_slides)} unaligned slides at the end")
        combined.append("---\n\n## Unaligned Slides\n")
        for result in unaligned_slides:
            combined.append(result["slide_content"])
            combined.append("")

    result = "\n".join(combined).strip()

    if verbose:
        logger.debug(
            f"Combined content built with {len(speech_paragraphs)} speech paragraphs"
        )
        logger.debug(f"Final combined content length: {len(result)} characters")
        # Calculate final preservation ratio (need to pass original speech length)
        # Note: This will be calculated in the calling function

    return result
