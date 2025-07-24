import dspy
import os
import logging
from wenbi.utils import segment


def configure_lm(model_string, verbose=False, **kwargs):
    """Configure the Language Model with verbose logging support"""
    logger = logging.getLogger(__name__)

    if not model_string:
        model_string = "ollama/qwen3"

    if verbose:
        logger.debug(f"Configuring LLM: {model_string}")

    parts = model_string.strip().split("/")
    provider = parts[0].lower() if parts else ""

    if verbose:
        logger.debug(f"LLM provider: {provider}")

    config = kwargs
    if provider == "ollama":
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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        config.update(
            {
                "api_base": "https://api.openai.com/v1",
                "model": model_string.replace("openai/", ""),
                "api_key": api_key,
            }
        )
        if verbose:
            logger.debug(f"OpenAI configuration: model={config['model']}")
        lm = dspy.OpenAI(**config)
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GOOGLE_API_KEY_JSON")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GOOGLE_API_KEY_JSON environment variable not set."
            )

        # Extract the actual model name (e.g., "gemini-2.5-flash" from "gemini/gemini-2.5-flash")
        model_name = (
            model_string.split(
                "/", 1)[1] if "/" in model_string else model_string
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
    llm="ollama/qwen3",
    chunk_length=20,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    cite_timestamps=False,
    verbose=False,
):
    """
    Translate text content using LLM with verbose logging support.
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.debug("=== Starting Translation Process ===")
        logger.debug(f"Input file: {input_file}")
        logger.debug(f"Target language: {translate_language}")
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

    class TranslateSignature(dspy.Signature):
        """Translate the given text to the target language while preserving the original meaning and style."""

        text_to_translate = dspy.InputField(
            desc="Text content to be translated")
        target_language = dspy.InputField(
            desc="Target language for translation")
        translated_text = dspy.OutputField(
            desc="Translated text in the target language"
        )

    translate_module = dspy.Predict(TranslateSignature)

    if verbose:
        logger.debug("LLM translation module initialized")

    # Read and segment the input text
    segmented_text = segment(input_file, chunk_length,
                             cite_timestamps, verbose=verbose)

    # Split into chunks for processing
    chunks = segmented_text.split("\n\n")
    if verbose:
        logger.debug(f"Text divided into {len(chunks)} chunks for processing")

    translated_chunks = []

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue

        if verbose:
            logger.debug(
                f"Translating chunk {
                    i}/{len(chunks)} ({len(chunk)} characters)"
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
                result = translate_module(
                    text_to_translate=chunk, target_language=translate_language
                )
                translated_chunks.append(result.translated_text)

                if verbose:
                    logger.debug(f"Chunk {i} translated successfully")

        except Exception as e:
            error_msg = f"Error translating chunk {i}: {e}"
            if verbose:
                logger.debug(error_msg)
            print(error_msg)
            translated_chunks.append(f"[Translation Error: {chunk}]")

    final_translation = "\n\n".join(translated_chunks)

    if verbose:
        logger.debug(
            f"Translation completed. Final text length: {
                len(final_translation)
            } characters"
        )
        logger.debug("=== Translation Process Completed ===")

    return final_translation


def rewrite(
    input_file,
    output_dir="",
    rewrite_language="Chinese",
    llm="ollama/qwen3",
    chunk_length=20,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    cite_timestamps=False,
    verbose=False,
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

    # Configure LLM
    lm = configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )

    class RewriteSignature(dspy.Signature):
        """
        Rewrite this text in {rewrite_lang} from oral to written.  Follow these rules strictly:
        1. Correct any basic grammar, punctuation, or usage errors.
        2. Improve clarity while preserving the original meaning and scholarly tone (trying your best not to change the structure of sentence)
        3. IMPORTANT: Maintaining the original meaning and length (97% of original)
        """

        oral_text = dspy.InputField(desc="Oral or spoken text to be rewritten")
        target_language = dspy.InputField(
            desc="Target language for the rewriting")
        written_text = dspy.OutputField(
            desc="Polished written version of the text")

    rewrite_module = dspy.Predict(RewriteSignature)

    if verbose:
        logger.debug("LLM rewrite module initialized")

    # Read and segment the input text
    segmented_text = segment(input_file, chunk_length,
                             cite_timestamps, verbose=verbose)

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
            logger.debug(f"Rewriting chunk {
                         i}/{len(chunks)} ({len(chunk)} characters)")

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
                rewritten_chunks.append(chunk)
            else:
                result = rewrite_module(
                    oral_text=chunk, target_language=rewrite_language
                )
                rewritten_chunks.append(result.written_text)

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
            f"Rewrite completed. Final text length: {
                len(final_rewrite)} characters"
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
    llm="ollama/qwen3",
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

    class AcademicSignature(dspy.Signature):
        """
        Rewrite this text in formal academic style in {academic_lang}. Follow these rules strictly:
        1. Correct any basic grammar, punctuation, or usage errors.
        2. Improve clarity while preserving the original meaning and scholarly tone (trying your best not to change the structure of sentence)
        3. Maintaining the original meaning and length (97% of original)
        4. IMPORTANT: Preserve ALL footnote references (e.g., [^1], [^2]) exactly as they appear
        """

        input_text = dspy.InputField(
            desc="Original text to be transformed into academic style"
        )
        target_language = dspy.InputField(
            desc="Target language for academic writing")
        academic_text = dspy.OutputField(
            desc="Text rewritten in formal academic style")

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
            logger.debug(
                f"Converting chunk {i}/{len(chunks)} to academic style ({
                    len(chunk)
                } characters)"
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
                academic_chunks.append(chunk)
            else:
                result = academic_module(
                    input_text=chunk, target_language=academic_lang
                )
                academic_chunks.append(result.academic_text)

                if verbose:
                    logger.debug(
                        f"Chunk {i} converted to academic style successfully")

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
