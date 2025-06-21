import dspy
import os
import re
from datetime import datetime
from wenbi.utils import segment


def get_lm_config(model_string, base_url=None):
    """
    Determine provider and return config dict for dspy.LM.
    Supports: ollama, openai, gemini (google-genai)
    """
    if not model_string:
        # Default to Ollama
        return {
            "base_url": "http://localhost:11434",
            "model": "ollama/qwen3",
        }
    parts = model_string.strip().split("/")
    provider = parts[0].lower() if parts else ""
    if provider == "ollama":
        return {
            "base_url": base_url or "http://localhost:11434",
            "model": model_string,
        }
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return {
            "base_url": base_url or "https://api.openai.com/v1",
            "model": model_string.replace("openai/", ""),
            "api_key": api_key,
        }
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_JSON")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GOOGLE_API_KEY_JSON environment variable not set."
            )
        return {
            "base_url": base_url or "https://generativelanguage.googleapis.com/v1beta",
            "model": model_string.replace("gemini/", ""),
            "api_key": api_key,
        }
    else:
        # Unknown provider, fallback to user input
        return {"base_url": base_url, "model": model_string}


def is_ollama(model_string):
    """
    Check the model_string (if provided) for the provider.
    If the model string starts with "ollama/", return "http://localhost:11434".
    If model_string is empty, default to "ollama/qwen3" with base_url "http://localhost:11434".
    Otherwise, return None.
    """
    if not model_string:
        return "http://localhost:11434"  # default for empty input
    parts = model_string.strip().split("/")
    if parts and parts[0].lower() == "ollama":
        return "http://localhost:11434"
    return None


def translate(
    vtt_path,
    output_dir=None,
    translate_language="Chinese",
    llm="",
    chunk_length=8,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    base_url="http://localhost:11434",
):
    """
    Translate English VTT content to a bilingual markdown file using the target language provided.

    Args:
        vtt_path (str): Path to the English VTT file
        output_dir (str): Directory for output files
        translate_language (str): Target language for translation
        llm (str): LLM model identifier
        chunk_length (int): Number of sentences per chunk for segmentation
        max_tokens (int): Maximum number of tokens for the LLM
        timeout (int): Timeout for the LLM in seconds
        temperature (float): Temperature for the LLM
        base_url (str): Base URL for the LLM

    Returns:
        str: Path to the generated markdown file
    """
    segmented_text = segment(vtt_path, sentence_count=chunk_length)
    paragraphs = segmented_text.split("\n\n")

    model_id = llm if llm else "ollama/qwen3"
    lm_config = get_lm_config(model_id, base_url=base_url)
    lm_config["max_tokens"] = max_tokens
    lm_config["timeout_s"] = timeout
    lm_config["temperature"] = temperature
    lm = dspy.LM(**lm_config)
    dspy.configure(lm=lm)

    class Translate(dspy.Signature):
        english_text = dspy.InputField(desc="English text to translate")
        translated_text = dspy.OutputField(
            desc=f"Translation into {translate_language}"
        )

    translator = dspy.ChainOfThought(Translate)
    translated_pairs = []

    for para in paragraphs:
        if para.strip():
            response = translator(english_text=para)
            translated_pairs.append(
                f"# English\n{para}\n\n# {translate_language}\n{response.translated_text}\n\n---\n"
            )

    markdown_content = "\n".join(translated_pairs)
    output_file = os.path.splitext(vtt_path)[0] + "_bilingual.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_file


def rewrite(
    file_path,
    output_dir=None,
    llm="",
    rewrite_lang="Chinese",
    chunk_length=8,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    base_url="http://localhost:11434",
):
    """
    Rewrites text by first segmenting the file into paragraphs.

    Args:
        file_path (str): Path to the input file
        output_dir (str, optional): Output directory
        llm (str): LLM model identifier
        rewrite_lang (str): Target language for rewriting (default: Chinese)
        chunk_length (int): Number of sentences per chunk for segmentation
        max_tokens (int): Maximum number of tokens for the LLM
        timeout (int): Timeout for the LLM in seconds
        temperature (float): Temperature for the LLM
        base_url (str): Base URL for the LLM
    """
    segmented_text = segment(file_path, sentence_count=chunk_length)
    paragraphs = segmented_text.split("\n\n")

    model_id = llm if llm else "ollama/qwen3"
    lm_config = get_lm_config(model_id, base_url=base_url)
    lm_config["max_tokens"] = max_tokens
    lm_config["timeout_s"] = timeout
    lm_config["temperature"] = temperature
    lm = dspy.LM(**lm_config)
    dspy.configure(lm=lm)

    rewritten_paragraphs = []
    for para in paragraphs:
        class ParaRewrite(dspy.Signature):
            """
            Rewrite this text in {rewrite_lang} from oral to written.  Follow these rules strictly:
            1. Correct any basic grammar, punctuation, or usage errors.
            2. Improve clarity while preserving the original meaning and scholarly tone (trying your best not to change the structure of sentence)
            3. IMPORTANT: Maintaining the original meaning and length (97% of original)
            """
            text: str = dspy.InputField(
                desc=f"Spoken text to rewrite in {rewrite_lang}"
            )
            rewritten: str = dspy.OutputField(
                desc=f"Rewritten paragraph in {rewrite_lang}"
            )
        rewrite = dspy.ChainOfThought(ParaRewrite)
        response = rewrite(text=para)
        rewritten_paragraphs.append(response.rewritten)

    rewritten_text = "\n\n".join(rewritten_paragraphs)
    if output_dir:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(output_dir, f"{base_name}_rewritten.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(rewritten_text)
    else:
        out_file = None

    return rewritten_text


def academic(
    file_path,
    output_dir=None,
    llm="",
    academic_lang="English",
    chunk_length=8,
    max_tokens=250000,
    timeout=3600,
    temperature=0.1,
    base_url="http://localhost:11434",
):
    """Rewrites text in academic style while preserving markdown formatting and footnotes."""
    # Read markdown content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into blocks while preserving markdown and footnote elements
    blocks = []
    current_block = []
    footnotes = {}
    footnote_order = []  # Maintain footnote order

    # First pass: collect all footnote definitions
    for line in content.split('\n'):
        if line.startswith('[^'):
            match = re.match(r'\[\^(\d+)\]:\s*(.*)', line)
            if match:
                num, text = match.groups()
                footnotes[num] = text
                if num not in footnote_order:
                    footnote_order.append(num)

    # Second pass: process text while preserving footnote references
    for line in content.split('\n'):
        if line.startswith('[^'):
            continue  # Skip footnote definitions here
        elif line.startswith(('#', '>', '- ', '* ', '1.')) or line.strip().startswith('|'):
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
            blocks.append(line)
        elif not line.strip():
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append('\n'.join(current_block))

    # Setup LLM
    model_id = llm if llm else "ollama/qwen3"
    lm_config = get_lm_config(model_id, base_url=base_url)
    lm_config["max_tokens"] = max_tokens
    lm_config["timeout_s"] = timeout
    lm_config["temperature"] = temperature
    lm = dspy.LM(**lm_config)
    dspy.configure(lm=lm)

    # Process each block
    academic_blocks = []
    for block in blocks:
        if not block.strip():
            continue
        if block.startswith(('#', '>', '- ', '* ', '1.')) or block.strip().startswith('|'):
            academic_blocks.append(block)
            continue

        class AcademicRewrite(dspy.Signature):
            """
            Rewrite this text in formal academic style in {academic_lang}. Follow these rules strictly:
            1. Correct any basic grammar, punctuation, or usage errors.
            2. Improve clarity while preserving the original meaning and scholarly tone (trying your best not to change the structure of sentence)
            3. Maintaining the original meaning and length (97% of original)
            4. IMPORTANT: Preserve ALL footnote references (e.g., [^1], [^2]) exactly as they appear
            """
            text: str = dspy.InputField(desc=f"Text to rewrite in academic {academic_lang}")
            academic: str = dspy.OutputField(desc=f"Academic rewritten text in {academic_lang}")

        academic_rewrite = dspy.ChainOfThought(AcademicRewrite)
        response = academic_rewrite(text=block)
        academic_blocks.append(response.academic)

    # Combine processed blocks
    academic_text = '\n\n'.join(academic_blocks)

    # Append footnotes in original order
    if footnotes:
        academic_text += '\n\n'
        for num in footnote_order:
            academic_text += f'[^{num}]: {footnotes[num]}\n'

    if output_dir:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(output_dir, f"{base_name}_academic.md")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(academic_text)
    else:
        out_file = None

    return academic_text


def process_docx(
    file_path,
    output_dir=None,
    llm="",
    academic_lang="English",
    chunk_length=8,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    base_url="http://localhost:11434",
):
    """Process a docx file with multiple outputs:
    1. Original markdown
    2. Academic rewrite docx
    3. Academic rewrite markdown
    4. Comparison markdown (redlines)
    5. Track changes docx (pandoc)
    """
    from docx import Document
    import subprocess
    from redlines import Redlines
    import re

    if not output_dir:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 1. Convert original to markdown using pandoc to preserve footnotes
    original_md = os.path.join(output_dir, f"{base_name}_original.md")
    try:
        subprocess.run([
            'pandoc',
            '-f', 'docx',
            '-t', 'markdown',
            '--wrap=none',
            '--markdown-headings=atx',
            '--reference-links',
            file_path,
            '-o', original_md
        ], check=True)
    except subprocess.CalledProcessError:
        # Fallback to simple conversion if pandoc fails
        doc = Document(file_path)
        with open(original_md, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(p.text for p in doc.paragraphs if p.text.strip()))

    # 2. Generate academic rewrite using original markdown as input
    academic_text = academic(
        original_md,  # Changed from file_path to original_md
        output_dir=output_dir,
        llm=llm,
        academic_lang=academic_lang,
        chunk_length=chunk_length,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        base_url=base_url,
    )

    # 3. Generate comparison markdown using redlines
    compare_md = os.path.join(output_dir, f"{base_name}_compare.md")
    with open(original_md, 'r', encoding='utf-8') as f:
        original_text = f.read()
    diff = Redlines(original_text, academic_text)
    with open(compare_md, 'w', encoding='utf-8') as f:
        f.write("# Document Comparison\n\n")
        f.write(f"**Original:** {base_name}\n")
        f.write(f"**Academic Rewrite:** {base_name}_academic\n\n")
        f.write("## Changes\n\n")
        f.write(diff.output_markdown)

    return {
        'original_md': original_md,
        'compare_md': compare_md
    }

def split_text_preserve_footnotes(text):
    """Split text into paragraphs while preserving footnote references"""
    paras = []
    current = []
    footnotes = {}

    for line in text.split('\n'):
        if line.startswith('[^'):
            # Store footnote definition
            match = re.match(r'\[\^(\d+)\]:\s*(.*)', line)
            if match:
                footnotes[match.group(1)] = match.group(2)
        elif line.strip():
            current.append(line)
        elif current:
            paras.append(('\n'.join(current), footnotes))
            current = []
            footnotes = {}

    if current:
        paras.append(('\n'.join(current), footnotes))

    return paras

def add_text_with_footnotes(paragraph, content):
    """Add text to paragraph while properly handling footnote references"""
    text, footnotes = content
    parts = re.split(r'(\[\^\d+\])', text)

    for part in parts:
        if re.match(r'\[\^(\d+)\]', part):
            # Add footnote reference
            footnote_num = re.search(r'\[\^(\d+)\]', part).group(1)
            if footnote_num in footnotes:
                paragraph.add_run().add_footnote(footnotes[footnote_num])
        else:
            paragraph.add_run(part)
