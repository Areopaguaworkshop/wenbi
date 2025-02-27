import dspy
import os
from wenbi.utils import segment

# following functions associated with the models,
# for a better performance, we set a --llm option in both command line
# and gradio interface to allow users to specify the model they want to use.


def translate(vtt_path, output_dir=None, translate_language="Chinese", llm="", chunk_length=8):
    """
    Translate English VTT content to a bilingual markdown file using the target language provided.

    Args:
        vtt_path (str): Path to the English VTT file
        output_dir (str): Directory for output files
        translate_language (str): Target language for translation
        llm (str): LLM model identifier
        chunk_length (int): Number of sentences per chunk for segmentation

    Returns:
        str: Path to the generated markdown file
    """
    segmented_text = segment(vtt_path, sentence_count=chunk_length)
    paragraphs = segmented_text.split("\n\n")

    # Use provided LLM model or default to "ollama/qwen2.5"
    model_id = llm if llm else "ollama/qwen2.5"
    lm = dspy.LM(
        base_url="http://localhost:11434",
        model=model_id,
        max_tokens=50000,
        temperature=0.1,
    )
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
                f"# English\n{para}\n\n# {translate_language}\n{
                    response.translated_text
                }\n\n---\n"
            )

    markdown_content = "\n".join(translated_pairs)
    output_file = os.path.splitext(vtt_path)[0] + "_bilingual.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_file


def rewrite(file_path, output_dir=None, llm="", rewrite_lang="Chinese", chunk_length=8):
    """
    Rewrites text by first segmenting the file into paragraphs.

    Args:
        file_path (str): Path to the input file
        output_dir (str, optional): Output directory
        llm (str): LLM model identifier
        rewrite_lang (str): Target language for rewriting (default: Chinese)
        chunk_length (int): Number of sentences per chunk for segmentation
    """
    segmented_text = segment(file_path, sentence_count=chunk_length)
    paragraphs = segmented_text.split("\n\n")

    # Configure the LM without hard-coding the model parameter (this will be patched externally)
    model_id = llm if llm else "ollama/qwen2.5"
    lm = dspy.LM(
        model=model_id,
        base_url="http://localhost:11434",
        max_tokens=50000,
        timeout_s=3600,
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    rewritten_paragraphs = []
    # Loop over paragraphs and rewrite each individually
    for para in paragraphs:

        class ParaRewrite(dspy.Signature):
            """
            Rewrite this text in {rewrite_lang}, converting from spoken to written form
            while preserving the meaning. Ensure the rewritten text is at least 95%
            of the original length.
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
        return out_file
    return rewritten_text
