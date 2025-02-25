import dspy
import os
from utils import segment
import spacy

# following functions associated with the models, 
# for a better performance, we set a --llm option in both command line
# and gradio interface to allow users to specify the model they want to use. 

def translate_zh(vtt_path):
    """
    Translate English VTT content to Chinese and create a bilingual markdown file.

    Args:
        vtt_path (str): Path to the English VTT file

    Returns:
        str: Path to the generated markdown file
    """
    # Get segmented English text
    segmented_text = segment(vtt_path, sentence_count=10)
    paragraphs = segmented_text.split("\n\n")

    # Configure dspy
    lm = dspy.LM(
        base_url="http://localhost:11434",
        model="ollama/qwen2.5",
        max_tokens=50000,
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    class TranslateToZh(dspy.Signature):
        """Translate English text to Chinese maintaining accuracy and natural expression."""

        english_text = dspy.InputField(desc="English text to translate")
        chinese_translation = dspy.OutputField(desc="Chinese translation")

    translator = dspy.ChainOfThought(TranslateToZh)
    translated_pairs = []

    # Process each paragraph
    for para in paragraphs:
        if para.strip():
            response = translator(english_text=para)
            translated_pairs.append(
                f"# English\n{para}\n\n# 中文\n{
                    response.chinese_translation}\n\n---\n"
            )

    # Combine all translations
    markdown_content = "\n".join(translated_pairs)

    # Save as markdown file
    output_file = os.path.splitext(vtt_path)[0] + "_bilingual.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_file


def rewrite_zh(file_path):
    """Rewrites text by first segmenting the file into paragraphs,
    then rewriting each paragraph one at a time. This is the Chinese rewrite function.
    
    Returns:
        str: The rewritten text (Chinese)
    """
    # Get segmented text (paragraphs separated by double newlines)
    segmented_text = segment(file_path)
    paragraphs = segmented_text.split("\n\n")
    
    # Set up spacy model (using English or Chinese model as needed; here we assume "en")
    language = "en"
    if language == "zh":
        nlp = spacy.load("zh_core_web_sm")
    elif language == "en":
        nlp = spacy.load("en_core_web_sm")
    else:
        raise ValueError("Invalid language. Supported languages are 'zh' and 'en'.")
    
    # Configure the LM without hard-coding the model parameter (this will be patched externally)
    lm = dspy.LM(
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
            重写此段，将口语表达变成书面表达，确保意思不变。
            保证重写后的文本长度不少于原文的95%。
            """
            text: str = dspy.InputField(desc="需要重写的口语讲座")
            rewritten: str = dspy.OutputField(desc="重写后的段落")
        
        rewrite = dspy.ChainOfThought(ParaRewrite)
        response = rewrite(text=para)
        rewritten_paragraphs.append(response.rewritten)
    
    rewritten_text = "\n\n".join(rewritten_paragraphs)
    return rewritten_text
