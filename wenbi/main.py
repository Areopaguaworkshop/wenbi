from wenbi.utils import (
    transcribe,
    parse_subtitle,
    extract_audio_segment,  # replace video_to_audio with extract_audio_segment
    language_detect,
    download_audio,
)
from wenbi.model import rewrite, translate
import os
import gradio as gr
import sys
import dspy

# Only use package-relative output dir for web interface
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def process_input(
    file_path=None,
    url="",
    transcribe_lang="",  # renamed from 'language'
    rewrite_llm="",
    translate_llm="",
    multi_language=False,
    translate_lang="Chinese",
    output_dir="",
    rewrite_lang="Chinese",
    chunk_length=8,
    max_tokens=50000,
    timeout=3600,
    temperature=0.1,
    base_url="http://localhost:11434",
    transcribe_model="large-v3",
    timestamp=None,  # Add timestamp parameter
    output_wav=False,  # Add output_wav parameter
):  # Add transcribe_model parameter
    """Process input in three steps:
    1. Convert input (URL/video/audio) to WAV
    2. Generate VTT file(s) via transcription
    3. Process VTT based on language detection
    """
    # Use current directory for CLI, package directory for web interface
    out_dir = (
        output_dir
        if output_dir.strip()
        else OUTPUT_DIR
        if __name__ == "__main__"
        else os.getcwd()
    )
    os.makedirs(out_dir, exist_ok=True)

    if not file_path and not url:
        return "Error: No input provided", None, None, None

    # Step 1: Convert input to WAV file with segment extraction
    try:
        if url:
            file_path = download_audio(url.strip(), output_dir=out_dir, timestamp=timestamp)
        elif file_path:
            # Use extract_audio_segment for all audio/video files
            if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", 
                                         ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".webm", ".opus")):
                file_path = extract_audio_segment(file_path, timestamp, out_dir)
            # Note: subtitle files don't need conversion
    except Exception as e:
        print(f"Error in Step 1 (Converting to WAV): {e}")
        return "Error: Failed to process input", None, None, None

    # Step 2: Generate VTT file(s) through transcription
    try:
        if multi_language:
            from wenbi.mutilang import transcribe_multi_speaker, speaker_vtt

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            transcriptions = transcribe_multi_speaker(
                file_path, model_size=transcribe_model
            )  # Pass model size
            vtt_files = speaker_vtt(
                transcriptions, output_dir=out_dir, base_filename=base_name
            )
        else:
            if file_path.lower().endswith(
                (".vtt", ".srt", ".ass", ".ssa", ".sub", ".smi", ".txt")
            ):
                vtt_files = {None: file_path}
            else:
                lang = transcribe_lang if transcribe_lang.strip() else None
                vtt_file, _ = transcribe(
                    file_path,
                    language=lang,
                    output_dir=out_dir,
                    model_size=transcribe_model,
                )  # Pass model size
                vtt_files = {None: vtt_file}
    except Exception as e:
        print(f"Error in Step 2 (Transcription): {e}")
        return "Error: Failed during transcription", None, None, None

    # Step 3: Process VTT file(s) based on language detection
    final_outputs = {}
    try:
        for speaker, vtt_file in vtt_files.items():
            if not multi_language:
                base_name = os.path.splitext(os.path.basename(vtt_file))[0]
                csv_file = os.path.join(out_dir, f"{base_name}.csv")
                parse_subtitle(vtt_file).to_csv(
                    csv_file, index=True, encoding="utf-8")
                print(f"CSV file '{csv_file}' created successfully.")

            detected_lang = language_detect(vtt_file)
            print(f"Detected language for {
                  speaker or 'input'}: {detected_lang}")

            if detected_lang == "zh" and translate_lang.lower() == "chinese":
                output = rewrite(
                    vtt_file,
                    output_dir=out_dir,
                    llm=rewrite_llm,
                    rewrite_lang=rewrite_lang,
                    chunk_length=chunk_length,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                    base_url=base_url,
                )
            else:
                output = translate(
                    vtt_file,
                    output_dir=out_dir,
                    translate_language=translate_lang,
                    llm=translate_llm,
                    chunk_length=chunk_length,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                    base_url=base_url,
                )
            final_outputs[speaker if speaker else "output"] = output

        if multi_language:
            return final_outputs
        else:
            result = final_outputs["output"]
            return result, result, csv_file, base_name

    except Exception as e:
        print(f"Error in Step 3 (Language Processing): {e}")
        return "Error: Failed during language processing", None, None, None


def create_interface():
    # Updated textbox label for rewrite LLM model.
    def process_wrapper(
        file_path,
        url,
        transcribe_lang,  # renamed from 'language'
        rewrite_llm,
        translate_llm,
        multi_language,
        translate_lang,
    ):
        multi_lang_bool = multi_language == "True"
        return process_input(
            file_path,
            url,
            transcribe_lang,  # pass as transcribe_lang
            rewrite_llm,
            translate_llm,
            multi_lang_bool,
            translate_lang,
        )

    iface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.File(label="Upload File", type="filepath"),
            gr.Textbox(
                label="Or Enter URL (YouTube, etc)",
                value="",
                placeholder="https://youtube.com/watch?v=...",
            ),
            gr.Textbox(
                label="Transcribe Language (optional)",
                value="",
                placeholder="e.g., Chinese, English",
            ),
            gr.Textbox(
                label="Rewrite LLM Model (optional)",
                value="ollama/qwen3",
                placeholder="Enter rewrite LLM model identifier",
            ),
            gr.Textbox(
                label="Translation LLM Model (optional)",
                value="ollama/qwen3",
                placeholder="Enter translation LLM model identifier",
            ),
            gr.Dropdown(
                label="Multi-language Processing",
                choices=["False", "True"],
                value="False",
                type="value",
            ),
            gr.Textbox(
                label="Translation Language (optional)",
                value="Chinese",
                placeholder="Enter target translation language",
            ),
        ],
        outputs=[
            gr.Textbox(label="Final Rewritten Output"),
            gr.File(label="Download Markdown", type="filepath"),
            gr.File(label="Download CSV", type="filepath"),
            gr.Textbox(label="Filename (without extension)"),
        ],
        title="Wenbi, rewriting or tranlsaing all video, audio and subtitle files into a readable markdown files",
        description="Upload a file or provide a URL to convert audio/video/subtitles to markdown and CSV.",
    )
    return iface


if __name__ == "__main__":
    iface = create_interface()
    iface.launch()
