from download import download_all  # Import download helper
from utils import transcribe, parse_subtitle, video_to_audio, language_detect, audio_wav, download_audio
from model import rewrite, translate
import os
import gradio as gr
import sys
import dspy

# Add output directory constant
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Download required models before processing
download_all()

def process_input(file_path=None, url="", language="", rewrite_llm="", translate_llm="", multi_language=False, translate_lang="Chinese", output_dir=""):
    """Process input in three steps:
    1. Convert input (URL/video/audio) to WAV
    2. Generate VTT file(s) via transcription
    3. Process VTT based on language detection
    """
    out_dir = output_dir if output_dir and output_dir.strip() else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    if not file_path and not url:
        return "Error: No input provided", None, None, None

    # Step 1: Convert input to WAV file
    try:
        if url:
            file_path = download_audio(url.strip(), output_dir=out_dir)
        elif file_path:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}:
                file_path = video_to_audio(file_path, output_dir=out_dir)
            elif ext in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".webm", ".opus"}:
                file_path = audio_wav(file_path, output_dir=out_dir)
            # Note: subtitle files don't need conversion
    except Exception as e:
        print(f"Error in Step 1 (Converting to WAV): {e}")
        return "Error: Failed to process input", None, None, None

    # Step 2: Generate VTT file(s) through transcription
    try:
        if multi_language:
            # Multi-speaker transcription path
            from mutilang import transcribe_multi_speaker, speaker_vtt
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            transcriptions = transcribe_multi_speaker(file_path)
            vtt_files = speaker_vtt(transcriptions, output_dir=out_dir, base_filename=base_name)
        else:
            # Single speaker transcription path
            if file_path.lower().endswith(('.vtt', '.srt', '.ass', '.ssa', '.sub', '.smi', '.txt')):
                vtt_files = {None: file_path}  # Use existing subtitle file
            else:
                lang = language if language.strip() else None
                vtt_file, _ = transcribe(file_path, language=lang, output_dir=out_dir)
                vtt_files = {None: vtt_file}
    except Exception as e:
        print(f"Error in Step 2 (Transcription): {e}")
        return "Error: Failed during transcription", None, None, None

    # Step 3: Process VTT file(s) based on language detection
    final_outputs = {}
    try:
        for speaker, vtt_file in vtt_files.items():
            # Generate CSV first (if not multi-speaker)
            if not multi_language:
                base_name = os.path.splitext(os.path.basename(vtt_file))[0]
                csv_file = os.path.join(out_dir, f"{base_name}.csv")
                parse_subtitle(vtt_file).to_csv(csv_file, index=True, encoding='utf-8')
                print(f"CSV file '{csv_file}' created successfully.")

            # Detect language and process accordingly
            detected_lang = language_detect(vtt_file)
            print(f"Detected language for {speaker or 'input'}: {detected_lang}")

            # Configure LLM model and process
            if detected_lang == "zh" and translate_lang.lower() == "chinese":
                # Chinese content - use rewrite with specified model
                output = rewrite(vtt_file, output_dir=out_dir, llm=rewrite_llm)
            else:
                # Non-Chinese content - use translate with specified model
                output = translate(vtt_file, output_dir=out_dir, 
                                translate_language=translate_lang, 
                                llm=translate_llm)

            final_outputs[speaker if speaker else 'output'] = output

        # Return appropriate output based on processing mode
        if multi_language:
            return final_outputs
        else:
            result = final_outputs['output']
            return result, result, csv_file, base_name

    except Exception as e:
        print(f"Error in Step 3 (Language Processing): {e}")
        return "Error: Failed during language processing", None, None, None

def create_interface():
    # Updated textbox label for rewrite LLM model.
    def process_wrapper(file_path, url, language, rewrite_llm, translate_llm, multi_language, translate_lang):
        multi_lang_bool = multi_language == "True"
        return process_input(file_path, url, language, rewrite_llm, translate_llm, multi_lang_bool, translate_lang)
    iface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.File(label="Upload File", type="filepath"),
            gr.Textbox(label="Or Enter URL (YouTube, etc)", value="", placeholder="https://youtube.com/watch?v=..."),
            gr.Textbox(label="Transcribe Language (optional)", value="", placeholder="e.g., Chinese, English"),
            gr.Textbox(label="Rewrite LLM Model (optional)", value="ollama/qwen2.5", placeholder="Enter rewrite LLM model identifier"),
            gr.Textbox(label="Translation LLM Model (optional)", value="ollama/qwen2.5", placeholder="Enter translation LLM model identifier"),
            gr.Dropdown(label="Multi-language Processing", choices=["False", "True"], value="False", type="value"),
            gr.Textbox(label="Translation Language (optional)", value="Chinese", placeholder="Enter target translation language"),
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

