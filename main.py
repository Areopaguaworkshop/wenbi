from utils import transcribe, parse_subtitle, video_to_audio, language_detect, audio_wav, download_audio  # updated imports
from model import rewrite, translate  # import both rewriting functions
import os
import gradio as gr
import sys
import dspy

# Add output directory constant
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def process_input(file_path, url="", language="", llm="", multi_language=False, translate_lang="Chinese", output_dir=""):  # Updated function signature
    # Use provided output_dir or fallback to constant OUTPUT_DIR
    out_dir = output_dir if output_dir.strip() else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    # Patch dspy.LM.__init__ so that its "model" parameter is always a string.
    default_model = llm.strip() if llm.strip() else "ollama/qwen2.5"
    orig_init = dspy.LM.__init__
    def new_init(self, *args, **kw):
        kw["model"] = str(default_model)
        orig_init(self, *args, **kw)
    dspy.LM.__init__ = new_init
    
    if not file_path and not url.strip():
        return "Error: No input provided", None, None, None
    
    # Define supported extensions
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    audio_exts = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".opus"}
    subtitle_exts = {".vtt", ".srt", ".ass", ".ssa", ".sub", ".smi", ".txt"}
    
    # Process based on input type
    if url and url.strip():
        try:
            file_path = download_audio(url.strip())
            lang = language if language.strip() else None
            vtt_file, auto_lang = transcribe(file_path, language=lang)
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            return "Error: Failed to process URL", None, None, None
    elif file_path:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in video_exts:
            audio_file = video_to_audio(file_path)
            lang = language if language.strip() else None
            vtt_file, auto_lang = transcribe(audio_file, language=lang)
        elif ext in audio_exts:
            lang = language if language.strip() else None
            wav_file = audio_wav(file_path)
            vtt_file, auto_lang = transcribe(wav_file, language=lang)
        elif ext in subtitle_exts:
            vtt_file = file_path
            auto_lang = language or "unknown"
        else:
            print(f"Warning: Unknown file type {ext}, treating as subtitle file")
            vtt_file = file_path
            auto_lang = language or "unknown"
    else:
        return "Error: Invalid input", None, None, None
        
    # Generate CSV after getting VTT file
    _, filename = os.path.split(vtt_file)
    base_name, _ = os.path.splitext(filename)
    csv_file_path = os.path.join(out_dir, base_name + ".csv")
    vtt_df = parse_subtitle(vtt_file)
    vtt_df.to_csv(csv_file_path, index=True, encoding='utf-8')
    print(f"CSV file '{csv_file_path}' created successfully.")

    # Use language detection and process accordingly
    detected_lang = language_detect(vtt_file)
    print(f"Detected language: {detected_lang}")
    if detected_lang == "en":
        final_output = translate(vtt_file, output_dir=out_dir, translate_language=translate_lang)
    else:  # zh or unknown
        final_output = rewrite(vtt_file, output_dir=out_dir)
        
    return final_output  # Return final markdown output

def create_interface():
    def process_wrapper(file_path, url, language, llm, multi_language, translate_lang):
        multi_lang_bool = multi_language == "True"
        return process_input(file_path, url, language, llm, multi_lang_bool, translate_lang)

    iface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.File(label="Upload File", type="filepath"),
            gr.Textbox(
                label="Or Enter URL (YouTube, etc)",
                value="",
                placeholder="https://youtube.com/watch?v=..."
            ),
            gr.Textbox(
                label="Transcribe Language (optional)",
                value="",
                placeholder="e.g., Chinese, English",
            ),
            gr.Textbox(
                label="LLM Model (optional)",
                value="ollama/qwen2.5",
                placeholder="Enter LLM model identifier"
            ),
            gr.Dropdown(
                label="Multi-language Processing",
                choices=["False", "True"],
                value="False",
                type="value"
            ),
            gr.Textbox(
                label="Translation Language (optional)",
                value="Chinese",
                placeholder="Enter target translation language"
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

