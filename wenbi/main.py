from wenbi.utils import transcribe, parse_subtitle, video_to_audio, language_detect, audio_wav, download_audio
from wenbi.model import rewrite, translate
import os
import gradio as gr
import sys

# Change default output directory to current working directory
OUTPUT_DIR = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def prepare_wav(file_path, url, output_dir):
    """
    Converts the input (URL, video, or audio file) to a WAV file.
    If a URL is provided, downloads the audio.
    If file_path is given:
      - For video files, converts to audio.
      - For audio files, converts to WAV if needed.
      - For subtitle files, returns the file_path unchanged.
    Returns the path to the WAV file (or the original subtitle file).
    """
    if url:
        return download_audio(url.strip(), output_dir=output_dir)
    elif file_path:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}:
            return video_to_audio(file_path, output_dir=output_dir)
        elif ext in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".webm", ".opus"}:
            return audio_wav(file_path, output_dir=output_dir)
        else:
            # Assume it's already a subtitle file.
            return file_path
    else:
        raise ValueError("No valid input provided.")

def wav_vtt(wav_file, language, out_dir, multi_language):
    """
    Transcribes the given audio file.
    If multi_language is True, uses multi-speaker transcription.
    Otherwise, uses single-speaker transcription.
    Returns a tuple (vtt_files, detected_lang).
    """
    if multi_language:
        from wenbi.mutilang import transcribe_multi_speaker, speaker_vtt
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        transcriptions = transcribe_multi_speaker(wav_file, language_hints={"default": language})
        vtt_files = speaker_vtt(transcriptions, output_dir=out_dir, base_filename=base_name)
        detected_lang = language if language.strip() else "unknown"
    else:
        vtt_file, detected_lang = transcribe(wav_file, language=language if language.strip() else None, output_dir=out_dir)
        vtt_files = {None: vtt_file}
    return vtt_files, detected_lang

def process_input(file_path=None, url="", language="", rewrite_llm="", translate_llm="",
                  multi_language=False, translate_lang="Chinese", output_dir="", 
                  rewrite_lang="Chinese", chunk_length=8):
    out_dir = output_dir if output_dir and output_dir.strip() else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    if not file_path and not url:
        return "Error: No input provided", None, None, None

    try:
        wav_file = prepare_wav(file_path, url, out_dir)
        
        # Determine if input is subtitle and handle language detection
        subtitle_exts = {'.vtt', '.srt', '.ass', '.ssa', '.sub', '.smi', '.txt'}
        if wav_file.lower().endswith(tuple(subtitle_exts)):
            # For subtitle files, use direct language detection
            detected_lang = language_detect(wav_file, language)
            vtt_files = {None: wav_file}
        else:
            # For audio/video files, use transcription
            vtt_files, detected_lang = wav_vtt(wav_file, language, out_dir, multi_language)

        # Set default model and select appropriate model based on language
        default_model = "ollama/qwen2.5"
        if detected_lang.lower() in ["zh", "chinese", "zh-cn", "zh-tw"]:
            current_model = rewrite_llm.strip() if rewrite_llm.strip() else default_model
        else:
            current_model = translate_llm.strip() if translate_llm.strip() else default_model

    except Exception as e:
        print(f"Error in processing step: {e}")
        return "Error: Failed to process input", None, None, None

    # Post-process VTT files
    final_outputs = {}
    try:
        for speaker, vtt_file in vtt_files.items():
            base_name = os.path.splitext(os.path.basename(vtt_file))[0]
            csv_file = os.path.join(out_dir, f"{base_name}.csv")
            # Pass chunk_length to parse_subtitle
            parse_subtitle(vtt_file).to_csv(csv_file, index=True, encoding='utf-8')
            
            if detected_lang.lower() in ["zh", "chinese", "zh-cn", "zh-tw"]:
                output = rewrite(vtt_file, output_dir=out_dir, 
                               llm=current_model, rewrite_lang=rewrite_lang,
                               chunk_length=chunk_length)  # Add chunk_length
            else:
                output = translate(vtt_file, output_dir=out_dir,
                                translate_language=translate_lang, 
                                llm=current_model,
                                chunk_length=chunk_length)  # Add chunk_length
            
            # Use consistent key naming
            key = f"speaker_{speaker}" if multi_language and speaker is not None else 'output'
            final_outputs[key] = output

        if multi_language:
            return final_outputs
        else:
            result = final_outputs['output']
            return result, result, csv_file, base_name

    except Exception as e:
        print(f"Error in post-processing: {e}")
        return "Error: Failed during language processing", None, None, None

def create_interface():
    def process_wrapper(file_path, url, language, rewrite_llm, translate_llm, 
                       multi_language, translate_lang, chunk_length):
        multi_lang_bool = multi_language == "True"
        return process_input(file_path, url, language, rewrite_llm, translate_llm,
                           multi_lang_bool, translate_lang, rewrite_lang="Chinese",
                           chunk_length=int(chunk_length))
    
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
            gr.Number(label="Chunk Length", value=8, 
                     info="Number of sentences per paragraph in output"),
        ],
        outputs=[
            gr.Textbox(label="Final Rewritten Output"),
            gr.File(label="Download Markdown", type="filepath"),
            gr.File(label="Download CSV", type="filepath"),
            gr.Textbox(label="Filename (without extension)"),
        ],
        title="Wenbi: Convert audio/video/subtitles into Markdown and CSV",
        description="Upload a file or provide a URL to convert audio/video/subtitles to markdown and CSV.",
    )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()

