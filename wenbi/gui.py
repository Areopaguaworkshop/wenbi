import gradio as gr
import os
import logging
import io
from wenbi.main import process_input

# Setup logging for GUI
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def launch_gui():
    """Launches the Gradio GUI."""
    iface = create_interface()
    iface.launch()


def process_wrapper(
    file_path,
    url,
    subcommand,
    transcribe_lang,
    target_lang,
    # Advanced options
    transcribe_model,
    llm,
    chunk_length,
    max_tokens,
    temperature,
    timeout,
    cite_timestamps,
    multi_language,
    show_debug_logs,
):
    """
    Wrapper function to call the core processing logic and handle GUI I/O.
    """
    log_capture_string.seek(0)
    log_capture_string.truncate(0)

    params = {
        "file_path": file_path.name if file_path else None,
        "url": url,
        "subcommand": subcommand.lower() if subcommand else "rewrite",
        "transcribe_lang": transcribe_lang,
        "lang": target_lang,
        "transcribe_model": transcribe_model,
        "llm": llm,
        "chunk_length": chunk_length,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "cite_timestamps": cite_timestamps,
        "multi_language": multi_language,
        "verbose": True,  # Always verbose for GUI debugging
    }

    try:
        output_text, md_file, csv_file, _ = process_input(**params)
        
        debug_logs = log_capture_string.getvalue() if show_debug_logs else ""
        
        # Ensure file paths are valid for Gradio output
        md_output = md_file if md_file and os.path.exists(md_file) else None
        csv_output = csv_file if csv_file and os.path.exists(csv_file) else None

        return output_text, md_output, csv_output, debug_logs

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        debug_logs = log_capture_string.getvalue()
        return f"Error: {e}", None, None, debug_logs


def create_interface():
    """Creates and configures the Gradio interface."""
    with gr.Blocks() as iface:
        gr.Markdown(
            "# Wenbi: Rewrite or translate video, audio, and text.",
            "Upload a file or provide a URL to transcribe, rewrite, or translate content.",
        )

        with gr.Row():
            file_input = gr.File(label="Upload File", type="filepath")
            url_input = gr.Textbox(label="Or Enter URL", placeholder="https://youtube.com/watch?v=...")

        with gr.Row():
            subcommand = gr.Dropdown(
                label="Action",
                choices=["Rewrite", "Translate", "Academic"],
                value="Rewrite",
                info="Select the main operation to perform.",
            )
            transcribe_lang = gr.Textbox(
                label="Source Language",
                placeholder="e.g., English, Chinese (leave empty for auto-detect)",
            )
            target_lang = gr.Textbox(
                label="Target Language",
                value="Chinese",
                placeholder="e.g., Chinese, English",
            )

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                transcribe_model = gr.Dropdown(
                    label="Whisper Model",
                    choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"],
                    value="large-v3-turbo",
                )
                llm = gr.Textbox(
                    label="LLM Model",
                    value="ollama/qwen3",
                    placeholder="e.g., ollama/qwen3, gemini/gemini-1.5-flash",
                )
            with gr.Row():
                chunk_length = gr.Slider(label="Chunk Length (Sentences)", minimum=1, maximum=50, value=20, step=1)
                max_tokens = gr.Slider(label="Max Tokens", minimum=1000, maximum=150000, value=130000, step=1000)
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.1, step=0.1)
                timeout = gr.Slider(label="Timeout (seconds)", minimum=60, maximum=7200, value=3600, step=60)
            with gr.Row():
                cite_timestamps = gr.Checkbox(label="Cite Timestamps", value=False)
                multi_language = gr.Checkbox(label="Multi-language Diarization", value=False)
                show_debug_logs = gr.Checkbox(label="Show Debug Logs", value=True)

        submit_btn = gr.Button("Process", variant="primary")

        with gr.Tab("Outputs"):
            output_text = gr.Markdown(label="Processed Text")
            with gr.Row():
                md_output = gr.File(label="Download Markdown")
                csv_output = gr.File(label="Download CSV")
        
        with gr.Tab("Logs"):
            debug_logs = gr.Textbox(label="Debug Logs", lines=15, interactive=False)

        submit_btn.click(
            fn=process_wrapper,
            inputs=[
                file_input, url_input, subcommand, transcribe_lang, target_lang,
                transcribe_model, llm, chunk_length, max_tokens, temperature, timeout,
                cite_timestamps, multi_language, show_debug_logs
            ],
            outputs=[output_text, md_output, csv_output, debug_logs],
        )

    return iface
