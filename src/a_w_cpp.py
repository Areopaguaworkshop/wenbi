import os
import re
import json
import gradio as gr
from whisper_cpp_python import Whisper
from docx import Document

# Initialize whispercpp in python 
model = Whisper(model_path="/home/ajiap/ai/whisper.cpp/models/ggml-tiny.bin")

# Supported file formats
supported_audio_formats = [".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"]
supported_subtitle_formats = [".srt", ".vtt", ".cc.vtt", ".ass", ".ssa", ".sub", ".txt" ]
supported_formats = supported_audio_formats + supported_subtitle_formats

# Function to transcribe audio to plain text without timestamps
def transcribe_audio_to_text(file_path):
    result = model.transcribe(file_path)
    return result["text"].strip()

# Function to remove timestamps and join lines for subtitle files
def remove_timestamps_and_join_lines(subtitle_text, file_extension):
    if file_extension == ".srt":
        pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
    elif file_extension in [".cc.vtt", ".vtt"]:
        # Updated pattern to handle WebVTT timestamps and metadata
        pattern = (
            r'\d{2}:\d{2}:\d{2}\.\d{3} ?(?:-->|—>|-) ?\d{2}:\d{2}:\d{2}\.\d{3}'  # Standard VTT pattern
            r'|\d{2}:\d{2}\.\d{3} ?(?:-->|—>|-) ?\d{2}:\d{2}\.\d{3}'  # MM:SS.mmm pattern
        )
       # pattern = r'\d{2}:\d{2}\.\d{3} ?(?:-->|->|-) ?\d{2}:\d{2}\.\d{3}' # MM:SS.mmm pattern
                # r'\d{2}:\d{2}:\d{2}\.\d{3} ?(?:-->|->|-) ?\d{2}:\d{2}:\d{2}\.\d{3}' # standard vtt 
                
        subtitle_text = re.sub(r'WEBVTT.*?\n\n', '', subtitle_text, flags=re.DOTALL)  # Remove header
        subtitle_text = re.sub(r'^\d+\n', '', subtitle_text, flags=re.MULTILINE)  # Remove cue identifiers
    elif file_extension in [".ass", ".ssa"]:
        pattern = r'Dialogue: \s*\d+,\s*\d{1}:\d{2}:\d{2}\.\d{2},\s*\d{1}:\d{2}:\d{2}\.\d{2},.*?,(.*?)(?:,\d+)*'
         # r'Dialogue: \d+,\d{1}:\d{2}:\d{2}.\d{2},\d{1}:\d{2}:\d{2}.\d{2},.*?,(.*?),\d{2},\d{2},\d{2},.*'
        cleaned_text = re.sub(pattern, r'\1', subtitle_text)  # Retain only the spoken text
        return cleaned_text.replace("\\N", " ")  # Replace ASS/SSA line breaks with spaces
    elif file_extension == ".sub":
        pattern = r'\d{2}:\d{2}:\d{2}.\d{2},\d{2}:\d{2}:\d{2}.\d{2}' or r'{\d+}{\d+}.*'  # Simplified pattern, may need refinement based on specific .sub format
    elif file_extension == ".txt":
        pattern = r'\d+\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+.*'  # Remove timestamped lines
    else:
        return subtitle_text

    # Remove timestamps
    cleaned_text = re.sub(pattern, '', subtitle_text)

    # Remove line numbers (in SRT files) and extra newlines
    cleaned_text = re.sub(r'^\d+\n', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\n{2,}', ' ', cleaned_text)  # Join paragraphs

    # Join remaining lines
    cleaned_text = re.sub(r'\n', ' ', cleaned_text).strip()

    return cleaned_text

# Function to generate DOCX file
def generate_docx(text, filename="cleaned_subtitles.docx"):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    return filename

# Main function to handle input files
def process_files(files):
    # Check if files input is valid
    subtitles = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()

        # Step 1: If audio file, transcribe to plain text without timestamps
        if file_extension in supported_audio_formats:
            cleaned_text = transcribe_audio_to_text(file_path)
        # Step 2: If it's already a subtitle file, remove timestamps and join lines
        elif file_extension in supported_subtitle_formats:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    subtitle_text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="ISO-8859-1") as f:
                    subtitle_text = f.read()

            # Clean the subtitles by removing timestamps and joining lines
            cleaned_text = remove_timestamps_and_join_lines(subtitle_text, file_extension)
        else:
            print(f"Unsupported file type: {file_extension}. Skipping...")
            continue  # Skip unsupported file types

        subtitles.append((file_name, cleaned_text.strip()))

    # Create outputs for different formats
    markdown_output = ""
    docx_output = "cleaned_subtitles.docx"
    txt_output = "cleaned_subtitles.txt"
    json_output = "cleaned_subtitles.json"

    json_data = {}

    for filename, cleaned_text in subtitles:
        # Markdown format
        markdown_output += f"# {filename}\n\n"
        markdown_output += cleaned_text + "\n\n"
        
        # JSON format
        json_data[filename] = cleaned_text

    # Save Markdown
    with open("cleaned_subtitles.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)

    # Save TXT
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    # Save JSON
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Save DOCX
    generate_docx(markdown_output, docx_output)

    print(f"Files processed. Output generated:\nMarkdown: cleaned_subtitles.md\nDOCX: {docx_output}\nTXT: {txt_output}\nJSON: {json_output}")

    # Return file outputs and error messages
    return markdown_output, docx_output, txt_output, json_output, error_message

# Gradio interface function
def gradio_interface(files):
    markdown_output, docx_output, txt_output, json_output, error_message = process_files(files)
    
    # Read the contents of the markdown file to display in the interface
    with open("cleaned_subtitles.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
        
    # Return markdown content, download links, and error message
    return (
        markdown_content,
        docx_output,
        txt_output,
        json_output,
        error_message
    )

# Define Gradio UI components
file_input = gr.File(file_count="multiple", type="filepath", label="Upload audio or subtitle files")  # Use "filepath"
markdown_output = gr.Markdown(label="Processed Markdown Content")
docx_output = gr.File(label="Download DOCX")  # Do not set type for output file
txt_output = gr.File(label="Download TXT")  # Do not set type for output file
json_output = gr.File(label="Download JSON")  # Do not set type for output file
error_message = gr.Textbox(label="Error Message", lines=2, interactive=False)

# Display supported file formats in Gradio UI
supported_formats_message = f"Supported file formats: {', '.join(supported_formats)}"
info_text = gr.Markdown(supported_formats_message)

# Create Gradio interface
gr.Interface(
    fn=gradio_interface,
    inputs=[file_input],
    outputs=[markdown_output, docx_output, txt_output, json_output, error_message],
    title="AI Subtitle Editor",
    description="Supported audio formats are .mp3, .m4a, .wav, .flac, .ogg, .aac, Supported Subtitle formats are .srt, .vtt, .cc.vtt, .ass, .ssa, .sub, .txt. Other file formats will not supported. Expecting errors due the variaties of patterns of timestamps.I use the whispercpp tiny model in this demo, the result will may not good as you think, just for this demo.").launch()
