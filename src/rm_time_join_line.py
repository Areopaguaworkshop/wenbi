import re
import gradio as gr
import os

# Function to remove timestamps for both .srt and .vtt formats
def remove_timestamps(subtitle_text, file_extension):
    if file_extension == ".srt":
        # Regex to match SRT timestamps like '00:00:01,000 --> 00:00:04,000'
        pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
    elif file_extension == ".cc.vtt" or file_extension == ".vtt":
        # Regex to match VTT timestamps like '00:00:05.000 --> 00:00:10.000'
        pattern = r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}'
    else:
        return subtitle_text  # No cleaning if format is unknown
    
    # Remove timestamps
    cleaned_text = re.sub(pattern, '', subtitle_text)
    
    # Join lines into paragraphs (by removing unnecessary line breaks)
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)  # Remove extra newlines
    cleaned_text = ' '.join(cleaned_text.splitlines())  # Join remaining lines into a single paragraph
    
    return cleaned_text

# Function to process subtitle files and generate cleaned output in markdown
def process_subtitles(file_paths):
    subtitles = []
    
    # Process each uploaded file
    for file_path in file_paths:
        file_name = file_path.split("/")[-1]
        # Extract the file extension based on the file metadata
        file_extension = ".srt" if file_name.endswith(".srt") else ".vtt" if file_name.endswith(".vtt") else ".cc.vtt"

        # Try to read the file content as UTF-8; if that fails, try ISO-8859-1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content= f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                content= f.read()

        # Clean the subtitles by removing timestamps and joining lines
        cleaned_text = remove_timestamps(content, file_extension)
        
        # Append filename and cleaned text for markdown generation
        subtitles.append((file_name, cleaned_text.strip()))
    
    # Generate markdown output
    markdown_output = ""
    for filename, cleaned_text in subtitles:
        markdown_output += f"# {filename}\n\n"
        markdown_output += cleaned_text + "\n\n"
    
    # Save markdown output to a temporary file
    output_file = "cleaned_subtitles.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    
    return output_file  # Return the file path for download

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        # Upload component for .srt and .cc.vtt files, set type to 'binary'
        subtitle_input = gr.File(label="Upload .srt or .vtt files", file_count="multiple", type="filepath", file_types=[".srt", ".vtt", ".cc.vtt"])
        
        # Output component for markdown file download
        file_output = gr.File(label="Download Cleaned Subtitles")

        # Button to process the files
        submit_button = gr.Button("Process Subtitles")
        
        # Process button action and return downloadable markdown file
        submit_button.click(fn=process_subtitles, inputs=subtitle_input, outputs=file_output)
    
    return demo

# Run Gradio app
if __name__ == "__main__":
    app = create_interface()
    app.launch()
