# Wenbi: Intelligent Content Transformation

Wenbi is a versatile command-line interface (CLI) and web application designed to process various forms of media and text, transforming them into structured Markdown and CSV outputs. It leverages Large Language Models (LLMs) for advanced functionalities like transcription, translation, text rewriting, and academic rewriting.

## Features

*   **Multi-Input Support:** Process video, audio, YouTube/web URLs, VTT, SRT, ASS, SSA, SUB, SMI, TXT, Markdown, DOCX, and PDF files.
*   **Transcription:** Convert spoken content from audio/video into text.
*   **Translation:** Translate transcribed or existing text into a target language.
*   **Text Rewriting:** Rewrite text, converting oral expressions to written form, with grammar correction and proofreading.
*   **Academic Rewriting:** Transform text into a formal academic style, preserving meaning and citations.
*   **Batch Processing:** Efficiently process multiple media files within a directory.
*   **LLM Integration:** Seamlessly integrate with various LLMs, including:
    *   Ollama (e.g., `ollama/qwen3`)
    *   Gemini (e.g., `gemini/gemini-1.5-flash`)
    *   OpenAI (e.g., `openai/gpt-4o`)
*   **Configuration:** Flexible configuration via command-line arguments or YAML files.
*   **Gradio GUI:** An intuitive web-based graphical user interface for easy interaction.
*   **Multi-language Processing:** Support for processing content in multiple languages.

## Installation

Wenbi uses `rye` for dependency management. To install, ensure you have `rye` installed, then clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/wenbi.git # Replace with actual repo URL
cd wenbi
rye sync
```

## Usage

### CLI (Command Line Interface)

Wenbi provides a powerful CLI for various tasks. The main entry point is `wenbi`.

#### Main Command

Process a single input file (video, audio, URL, or text file) to generate Markdown and CSV outputs.

```bash
wenbi <input_file_or_url> [options]

# Example: Process a video file
wenbi my_video.mp4 --output-dir ./output --lang English

# Example: Process a YouTube URL
wenbi https://www.youtube.com/watch?v=dQw4w9WgXcQ --llm gemini/gemini-1.5-flash --lang Chinese

# Example: Process a VTT subtitle file
wenbi subtitles.vtt --output-dir ./output --lang English

# Example: Process a DOCX file for academic rewriting (requires --llm)
wenbi document.docx --llm ollama/qwen3 --lang English

# Example: Process a PDF file (requires --llm)
wenbi research_paper.pdf --llm ollama/qwen3 --lang English
```

**Common Options:**

*   `-c, --config <path>`: Path to a YAML configuration file.
*   `-o, --output-dir <path>`: Directory to save output files.
*   `--llm <model_identifier>`: Specify the LLM model to use (e.g., `ollama/qwen3`, `gemini/gemini-1.5-flash`, `openai/gpt-4o`).
*   `-s, --transcribe-lang <language>`: Language for transcription (e.g., `Chinese`, `English`).
*   `-l, --lang <language>`: Target language for translation/rewriting (default: `Chinese`).
*   `-m, --multi-language`: Enable multi-language processing.
*   `-cl, --chunk-length <int>`: Number of sentences per paragraph (default: 8).
*   `-mt, --max-tokens <int>`: Maximum tokens for LLM output (default: 130000).
*   `-to, --timeout <int>`: LLM request timeout in seconds (default: 3600).
*   `-tm, --temperature <float>`: LLM temperature parameter (default: 0.1).
*   `-tsm, --transcribe-model <model_size>`: Whisper model size for transcription (e.g., `large-v3-turbo`).
*   `-ow, --output_wav <filename>`: Filename for saving the segmented WAV (optional).
*   `-st, --start_time <HH:MM:SS>`: Start time for extraction from media.
*   `-et, --end_time <HH:MM:SS>`: End time for extraction from media.

#### Subcommands

Wenbi also provides specific subcommands for `rewrite`, `translate`, and `academic` tasks.

```bash
# Rewrite text
wenbi rewrite <input_file> --llm ollama/qwen3 --lang Chinese

# Translate text
wenbi translate <input_file> --llm gemini/gemini-1.5-flash --lang French

# Academic rewriting
wenbi academic <input_file> --llm openai/gpt-4o --lang English
```

Subcommands share common options with the main command.

### Batch Processing

Process multiple media files in a directory using `wenbi-batch`.

```bash
wenbi-batch <input_directory> [options]

# Example: Process all media files in 'my_media_folder'
wenbi-batch my_media_folder --output-dir ./batch_output --translate-lang English

# Example: Process with a config file and combine markdown outputs
wenbi-batch my_media_folder -c config/batch-config.yml --md combined_output.md
```

**Batch Options:**

*   `-c, --config <path>`: Path to a YAML configuration file for batch processing.
*   `--output-dir <path>`: Output directory for batch results.
*   `--rewrite-llm <model_id>`: LLM for rewriting.
*   `--translate-llm <model_id>`: LLM for translation.
*   `--transcribe-lang <language>`: Language for transcription.
*   `--translate-lang <language>`: Target language for translation (default: `Chinese`).
*   `--rewrite-lang <language>`: Target language for rewriting (default: `Chinese`).
*   `--multi-language`: Enable multi-language processing.
*   `--chunk-length <int>`: Number of sentences per chunk.
*   `--max-tokens <int>`: Maximum tokens for LLM.
*   `--timeout <int>`: LLM timeout in seconds.
*   `--temperature <float>`: LLM temperature.
*   `--md [path]`: Output combined markdown file. If no path, uses input folder name.

### Configuration Files (YAML)

Wenbi supports YAML configuration files for both single input and batch processing. This allows for more complex and reusable configurations.

**Example `single-input.yaml`:**

```yaml
input: "path/to/your/video.mp4"
output_dir: "./my_output"
llm: "gemini/gemini-1.5-flash"
lang: "English"
chunk_length: 10
```

**Example `multiple-inputs.yaml` (for `wenbi` main command):**

```yaml
inputs:
  - input: "path/to/video1.mp4"
    segments:
      - start_time: "00:00:10"
        end_time: "00:00:30"
        title: "Introduction"
      - start_time: "00:01:00"
        end_time: "00:01:30"
        title: "Key Points"
  - input: "path/to/audio.mp3"
    llm: "ollama/qwen3"
    lang: "Chinese"
```

**Example `batch-folder-config.yml` (for `wenbi-batch`):**

```yaml
output_dir: "./batch_results"
translate_llm: "gemini/gemini-1.5-flash"
translate_lang: "French"
chunk_length: 12
```

### Gradio GUI

Launch the web-based Gradio interface for an interactive experience:

```bash
wenbi --gui
```

## Supported Input Types

*   **Video:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.m4v`, `.webm`
*   **Audio:** `.mp3`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.opus`
*   **URLs:** YouTube and other web URLs.
*   **Subtitle Files:** `.vtt`, `.srt`, `.ass`, `.ssa`, `.sub`, `.smi`
*   **Text Files:** `.txt`, `.md`, `.markdown`
*   **Document Files:** `.docx`, `.pdf`

## Output

Wenbi generates the following output files:

*   **Markdown (`.md`):** Contains the processed text (transcribed, translated, rewritten, or academic).
*   **CSV (`.csv`):** For transcribed content, provides a structured breakdown of segments and timestamps.
*   **Comparison Markdown (`_compare.md`):** For academic rewriting, a markdown file showing changes between original and academic text (requires `redlines` library).

## LLM Integration

Wenbi uses `dspy` for LLM integration, allowing flexibility in choosing your preferred model. Ensure your environment variables are set for API keys if using commercial LLMs (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

To use Ollama models, ensure your Ollama server is running locally.

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` (if available) for guidelines on how to contribute to this project. If not, please open an issue to discuss your proposed changes.

## License

This project is licensed under the Apache-2.0 License. See the `LICENSE` file for details.))
