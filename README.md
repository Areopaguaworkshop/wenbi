# Wenbi

Wenbi is a versatile tool for processing video, audio, and subtitle files. It can transcribe, translate, and rewrite content, making it easy to convert spoken language into written text.

## Features

- **Transcription:** Convert video and audio files into text using OpenAI's Whisper models.
- **Translation:** Translate transcribed text into a specified language.
- **Rewriting:** Refine transcribed text into a more formal, written style.
- **Multiple Input Formats:** Supports video files, audio files, YouTube URLs, and subtitle files.
- **Batch Processing:** Process multiple files in a directory at once.
- **Configuration:** Use a YAML file to define complex workflows, including processing file segments and multiple files.
- **Gradio GUI:** Provides a graphical user interface for ease of use.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/wenbi.git
    cd wenbi
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.lock
    ```

## Usage

### `wenbi`

The `wenbi` command is the main entry point for the tool.

**Basic Usage:**

```bash
wenbi <input_file_or_url> [options]
```

**Arguments:**

*   `input`: Path to the input file (video, audio, or subtitle) or a URL.

**Options:**

*   `--config, -c`: Path to a YAML configuration file.
*   `--output-dir, -o`: Output directory.
*   `--gui, -g`: Launch the Gradio GUI.
*   `--rewrite_llm`: Rewrite LLM model identifier.
*   `--translate_llm`: Translation LLM model identifier.
*   `--transcribe-lang, -s`: Transcribe language.
*   `--translate-lang, -t`: Target translation language (default: Chinese).
*   `--rewrite-lang, -r`: Target language for rewriting (default: Chinese).
*   `--academic-lang, -a`: Target language for academic writing (default: English).
*   `--multi-language, -m`: Enable multi-language processing.
*   `--chunk-length, -cl`: Number of sentences per paragraph for LLM processing (default: 8).
*   `--max-tokens, -mt`: Maximum tokens for LLM output (default: 50000).
*   `--timeout, -to`: LLM request timeout in seconds (default: 3600).
*   `--temperature, -tm`: LLM temperature parameter (default: 0.1).
*   `--base-url, -u`: Base URL for LLM API (default: http://localhost:11434).
*   `--transcribe-model, -tsm`: Whisper model size for transcription (default: large-v3-turbo).
*   `--output_wav, -ow`: Filename for saving the segmented WAV.
*   `--start_time, -st`: Start time for extraction (format: HH:MM:SS).
*   `--end_time, -et`: End time for extraction (format: HH:MM:SS).

### `wenbi-batch`

The `wenbi-batch` command processes all media files in a directory.

**Basic Usage:**

```bash
wenbi-batch <input_directory> [options]
```

**Arguments:**

*   `input_dir`: Input directory containing media files.

**Options:**

*   `--config, -c`: YAML configuration file.
*   `--output-dir`: Output directory.
*   `--md`: Output combined markdown file path.

### YAML Configuration

The YAML configuration file allows for more complex workflows.

**Example:**

```yaml
# config/single-input.yaml
input: "example/Phd-finalDraft-2025-07-02_split/1.wav"
output_dir: "output"
rewrite_llm: "mistral"
translate_llm: "mistral"
transcribe_lang: "en"
translate_lang: "zh"
rewrite_lang: "zh"
multi_language: true
chunk_length: 8
max_tokens: 50000
timeout: 3600
temperature: 0.1
base_url: "http://localhost:11434"
transcribe_model: "large-v3-turbo"
```

**To use the config file:**

```bash
wenbi --config config/single-input.yaml
```

### `mini.py`

The `mini.py` script is a simple tool for processing subtitle files.

**Usage:**

```bash
python mini.py <subtitle_file> [-s <sentence_count>]
```

**Arguments:**

*   `file`: Path to the subtitle file.
*   `--sentences, -s`: Maximum sentences per paragraph (default: 10).