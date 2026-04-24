# Wenbi

Wenbi converts media and text into structured Markdown, then rewrites or translates it.

It supports:
- Video/audio/URL transcription to VTT/Markdown
- Text rewriting (`rewrite`, `academic` style)
- Translation (`translate`) with **DeepL first**, then **LLM fallback**
- PPT-style slide + speech combination (`ppt`)
- Batch directory processing (`wenbi-batch`)

## Install

Prerequisites:
- Python 3.10+
- `ffmpeg` in PATH

Install:

```bash
pip install wenbi
```

Or from a local checkout:

```bash
# from the project directory
pip install -e .
```

## Quick Start

Rewrite:

```bash
wenbi rewrite input.mp4 --lang Chinese --llm ollama/qwen3.5:cloud
```

Translate (DeepL first):

```bash
wenbi translate input.md --lang Chinese --deepl-key "$DEEPL_API_KEY"
```

PPT workflow:

```bash
wenbi ppt lecture.mp4 --lang English
```

## Commands

### `rewrite` (`rw`)
Rewrite spoken/transcribed text into written style.

```bash
wenbi rewrite <input> [options]
```

Key options:
- `--style rewrite|academic`
- `--lang`
- `--llm`
- `--cite-timestamps`
- `--start-time`, `--end-time` (media/URL)

### `translate` (`tr`)
Translate content to a target language.

```bash
wenbi translate <input> --lang <target> [options]
```

Key options:
- `--deepl-key` (or `DEEPL_API_KEY` env var)
- `--llm` (fallback model)
- `--keep-original-lang`
- `--cite-timestamps`

#### Translation behavior
`translate` uses this order:
1. Try DeepL API first (when key is available).
2. If DeepL is unavailable or chunk translation fails, fallback to LLM.

If both DeepL and LLM are unavailable, translation cannot complete successfully.

### `ppt` (`p`)
Extract slides from video, align with speech, and export combined markdown.

```bash
wenbi ppt <video_or_url> [options]
```

Key options:
- `--frame-interval`
- `--cropped-slide [auto|x0,y0,x1,y1]`
- `--ppt <ppt/pdf/image/odp>`
- `--no-ocr`
- `--no-clean`
- `--ssim-threshold`, `--hist-threshold`, `--dedup-method`

## Supported Inputs

- Media: `.mp4 .avi .mov .mkv .flv .wmv .m4v .webm .mp3 .flac .aac .ogg .m4a .opus`
- Text/subtitles: `.vtt .srt .ass .ssa .sub .smi .txt .md .markdown .docx`
- URL inputs are supported for media flows.

## Common Global Options

Used by subcommands:
- `--output-dir`
- `--lang`
- `--llm`
- `--chunk-length`
- `--max-tokens`
- `--timeout`
- `--temperature`
- `--transcribe-model`
- `--transcribe-lang`
- `--multi-language`
- `--verbose`

## Output Files

Typical outputs:
- `*_rewritten.md`
- `*_translated.md`
- `*_academic.md`
- `*_combine.md` / `*_combine_clean.md` (PPT workflows)
- `*.vtt`, `*.csv` (depending on flow)

## Batch Processing

Process a directory of media files:

```bash
wenbi-batch <input_dir> --output-dir <dir> --md
```

Optional config:

```bash
wenbi-batch <input_dir> --config config.yaml
```

## YAML Config (CLI)

`wenbi` supports YAML via `--config`.

Example:

```yaml
input: lecture.mp4
output_dir: ./out
llm: ollama/qwen3.5:cloud
lang: Chinese
chunk_length: 20
```

Multi-input format is also supported using `inputs:`.

## Python API

```python
from wenbi.main import process_input

text, md_file, csv_file, base_name = process_input(
    file_path="input.mp4",
    subcommand="translate",
    lang="Chinese",
    use_deepl=True,
    deepl_key="<DEEPL_KEY>",
    llm="ollama/qwen3.5:cloud",
)
```

## Troubleshooting

- No DeepL translation output:
  - Set `DEEPL_API_KEY` or `--deepl-key`
  - Run with `--verbose` to confirm DeepL connectivity
- Fallback LLM not working:
  - Ensure your provider is reachable (for example, Ollama running locally for `ollama/...`)
- PPT OCR issues:
  - Ensure `marker_single` and OCR dependencies are installed correctly

## License

Apache-2.0
