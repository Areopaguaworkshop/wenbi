# Wenbi

Wenbi is a CLI-first toolkit that turns media (video/audio/URL) and text into structured Markdown, then rewrites or translates the result. It is built around the `ollama/qwen3.5:cloud` rewrite/translation model by default, with DeepL as the preferred translator and an LLM fallback.

It supports:
- Video/audio/URL transcription to VTT/Markdown
- Text rewriting (`rewrite`, `academic` style)
- Translation (`translate`) with **DeepL first**, then **LLM fallback**
- English interview rewriting (`en-en`) with speaker-separated output
- Chinese interview rewriting (`zh-zh`) with speaker-separated output
- English/Chinese bilingual audio extraction (`en-zh`) — keep English, translate to Chinese
- Single-language multi-speaker diarization (`speaker`) with rewrite + translate
- PPT-style slide + speech combination (`ppt`)
- Batch directory processing (`wenbi-batch`)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/wenbi?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/wenbi)

## How it works

Every subcommand follows the same pipeline; the only difference is which stages run and the default options:

```
input (media / URL / text / subtitles)
  │
  1. Download & extract        ── yt-dlp for URLs, ffmpeg/pydub for media → audio
  │
  2. ASR / transcribe          ── auto | gladia | funasr | whisper
  │                                gladia is the default when GLADIA_API_KEY is set;
  │                                funasr (paraformer-zh, local) is the fallback;
  │                                whisper is the third fallback.
  │                                → VTT + raw transcript
  │
  3. Diarize (optional)        ── gladia / funasr cam++ → speaker labels
  │
  4. Rewrite (optional)        ── LLM (default ollama/qwen3.5:cloud)
  │                                rewrites oral speech into written style,
  │                                preserves speaker turns for interview flows
  │
  5. Translate (optional)      ── DeepL first (needs DEEPL_API_KEY),
  │                                LLM fallback if DeepL unavailable/fails.
  │                                EN→ZH glossary applied by default
  │                                (DeepL glossary API + LLM prompt).
  │
  6. Slide combine (ppt only)  ── frame extraction + OCR → slides aligned
  │                                with speech by timestamp
  │
  └── outputs in --output-dir:   *_rewritten.md, *_translated.md,
                                  *_bilingual.md, *_zh.md, *_en.md,
                                  *_combine.md, *_diagnostics.json, *.vtt, *.csv
```

Subcommand → pipeline mapping:

| Command | Stages |
|---|---|
| `rewrite` / `rw` | 1 → 2 → 4 |
| `translate` / `tr` | 1 (or text-only) → 5 |
| `en-en` / `enen` | 1 → 2 → 3 → 4 (interview rewrite defaults) |
| `zh-zh` / `zhzh` | 1 → 2 → 3 → 4 (interview rewrite defaults) |
| `en-zh` / `enzh` | 1 → 2 → 3 (keep EN, drop ZH) → 5 |
| `speaker` / `sp` | 1 → 2 → 3 → 4 → 5 |
| `ppt` / `p` | 1 → 2 → 6 (slides + speech, optional rewrite/translate) |
| `wenbi-batch` | runs one of the above over every media file in a directory |

## Install

Prerequisites:
- Python 3.11+
- `ffmpeg` in PATH
- **Gladia API key** (optional but recommended): set `GLADIA_API_KEY` for cloud ASR (the default for all audio-capable subcommands). Free tier is 10h/month — see https://www.gladia.io/pricing. Without the key, ASR auto-falls-back to FunASR (local, no key needed, requires the heavy ML deps).

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

English interview rewrite:

```bash
wenbi en-en interview.mp4 --gladia-key "$GLADIA_API_KEY"
```

Chinese interview rewrite:

```bash
wenbi zh-zh interview.mp4 --gladia-key "$GLADIA_API_KEY"
```

English/Chinese bilingual audio (keep English, translate to Chinese):

```bash
wenbi en-zh bilingual.mp4 --gladia-key "$GLADIA_API_KEY"
```

Single-language multi-speaker (diarize, rewrite, translate):

```bash
wenbi speaker panel.mp4 --source-lang en --gladia-key "$GLADIA_API_KEY"
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
- `--asr-provider auto|gladia|funasr|whisper`
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
- `--asr-provider auto|gladia|funasr|whisper`
- `--glossary` / `--no-glossary` (default: enabled, EN→ZH term consistency)
- `--glossary-file path.json` (custom `{english: chinese}` glossary)
- `--keep-original-lang`
- `--cite-timestamps`

#### Translation behavior
`translate` uses this order:
1. Try DeepL API first (when key is available).
2. If DeepL is unavailable or chunk translation fails, fallback to LLM.

If both DeepL and LLM are unavailable, translation cannot complete successfully.

### `en-en` (`enen`)
Transcribe an English interview, separate speaker turns, and rewrite it as polished written English using `ollama/qwen3.5:cloud` by default.

```bash
wenbi en-en <input> [options]
```

Key options:
- `--speaker-count` (default: `2`)
- `--asr-provider auto|gladia|funasr|whisper`
- `--gladia-key` (or `GLADIA_API_KEY` env var)
- `--llm` (default: `ollama/qwen3.5:cloud`)
- `--start-time`, `--end-time` (media/URL)

The rewrite preserves speaker labels and adds a `## Questions for Clarification` section when speaker roles, names, terms, or ambiguous ASR phrases need human confirmation.

### `zh-zh` (`zhzh`)
Transcribe a Chinese interview, separate speaker turns, and rewrite it as polished written Chinese using `ollama/qwen3.5:cloud` by default.

```bash
wenbi zh-zh <input> [options]
```

Key options:
- `--speaker-count` (default: `2`)
- `--asr-provider auto|gladia|funasr|whisper`
- `--gladia-key` (or `GLADIA_API_KEY` env var)
- `--llm` (default: `ollama/qwen3.5:cloud`)
- `--start-time`, `--end-time` (media/URL)

The rewrite preserves speaker labels and adds a `## 需要确认的问题` section when speaker roles, names, terms, or ambiguous ASR phrases need human confirmation.

### `en-zh` (`enzh`)
Extract English from English/Chinese bilingual audio (e.g. interpreted interviews), drop the interpreter language, and translate the kept English into Chinese using DeepL first with LLM fallback.

```bash
wenbi en-zh <input> [options]
```

Key options:
- `--asr-provider auto|gladia|funasr|whisper` (default: `auto`)
- `--source-lang` (default: `en`) — language to keep
- `--interpreter-lang` (default: `zh`) — language to drop
- `--gladia-key` (or `GLADIA_API_KEY` env var)
- `--lang` — target translation language (default: `Chinese`)
- `--glossary` / `--no-glossary` (default: enabled, EN→ZH term consistency)
- `--glossary-file path.json` (custom `{english: chinese}` glossary)
- `--no-speaker-labels` — disable speaker diarization
- `--save-json` — write segment diagnostics and raw provider JSON
- `--start-time`, `--end-time` (media/URL)

Outputs include the kept-language VTT/Markdown, a bilingual Markdown side-by-side, and (optionally) a rewritten English Markdown and diagnostics JSON.

### `speaker` (`sp`)
Transcribe single-language multi-speaker audio with diarization, then rewrite and translate it. Same engine as `en-en`/`zh-zh` but without the interview-style rewrite defaults — use it for panels, podcasts, and any multi-speaker source where you want to keep the source language.

```bash
wenbi speaker <input> [options]
```

Key options:
- `--asr-provider auto|gladia|funasr|whisper` (default: `auto`)
- `--source-lang` (default: `en`)
- `--speaker-count` (default: provider decides)
- `--gladia-key` (or `GLADIA_API_KEY` env var)
- `--lang` — target translation language (default: `Chinese`)
- `--glossary` / `--no-glossary` (default: enabled, EN→ZH term consistency)
- `--glossary-file path.json` (custom `{english: chinese}` glossary)
- `--no-speaker-labels` — disable speaker diarization
- `--save-json` — write segment diagnostics and raw provider JSON
- `--start-time`, `--end-time` (media/URL)

Outputs a transcript VTT, transcript Markdown, rewritten Markdown, and (when translation is requested) a bilingual Markdown.

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
- `--asr-provider auto|gladia|funasr|whisper`
- `--transcribe-lang`
- `--multi-language`
- `--verbose`

## Glossary

Translation subcommands (`translate`, `en-zh`, `speaker`) apply a built-in **EN→ZH patristic glossary** by default for term consistency. This covers ~800 Orthodox Christian / patristic terms (e.g. *theosis* → 神化, *theoria* → 静观, *Origen* → 奥利金).

- `--glossary` / `--no-glossary` — toggle the glossary (default: enabled). No-op when the target language is not Chinese.
- `--glossary-file path.json` — supply a custom glossary JSON (`{english: chinese}` dict). Overrides the built-in patristic glossary.

The glossary is wired into both translation paths:
- **DeepL**: creates a DeepL server-side glossary from the term pairs and passes it to `translate_text`.
- **LLM fallback**: injects the glossary as a `glossary` field on the DSPy `TranslateSignature` prompt.

## Output Files

Typical outputs:
- `*_rewritten.md`
- `*_translated.md`
- `*_academic.md`
- `*_en.md`, `*_en.vtt` (English interview transcripts)
- `*_zh.md`, `*_zh.vtt` (Chinese interview transcripts)
- `*_bilingual.md` (en-zh and speaker translated output)
- `*_diagnostics.json` (when `--save-json` is used)
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
