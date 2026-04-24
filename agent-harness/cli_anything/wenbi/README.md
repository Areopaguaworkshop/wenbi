# cli-anything-wenbi

Stateful CLI harness for the [wenbi](https://pypi.org/project/wenbi/) media-to-markdown application.

## Install

```bash
pip install -e .
```

Or with wenbi dependencies:

```bash
pip install -e ".[wenbi]"
```

## Quick Start

```bash
# Rewrite a video's speech to written form
cli-anything-wenbi rewrite lecture.mp4 --lang English

# Translate a markdown file
cli-anything-wenbi translate notes.md --lang Chinese

# Academic writing conversion
cli-anything-wenbi academic paper.docx

# PPT workflow: extract slides + combine with speech
cli-anything-wenbi ppt lecture.mp4 --lang English

# Batch process a directory
cli-anything-wenbi batch ./media/ --output-dir ./out

# View/set session parameters
cli-anything-wenbi session show
cli-anything-wenbi session set llm ollama/qwen3.5:cloud
cli-anything-wenbi session set lang Japanese
```

## JSON Output

All commands support `--json` for machine-readable output:

```bash
cli-anything-wenbi --json rewrite input.md
```

Returns:
```json
{
  "status": "ok",
  "command": "rewrite",
  "output_file": "/path/to/input_rewritten.md",
  "text_preview": "..."
}
```

## REPL Mode

Run without a subcommand to enter interactive mode:

```bash
cli-anything-wenbi
```

## Session Management

Save and restore session state:

```bash
cli-anything-wenbi session set llm ollama/qwen3.5:cloud
cli-anything-wenbi session set lang Chinese
cli-anything-wenbi session save my-session.json

# Later:
cli-anything-wenbi --session-file my-session.json rewrite input.md
```

## Project Management

Organize input files into a project:

```bash
cli-anything-wenbi project new my-lecture -o ./output
cli-anything-wenbi project add-file lecture1.mp4
cli-anything-wenbi project add-file lecture2.mp4
cli-anything-wenbi project list-files
cli-anything-wenbi project validate
```

## Commands

| Command | Description |
|---------|-------------|
| `rewrite` | Rewrite oral text to written form |
| `translate` | Translate text (DeepL first, LLM fallback) |
| `academic` | Convert to academic writing style |
| `ppt` | Extract slides from video, combine with speech |
| `batch` | Process all media files in a directory |
| `project` | Manage input files and settings |
| `session` | View and manage processing parameters |

## Requirements

- Python 3.10+
- wenbi (for processing operations)
- click >= 8.0

## License

Same as wenbi: Apache-2.0