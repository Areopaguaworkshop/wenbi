#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${DEEPL_API_KEY:-}" && -f "$HOME/project/videoedit/.env" ]]; then
  DEEPL_API_KEY="$(
    grep '^DEEPL_API_KEY=' "$HOME/project/videoedit/.env" \
      | head -n 1 \
      | cut -d= -f2- \
      | tr -d '"'
  )"
  export DEEPL_API_KEY
fi

export RAW_JSON="${RAW_JSON:-/tmp/wenbi-origen-prayer/watch?v=Xek64YJAZBY_gladia_raw.json}"
export OUT_DIR="${OUT_DIR:-/tmp/wenbi-origen-prayer}"
export BASE_NAME="${BASE_NAME:-watch?v=Xek64YJAZBY}"
export TARGET_LANGUAGE="${TARGET_LANGUAGE:-Chinese}"
export LLM="${LLM:-ollama/qwen3.5:cloud}"
export USE_DEEPL="${USE_DEEPL:-1}"

.venv/bin/python - <<'PY'
import json
import os

from wenbi.bilingual import (
    group_into_topics,
    merge_adjacent_segments,
    normalize_gladia_utterances,
    rewrite_english,
    translate_chunks,
    write_bilingual_markdown,
    write_rewritten_markdown,
)


raw_json = os.environ["RAW_JSON"]
out_dir = os.environ["OUT_DIR"]
base_name = os.environ["BASE_NAME"]
llm = os.environ["LLM"]
target_language = os.environ.get("TARGET_LANGUAGE", "Chinese")
deepl_key = os.environ.get("DEEPL_API_KEY")
use_deepl = os.environ.get("USE_DEEPL", "1").lower() not in {"0", "false", "no"}

os.makedirs(out_dir, exist_ok=True)

with open(raw_json, "r", encoding="utf-8") as f:
    raw = json.load(f)

segments = normalize_gladia_utterances(raw)
for segment in segments:
    if not segment.get("language"):
        segment["language"] = "en"

merged = merge_adjacent_segments(segments)

diagnostics_path = os.path.join(out_dir, f"{base_name}_speaker_segments.json")
with open(diagnostics_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

topic_paragraphs = group_into_topics(
    merged,
    llm=llm,
    max_tokens=64000,
    timeout=3600,
    temperature=0.1,
    verbose=True,
)

rewritten = rewrite_english(
    topic_paragraphs,
    llm=llm,
    max_tokens=64000,
    timeout=3600,
    temperature=0.1,
    verbose=True,
)

rewritten_path = os.path.join(out_dir, f"{base_name}_rewritten.md")
write_rewritten_markdown(rewritten, rewritten_path)

translations = translate_chunks(
    rewritten,
    target_language=target_language,
    llm=llm,
    max_tokens=64000,
    timeout=3600,
    temperature=0.1,
    deepl_key=deepl_key,
    use_deepl=use_deepl,
    verbose=True,
)

bilingual_path = os.path.join(out_dir, f"{base_name}_en_zh.md")
write_bilingual_markdown(rewritten, translations, bilingual_path)

print("Diagnostics JSON:", diagnostics_path)
print("English Rewritten Markdown:", rewritten_path)
print("Bilingual Markdown:", bilingual_path)
PY
