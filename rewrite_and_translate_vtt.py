"""One-off script: parse existing _en.vtt → topic group → rewrite → translate → save outputs.

Usage:
    python rewrite_and_translate_vtt.py Maximos3-unseen-warfare-20260520_en.vtt
"""

import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wenbi.bilingual import (
    group_into_topics,
    rewrite_english,
    translate_chunks,
    write_rewritten_markdown,
    write_bilingual_markdown,
)


def parse_vtt_segments(vtt_path: str) -> list[dict]:
    """Parse a WebVTT file into segments with start, end, text, speaker."""
    segments = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        # Find timestamp line
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", lines[0]
        )
        if not ts_match:
            continue
        start_str, end_str = ts_match.group(1), ts_match.group(2)
        start = parse_vtt_ts(start_str)
        end = parse_vtt_ts(end_str)
        # Text lines (may include <v Speaker N> tags)
        text_lines = lines[1:]
        text = " ".join(l.strip() for l in text_lines).strip()
        speaker = None
        speaker_match = re.match(r"<v\s+(.+?)>", text)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            text = re.sub(r"<v\s+.+?>", "", text).strip()
        text = text.replace("</v>", "").strip()
        if text:
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "language": "en",
                "speaker": speaker,
            })
    return segments


def parse_vtt_ts(ts: str) -> float:
    """Convert VTT timestamp to seconds."""
    parts = ts.replace(",", ".").split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_en.vtt>")
        sys.exit(1)

    vtt_path = sys.argv[1]
    if not os.path.exists(vtt_path):
        print(f"File not found: {vtt_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(vtt_path))[0]
    out_dir = os.path.dirname(vtt_path) or "."
    llm = sys.argv[2] if len(sys.argv) > 2 else "ollama/qwen3.5:cloud"

    print(f"1. Parsing {vtt_path} ...")
    segments = parse_vtt_segments(vtt_path)
    print(f"   Parsed {len(segments)} segments")

    print(f"2. Grouping {len(segments)} segments into topic paragraphs (via {llm}) ...")
    topic_paragraphs = group_into_topics(segments, llm=llm, verbose=True)
    print(f"   Grouped into {len(topic_paragraphs)} topic paragraphs")

    print(f"3. Rewriting English (remove oral fillers, conservative) ...")
    rewritten_paragraphs = rewrite_english(topic_paragraphs, llm=llm, verbose=True)
    print(f"   Rewritten {len(rewritten_paragraphs)} paragraphs")

    # Save rewritten markdown
    rewritten_path = os.path.join(out_dir, f"{base_name}_rewritten.md")
    write_rewritten_markdown(rewritten_paragraphs, rewritten_path)
    print(f"   Saved: {rewritten_path}")

    print(f"4. Translating to Chinese (DeepL → Ollama fallback) ...")
    translations = translate_chunks(rewritten_paragraphs, verbose=True)
    print(f"   Translated {len(translations)} paragraphs")

    # Save bilingual markdown
    bilingual_path = os.path.join(out_dir, f"{base_name}_zh.md")
    write_bilingual_markdown(rewritten_paragraphs, translations, bilingual_path)
    print(f"   Saved: {bilingual_path}")

    print("\nDone!")
    print(f"  Rewritten EN:   {rewritten_path}")
    print(f"  Bilingual ZH:   {bilingual_path}")


if __name__ == "__main__":
    main()