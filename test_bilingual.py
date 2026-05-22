import json

from wenbi.bilingual import (
    filter_source_segments,
    merge_adjacent_segments,
    normalize_gladia_utterances,
    seconds_to_display_time,
    seconds_to_vtt_time,
    write_bilingual_markdown,
    write_english_markdown,
    write_vtt,
)


def test_timestamp_formatting():
    assert seconds_to_vtt_time(5.2) == "00:00:05.200"
    assert seconds_to_display_time(3723.8) == "01:02:03"


def test_filter_source_segments_keeps_english_only():
    segments = [
        {"language": "en", "text": "hello", "language_confidence": 0.9},
        {"language": "zh", "text": "你好", "language_confidence": 0.9},
        {"language": "unknown", "text": "...", "language_confidence": 0.0},
    ]

    kept, dropped = filter_source_segments(segments, source_lang="en")

    assert [segment["text"] for segment in kept] == ["hello"]
    assert [segment["text"] for segment in dropped] == ["你好", "..."]


def test_merge_adjacent_segments_same_speaker_language():
    segments = [
        {"start": 0, "end": 2, "text": "First.", "language": "en", "speaker": "Speaker 1"},
        {"start": 2.5, "end": 4, "text": "Second.", "language": "en", "speaker": "Speaker 1"},
        {"start": 10, "end": 12, "text": "Third.", "language": "en", "speaker": "Speaker 2"},
    ]

    merged = merge_adjacent_segments(segments, max_gap=1)

    assert len(merged) == 2
    assert merged[0]["text"] == "First. Second."
    assert merged[0]["end"] == 4
    assert merged[1]["text"] == "Third."


def test_normalize_gladia_utterances():
    response = {
        "result": {
            "transcription": {
                "utterances": [
                    {
                        "start": 1.2,
                        "end": 3.4,
                        "text": "Hello world.",
                        "language": "en",
                        "confidence": 0.98,
                        "speaker": 0,
                        "words": [],
                    }
                ]
            }
        }
    }

    segments = normalize_gladia_utterances(response)

    assert segments == [
        {
            "start": 1.2,
            "end": 3.4,
            "text": "Hello world.",
            "language": "en",
            "language_confidence": None,
            "speaker": "Speaker 0",
            "confidence": 0.98,
            "provider": "gladia",
            "words": [],
        }
    ]


def test_markdown_and_vtt_writers(tmp_path):
    segments = [
        {
            "start": 5,
            "end": 10,
            "text": "English source.",
            "language": "en",
            "speaker": "Speaker 1",
        }
    ]
    vtt_path = tmp_path / "out.vtt"
    en_path = tmp_path / "en.md"
    bilingual_path = tmp_path / "bilingual.md"

    write_vtt(segments, str(vtt_path))
    write_english_markdown(segments, str(en_path))
    write_bilingual_markdown(segments, ["中文译文。"], str(bilingual_path))

    assert "<v Speaker 1>English source.</v>" in vtt_path.read_text(encoding="utf-8")
    assert "### **00:00:05 - 00:00:10** · Speaker 1" in en_path.read_text(encoding="utf-8")
    bilingual = bilingual_path.read_text(encoding="utf-8")
    assert "**[English]**" in bilingual
    assert "**[中文]**" in bilingual
    assert "中文译文。" in bilingual


def test_diagnostics_payload_is_json_serializable():
    payload = {"kept_segments": [{"start": 0, "end": 1, "text": "Hi", "language": "en"}]}
    assert json.loads(json.dumps(payload, ensure_ascii=False))["kept_segments"][0]["text"] == "Hi"
