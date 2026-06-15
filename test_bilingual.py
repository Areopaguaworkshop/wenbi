import json

from wenbi.bilingual import (
    filter_source_segments,
    format_speaker_turns_for_rewrite,
    merge_adjacent_segments,
    normalize_gladia_utterances,
    process_speaker,
    seconds_to_display_time,
    seconds_to_vtt_time,
    write_bilingual_markdown,
    write_english_markdown,
    write_rewritten_markdown,
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


def test_format_speaker_turns_for_rewrite_preserves_labels():
    segments = [
        {"speaker": "Speaker 0", "text": "主持人问题。"},
        {"speaker": "Speaker 1", "text": "受访者回答。"},
    ]

    chunks = format_speaker_turns_for_rewrite(segments)

    assert chunks == ["【Speaker 0】主持人问题。\n【Speaker 1】受访者回答。"]


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
    write_bilingual_markdown(["English source."], ["中文译文。"], str(bilingual_path))

    assert "<v Speaker 1>English source.</v>" in vtt_path.read_text(encoding="utf-8")
    assert "### **00:00:05 - 00:00:10** · Speaker 1" in en_path.read_text(encoding="utf-8")
    bilingual = bilingual_path.read_text(encoding="utf-8")
    assert "**[English]**" in bilingual
    assert "**[中文]**" in bilingual
    assert "中文译文。" in bilingual
    # Verify no timestamps or speaker labels in bilingual output
    assert "### **" not in bilingual
    assert "Speaker" not in bilingual


def test_diagnostics_payload_is_json_serializable():
    payload = {"kept_segments": [{"start": 0, "end": 1, "text": "Hi", "language": "en"}]}
    assert json.loads(json.dumps(payload, ensure_ascii=False))["kept_segments"][0]["text"] == "Hi"


def test_rewritten_markdown(tmp_path):
    paragraphs = ["First paragraph.", "Second paragraph.", "Third one."]
    path = tmp_path / "rewritten.md"
    write_rewritten_markdown(paragraphs, str(path))
    content = path.read_text(encoding="utf-8")
    assert "First paragraph." in content
    assert "Second paragraph." in content
    assert "---" in content


def test_bilingual_markdown_no_timestamps(tmp_path):
    paragraphs = ["Hello world.", "Goodbye."]
    translations = ["你好世界。", "再见。"]
    path = tmp_path / "bilingual.md"
    write_bilingual_markdown(paragraphs, translations, str(path))
    content = path.read_text(encoding="utf-8")
    assert "**[English]**" in content
    assert "**[中文]**" in content
    assert "你好世界。" in content
    # No timestamps or speaker labels
    assert "### **" not in content
    assert "Speaker" not in content
    assert "---" in content


def test_process_speaker_en_interview_uses_speaker_aware_rewrite(tmp_path, monkeypatch):
    audio_path = tmp_path / "interview.wav"
    audio_path.write_bytes(b"audio")
    captured = {}

    def fake_prepare_audio(*args, **kwargs):
        return str(audio_path)

    def fake_transcribe(*args, **kwargs):
        return [
            {
                "start": 0,
                "end": 1,
                "text": "Um welcome to the interview.",
                "language": "en",
                "speaker": "Speaker 0",
            },
            {
                "start": 2,
                "end": 3,
                "text": "Yeah thanks for having me.",
                "language": "en",
                "speaker": "Speaker 1",
            },
        ]

    def fake_rewrite(chunks, **kwargs):
        captured["chunks"] = chunks
        captured["expected_speakers"] = kwargs["expected_speakers"]
        return ["[Speaker 0] Welcome to the interview.\n[Speaker 1] Thank you for having me."]

    monkeypatch.setattr("wenbi.bilingual.prepare_audio", fake_prepare_audio)
    monkeypatch.setattr("wenbi.bilingual.transcribe_with_whisper_chunks", fake_transcribe)
    monkeypatch.setattr("wenbi.bilingual.rewrite_english_interview", fake_rewrite)

    result = process_speaker(
        str(audio_path),
        output_dir=str(tmp_path),
        asr_provider="whisper",
        source_lang="en",
        target_language="English",
        speaker_count=2,
        rewrite_mode="en-interview",
    )

    assert captured["chunks"] == [
        "【Speaker 0】Um welcome to the interview.\n【Speaker 1】Yeah thanks for having me."
    ]
    assert captured["expected_speakers"] == 2
    assert result.rewritten_md.endswith("_rewritten.md")
