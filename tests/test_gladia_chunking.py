from wenbi import asr


def test_long_gladia_audio_is_chunked_and_timestamps_restored(monkeypatch):
    monkeypatch.setattr(asr, "prepare_gladia_upload_audio", lambda path, verbose=False: path)
    monkeypatch.setattr(
        asr, "audio_duration_seconds", lambda path: asr.GLADIA_MAX_DURATION_SECONDS + 1
    )
    monkeypatch.setattr(
        asr,
        "create_gladia_chunks",
        lambda path, output_dir, duration, verbose=False: [(0.0, "first.m4a"), (7800.0, "second.m4a")],
    )

    def fake_transcribe(path, **kwargs):
        return (
            [{"start": 1.0, "end": 2.0, "text": path, "words": [{"start": 1.0, "end": 2.0}]}],
            {"status": "done", "result": {"transcription": {"utterances": [{"start": 1.0, "end": 2.0, "text": path}]}}},
        )

    monkeypatch.setattr(asr, "_transcribe_gladia_file", fake_transcribe)

    segments, result = asr.transcribe_with_gladia("lecture.m4a", "api-key")

    assert [segment["start"] for segment in segments] == [1.0, 7801.0]
    assert segments[1]["words"][0]["end"] == 7802.0
    utterances = result["result"]["transcription"]["utterances"]
    assert [utterance["start"] for utterance in utterances] == [1.0, 7801.0]