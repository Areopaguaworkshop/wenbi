import requests

from wenbi import bilingual


class FakeResponse:
    def __init__(
        self, payload, ok=True, status_code=200, url="https://api.gladia.io"
    ):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.url = url
        self.text = str(payload)

    def json(self):
        return self._payload


def test_transcribe_with_gladia_retries_upload_timeout(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    post_calls = []

    def fake_post(url, **kwargs):
        post_calls.append((url, kwargs["timeout"]))
        if url.endswith("/upload") and len(post_calls) == 1:
            raise requests.exceptions.ConnectionError("write timed out")
        if url.endswith("/upload"):
            return FakeResponse({"audio_url": "https://media.example/audio.wav"})
        return FakeResponse(
            {"id": "job-1", "result_url": "https://job.example/result"}
        )

    def fake_get(url, **kwargs):
        return FakeResponse(
            {
                "status": "done",
                "result": {
                    "transcription": {
                        "utterances": [
                            {
                                "start": 0,
                                "end": 1,
                                "text": "hello",
                                "language": "en",
                                "speaker": 1,
                            }
                        ]
                    }
                },
            }
        )

    monkeypatch.setenv("WENBI_GLADIA_UPLOAD_TIMEOUT_SECONDS", "321")
    monkeypatch.setenv("WENBI_GLADIA_UPLOAD_RETRIES", "2")
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(bilingual.time, "sleep", lambda _seconds: None)

    segments, raw_result = bilingual.transcribe_with_gladia(
        str(audio_path),
        "api-key",
        poll_interval=0,
        code_switching=False,
    )

    assert [call[0] for call in post_calls] == [
        "https://api.gladia.io/v2/upload",
        "https://api.gladia.io/v2/upload",
        "https://api.gladia.io/v2/pre-recorded",
    ]
    assert post_calls[0][1] == 321
    assert segments[0]["text"] == "hello"
    assert raw_result["status"] == "done"


def test_transcribe_with_gladia_compresses_large_wav(tmp_path, monkeypatch):
    audio_path = tmp_path / "large.wav"
    audio_path.write_bytes(b"audio")
    compressed_path = tmp_path / "large_gladia_upload.m4a"

    def fake_getsize(path):
        if str(path).endswith("large.wav"):
            return 100 * 1024 * 1024
        return 123

    def fake_run(cmd, stdout, stderr):
        compressed_path.write_bytes(b"compressed")

        class Result:
            returncode = 0
            stderr = b""

        return Result()

    post_files = []

    def fake_post(url, **kwargs):
        if url.endswith("/upload"):
            audio_file = kwargs["files"]["audio"]
            post_files.append((audio_file[0], audio_file[2]))
            return FakeResponse({"audio_url": "https://media.example/audio.m4a"})
        return FakeResponse({"id": "job-1", "result_url": "https://job.example/result"})

    def fake_get(url, **kwargs):
        return FakeResponse({"status": "done", "result": {"transcription": {}}})

    monkeypatch.setattr(bilingual.os.path, "getsize", fake_getsize)
    monkeypatch.setattr(bilingual.subprocess, "run", fake_run)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    bilingual.transcribe_with_gladia(
        str(audio_path),
        "api-key",
        poll_interval=0,
        code_switching=False,
    )

    assert post_files == [("large_gladia_upload.m4a", "audio/m4a")]
