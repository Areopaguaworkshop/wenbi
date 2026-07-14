"""Unified ASR dispatch.

Three backends behind one entry point (`transcribe_with_engine`):

  - gladia   — cloud, requires GLADIA_API_KEY (default when key present)
  - funasr   — local FunASR (paraformer-zh with cam++ speakers, or SenseVoice)
  - whisper  — local openai-whisper (third fallback)

`auto` picks gladia when a key is available, else funasr; if funasr fails under
`auto`, it falls back to whisper. Segment shape is normalized to
``{"start", "end", "text", "spk"?, ...}`` across all backends.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import torch
import whisper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

WHISPER_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
}

# Hardcoded whisper model for the unified dispatch (the --transcribe-model CLI
# flag is gone). PPT/mutilang still pass their own size; this is the default.
DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# FunASR paraformer-zh (with sentence timestamps + VAD + punc)
FUNASR_ZH_WITH_TS = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FUNASR_VAD_ZH = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FUNASR_PUNC_ZH = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
FUNASR_SPK_ZH = "iic/speech_campplus_sv_zh-cn_16k-common"

FUNASR_ALIASES = {
    "1.7B": FUNASR_ZH_WITH_TS,
    "0.6B": FUNASR_ZH_WITH_TS,
    "paraformer-zh": FUNASR_ZH_WITH_TS,
}


def _is_whisper_model(model_name: str) -> bool:
    return model_name in WHISPER_MODELS


def _normalize_funasr_model(model_name: str) -> str:
    return FUNASR_ALIASES.get(model_name, model_name)


def _normalize_funasr_time(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    # FunASR timestamps are often milliseconds → convert to seconds.
    if numeric > 1000.0:
        return numeric / 1000.0
    return numeric


# ---------------------------------------------------------------------------
# Language heuristic (mirrors bilingual.likely_language_from_text)
# ---------------------------------------------------------------------------


def _likely_language_from_text(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if cjk > latin:
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def transcribe_with_engine(
    audio_path: str,
    asr_provider: str = "auto",  # auto|gladia|funasr|whisper
    language: Optional[str] = None,
    verbose: bool = False,
    enable_speakers: bool = False,
    gladia_key: Optional[str] = None,
    gladia_source_lang: Optional[str] = None,
    gladia_speaker_labels: bool = False,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    save_raw_json: Optional[str] = None,
) -> Dict:
    """Transcribe audio with the unified provider dispatch.

    Returns a dict with ``text``, ``language``, ``segments`` (list of
    ``{"start", "end", "text", "spk"?}``). For gladia, the raw API result is
    also available under the ``"raw"`` key so bilingual callers that previously
    received a ``(segments, raw_result)`` tuple can still get it.
    """
    provider = asr_provider
    if provider == "auto":
        if gladia_key or os.getenv("GLADIA_API_KEY"):
            provider = "gladia"
        else:
            provider = "funasr"

    if provider == "gladia":
        key = gladia_key or os.getenv("GLADIA_API_KEY")
        if not key:
            raise ValueError(
                "Gladia provider requires --gladia-key or GLADIA_API_KEY"
            )
        segments, raw = transcribe_with_gladia(
            audio_path,
            key,
            source_lang=gladia_source_lang or "en",
            interpreter_lang=gladia_source_lang or "zh",
            speaker_labels=gladia_speaker_labels or enable_speakers,
            save_raw_json=save_raw_json,
            verbose=verbose,
        )
        return {
            "text": " ".join(s.get("text", "") for s in segments),
            "language": language or "unknown",
            "segments": segments,
            "raw": raw,
            "provider": "gladia",
        }

    if provider == "funasr":
        try:
            return _transcribe_funasr(
                audio_path,
                "paraformer-zh",
                language,
                verbose,
                enable_speakers,
            )
        except Exception as e:
            if asr_provider == "auto":
                logger.warning(f"FunASR failed ({e}), falling back to Whisper")
                return _transcribe_whisper(
                    audio_path, whisper_model, language, verbose
                )
            raise

    if provider == "whisper":
        return _transcribe_whisper(audio_path, whisper_model, language, verbose)

    raise ValueError(f"Unknown ASR provider: {asr_provider}")


# ---------------------------------------------------------------------------
# FunASR paraformer-zh backend (with cam++ speakers when enable_speakers)
# ---------------------------------------------------------------------------


def _transcribe_funasr(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    verbose: bool,
    enable_speakers: bool = False,
) -> Dict:
    try:
        from funasr import AutoModel

        normalized_model = _normalize_funasr_model(model_size)
        if verbose and normalized_model != model_size:
            logger.debug(f"Mapping model '{model_size}' -> '{normalized_model}'")
        if verbose:
            logger.debug(f"Loading FunASR model: {normalized_model}")

        model_kwargs: Dict[str, Any] = {
            "model": normalized_model,
            "vad_model": FUNASR_VAD_ZH,
            "punc_model": FUNASR_PUNC_ZH,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }
        if enable_speakers:
            if verbose:
                logger.debug("Enabling speaker diarization with cam++ model")
            model_kwargs["spk_model"] = FUNASR_SPK_ZH

        model = AutoModel(**model_kwargs)
        if verbose:
            logger.debug("Transcribing with FunASR...")

        result_list = model.generate(
            input=audio_path,
            batch_size_s=300,
            sentence_timestamp=True,
        )

        if isinstance(result_list, list) and len(result_list) > 0:
            result = result_list[0]
        elif isinstance(result_list, dict):
            result = result_list
        else:
            raise ValueError(f"Unexpected FunASR result type: {type(result_list)}")

        text = result.get("text", "") if isinstance(result, dict) else ""
        segments = _extract_funasr_segments(result)
        return {
            "text": text,
            "language": language or "unknown",
            "segments": segments,
            "provider": "funasr",
        }
    except ImportError as e:
        logger.warning(f"FunASR not available ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, DEFAULT_WHISPER_MODEL, language, verbose)
    except Exception as e:
        logger.warning(f"FunASR failed ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, DEFAULT_WHISPER_MODEL, language, verbose)


def _extract_funasr_segments(result: Any) -> List[Dict]:
    if not isinstance(result, dict):
        return []
    sentence_info = result.get("sentence_info")
    if isinstance(sentence_info, list) and sentence_info:
        return [_normalize_funasr_sentence(item) for item in sentence_info]
    timestamps = result.get("timestamp") or result.get("timestamps")
    text = result.get("text", "")
    if isinstance(timestamps, list) and timestamps:
        first = timestamps[0]
        if isinstance(first, dict):
            return [_normalize_funasr_sentence(item) for item in timestamps]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            end_time = _normalize_funasr_time(timestamps[-1][1])
            return [{"start": 0.0, "end": end_time, "text": text}]
    if text:
        return [{"start": 0.0, "end": 0.0, "text": text}]
    return []


def _normalize_funasr_sentence(item: Any) -> Dict:
    if isinstance(item, dict):
        start = _normalize_funasr_time(item.get("start", 0.0))
        end = _normalize_funasr_time(item.get("end", 0.0))
        text = item.get("text", "")
        spk = item.get("spk")
        result = {"start": start, "end": end, "text": text}
        if spk is not None:
            result["spk"] = spk
        return result
    start = _normalize_funasr_time(getattr(item, "start", 0.0))
    end = _normalize_funasr_time(getattr(item, "end", 0.0))
    text = getattr(item, "text", "")
    spk = getattr(item, "spk", None)
    result = {"start": start, "end": end, "text": text}
    if spk is not None:
        result["spk"] = spk
    return result


# ---------------------------------------------------------------------------
# FunASR SenseVoice backend (used by the bilingual interview flows)
# ---------------------------------------------------------------------------


def transcribe_with_sensevoice(
    audio_path: str,
    speaker_labels: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Transcribe with local FunASR SenseVoiceSmall (bilingual interview path)."""
    from funasr import AutoModel

    model_kwargs: Dict[str, Any] = {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "trust_remote_code": True,
        "device": "cpu",
    }
    if speaker_labels:
        model_kwargs["spk_model"] = "cam++"
        model_kwargs["punc_model"] = "ct-punc"

    if verbose:
        logger.debug(f"Loading SenseVoice with options: {model_kwargs}")
    model = AutoModel(**model_kwargs)
    result_list = model.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    result = result_list[0] if isinstance(result_list, list) else result_list
    sentence_info = result.get("sentence_info") if isinstance(result, dict) else None

    rows = sentence_info or []
    if not rows and isinstance(result, dict) and result.get("text"):
        rows = [{"start": 0, "end": 0, "text": result["text"]}]

    segments: List[Dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        lang = row.get("language") or _likely_language_from_text(text)
        speaker = row.get("spk")
        start = float(row.get("start") or 0)
        end = float(row.get("end") or 0)
        if start > 1000:
            start /= 1000
        if end > 1000:
            end /= 1000
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "language": lang,
                "language_confidence": row.get("language_confidence"),
                "speaker": f"Speaker {speaker}" if speaker is not None else None,
                "confidence": row.get("confidence"),
                "provider": "funasr",
            }
        )
    return segments


# ---------------------------------------------------------------------------
# Whisper backend (chunk-normalized, used by bilingual interview fallback)
# ---------------------------------------------------------------------------


def transcribe_with_whisper_chunks(
    audio_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Fallback Whisper transcription with text-based language tagging."""
    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(audio_path, fp16=False, verbose=verbose)
    segments: List[Dict[str, Any]] = []
    for row in result.get("segments") or []:
        text = str(row.get("text") or "").strip()
        segments.append(
            {
                "start": float(row.get("start") or 0),
                "end": float(row.get("end") or 0),
                "text": text,
                "language": _likely_language_from_text(text),
                "language_confidence": None,
                "speaker": None,
                "confidence": None,
                "provider": "whisper",
            }
        )
    return segments


def _transcribe_whisper(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    verbose: bool,
) -> Dict:
    if verbose:
        logger.debug(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size, device="cpu")
    if verbose:
        logger.debug("Transcribing with Whisper...")
    result = model.transcribe(
        audio_path,
        fp16=False,
        verbose=verbose,
        language=language,
    )
    return {
        "text": result.get("text", ""),
        "language": result.get("language", language or "unknown"),
        "segments": result.get("segments", []),
        "provider": "whisper",
    }


# ---------------------------------------------------------------------------
# Gladia backend (moved from bilingual.py)
# ---------------------------------------------------------------------------


def gladia_upload_mime_type(audio_path: str) -> str:
    extension = os.path.splitext(audio_path)[1].lower()
    if extension == ".m4a":
        return "audio/m4a"
    if extension == ".mp3":
        return "audio/mpeg"
    if extension == ".ogg":
        return "application/ogg"
    if extension == ".opus":
        return "audio/opus"
    if extension == ".flac":
        return "audio/flac"
    return "audio/wav"


def prepare_gladia_upload_audio(audio_path: str, verbose: bool = False) -> str:
    """Compress large WAV uploads to m4a to avoid fragile huge multipart uploads."""
    if not audio_path.lower().endswith(".wav"):
        return audio_path
    threshold_mb = int(os.getenv("WENBI_GLADIA_COMPRESS_UPLOAD_MB", "50"))
    threshold_bytes = threshold_mb * 1024 * 1024
    if os.path.getsize(audio_path) < threshold_bytes:
        return audio_path
    output_path = f"{os.path.splitext(audio_path)[0]}_gladia_upload.m4a"
    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(audio_path)
        and os.path.getsize(output_path) > 0
    ):
        if verbose:
            logger.debug("Using existing compressed Gladia upload: %s", output_path)
        return output_path
    if verbose:
        logger.debug("Compressing Gladia upload to m4a: %s", output_path)
    cmd = [
        "ffmpeg", "-y", "-i", audio_path, "-vn", "-acodec", "aac",
        "-b:a", "64k", "-ar", "16000", "-ac", "1", output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_output = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "ffmpeg failed while compressing Gladia upload: "
            f"{stderr_output[-500:]}"
        )
    return output_path


def normalize_gladia_utterances(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Gladia pre-recorded result into normalized segments."""
    transcription = (response.get("result") or {}).get("transcription") or {}
    segments: list[dict[str, Any]] = []
    for utterance in transcription.get("utterances") or []:
        lang = utterance.get("language") or _likely_language_from_text(
            utterance.get("text") or ""
        )
        speaker = utterance.get("speaker")
        segments.append(
            {
                "start": float(utterance.get("start") or 0),
                "end": float(utterance.get("end") or 0),
                "text": str(utterance.get("text") or "").strip(),
                "language": lang,
                "language_confidence": utterance.get("language_confidence"),
                "speaker": f"Speaker {speaker}" if speaker is not None else None,
                "confidence": utterance.get("confidence"),
                "provider": "gladia",
                "words": utterance.get("words") or [],
            }
        )
    return segments


def transcribe_with_gladia(
    audio_path: str,
    api_key: str,
    source_lang: str = "en",
    interpreter_lang: str = "zh",
    speaker_labels: bool = True,
    save_raw_json: str | None = None,
    timeout_seconds: int = 1800,
    poll_interval: float = 5.0,
    upload_timeout_seconds: int | None = None,
    upload_retries: int | None = None,
    verbose: bool = False,
    code_switching: bool = True,
    speaker_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transcribe with Gladia pre-recorded.

    When code_switching=True, enables bilingual code-switching mode (en-zh).
    When code_switching=False, sends a single-language config for mono-lingual
    multi-speaker audio.
    """
    import requests

    def positive_int_from_env(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using %d", name, raw_value, default
            )
            return default
        if value <= 0:
            logger.warning(
                "Ignoring non-positive %s=%r; using %d", name, raw_value, default
            )
            return default
        return value

    def raise_for_gladia_error(response: requests.Response) -> None:
        if response.ok:
            return
        body = response.text[:1000]
        raise RuntimeError(
            f"Gladia API error {response.status_code} for {response.url}: {body}"
        )

    headers = {"x-gladia-key": api_key}
    if verbose:
        logger.debug("Uploading audio to Gladia")
    upload_audio_path = prepare_gladia_upload_audio(audio_path, verbose=verbose)
    upload_mime_type = gladia_upload_mime_type(upload_audio_path)
    effective_upload_timeout = upload_timeout_seconds or positive_int_from_env(
        "WENBI_GLADIA_UPLOAD_TIMEOUT_SECONDS", 900
    )
    effective_upload_retries = upload_retries or positive_int_from_env(
        "WENBI_GLADIA_UPLOAD_RETRIES", 3
    )
    upload_response: requests.Response | None = None
    for attempt in range(1, effective_upload_retries + 1):
        try:
            with open(upload_audio_path, "rb") as audio_file:
                upload_response = requests.post(
                    "https://api.gladia.io/v2/upload",
                    headers=headers,
                    files={
                        "audio": (
                            os.path.basename(upload_audio_path),
                            audio_file,
                            upload_mime_type,
                        )
                    },
                    timeout=effective_upload_timeout,
                )
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            if attempt >= effective_upload_retries:
                raise RuntimeError(
                    "Gladia upload failed after "
                    f"{effective_upload_retries} attempts: {exc}"
                ) from exc
            if verbose:
                logger.warning(
                    "Gladia upload attempt %d/%d failed: %s",
                    attempt, effective_upload_retries, exc,
                )
            time.sleep(min(2 ** (attempt - 1), 10))
    if upload_response is None:
        raise RuntimeError("Gladia upload did not return a response")
    raise_for_gladia_error(upload_response)
    audio_url = upload_response.json()["audio_url"]

    if code_switching:
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "language_config": {
                "languages": [source_lang, interpreter_lang],
                "code_switching": True,
            },
            "sentences": True,
        }
    else:
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "language_config": {"languages": [source_lang]},
            "sentences": True,
        }
    if speaker_labels:
        payload["diarization"] = True
        if speaker_count:
            payload["diarization_config"] = {
                "min_speakers": speaker_count,
                "max_speakers": speaker_count,
            }
        else:
            payload["diarization_config"] = {"min_speakers": 1, "max_speakers": 6}

    if verbose:
        logger.debug("Creating Gladia transcription job")
    job_response = requests.post(
        "https://api.gladia.io/v2/pre-recorded",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    raise_for_gladia_error(job_response)
    job = job_response.json()
    result_url = job.get("result_url") or f"https://api.gladia.io/v2/pre-recorded/{job['id']}"

    deadline = time.time() + timeout_seconds
    result: dict[str, Any] = {}
    while time.time() < deadline:
        poll_response = requests.get(result_url, headers=headers, timeout=60)
        raise_for_gladia_error(poll_response)
        result = poll_response.json()
        status = result.get("status")
        if verbose:
            logger.debug(f"Gladia job status: {status}")
        if status == "done":
            break
        if status in {"error", "failed"}:
            raise RuntimeError(f"Gladia transcription failed: {result}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError("Timed out waiting for Gladia transcription")

    if save_raw_json:
        with open(save_raw_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return normalize_gladia_utterances(result), result


# ---------------------------------------------------------------------------
# Smoke check — provider routing logic without calling network models.
# Run: python -m wenbi.asr
# ---------------------------------------------------------------------------


def _smoke_check() -> None:
    """Assert the provider routing logic. Mocks the backend functions."""
    import sys
    import unittest.mock as mock

    # When run as `python -m wenbi.asr`, __name__ == "__main__" and the
    # module's globals live under sys.modules["__main__"]. Patch the running
    # module regardless of how it was invoked.
    _mod = sys.modules[__name__]
    _patch_target = _mod.__name__  # "wenbi.asr" or "__main__"

    calls: dict[str, dict] = {}

    def fake_gladia(audio_path, key, **kw):
        calls["gladia"] = {"key": key, **kw}
        return [{"start": 0.0, "end": 1.0, "text": "hi"}], {"raw": "gladia"}

    def fake_funasr(audio_path, model_size, language, verbose, enable_speakers=False):
        calls["funasr"] = {"model": model_size}
        return {"text": "hi", "language": "zh", "segments": [], "provider": "funasr"}

    def fake_whisper(audio_path, model_size, language, verbose):
        calls["whisper"] = {"model": model_size}
        return {"text": "hi", "language": "en", "segments": [], "provider": "whisper"}

    # Force GLADIA_API_KEY to be absent for the routing tests (a .env file may
    # have loaded one at import time).
    real_getenv = os.getenv

    def no_gladia_key(name, *a, **kw):
        if name == "GLADIA_API_KEY":
            return None
        return real_getenv(name, *a, **kw)

    with mock.patch(f"{_patch_target}.os.getenv", side_effect=no_gladia_key):
        # 1. auto + key → gladia
        with mock.patch(f"{_patch_target}.transcribe_with_gladia", side_effect=fake_gladia):
            transcribe_with_engine("x.wav", asr_provider="auto", gladia_key="K")
            assert calls.get("gladia"), "auto+key should route to gladia"
            assert calls["gladia"]["key"] == "K"
        calls.clear()

        # 2. auto, no key → funasr
        with mock.patch(f"{_patch_target}._transcribe_funasr", side_effect=fake_funasr):
            transcribe_with_engine("x.wav", asr_provider="auto")
            assert calls.get("funasr"), "auto-no-key should route to funasr"
        calls.clear()

        # 3. funasr fails under auto → whisper fallback
        def funasr_fail(*a, **kw):
            raise RuntimeError("funasr boom")
        with mock.patch(f"{_patch_target}._transcribe_funasr", side_effect=funasr_fail), \
             mock.patch(f"{_patch_target}._transcribe_whisper", side_effect=fake_whisper):
            transcribe_with_engine("x.wav", asr_provider="auto")
            assert calls.get("whisper"), "funasr-fail+auto should fall back to whisper"
        calls.clear()

        # 4. funasr fails under explicit funasr → re-raises (no fallback)
        with mock.patch(f"{_patch_target}._transcribe_funasr", side_effect=funasr_fail):
            try:
                transcribe_with_engine("x.wav", asr_provider="funasr")
                assert False, "explicit funasr should re-raise on failure"
            except RuntimeError:
                pass
        calls.clear()

        # 5. explicit gladia without key → ValueError
        try:
            transcribe_with_engine("x.wav", asr_provider="gladia")
            assert False, "gladia without key should raise ValueError"
        except ValueError:
            pass

        # 6. explicit whisper → whisper
        with mock.patch(f"{_patch_target}._transcribe_whisper", side_effect=fake_whisper):
            transcribe_with_engine("x.wav", asr_provider="whisper")
            assert calls.get("whisper"), "explicit whisper should route to whisper"

    print("asr dispatch smoke check: OK")


if __name__ == "__main__":
    _smoke_check()