"""English-to-Chinese bilingual audio workflow.

This module keeps provider-specific ASR code behind a small normalized segment
schema so the CLI, filtering, formatting, and translation steps can stay stable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

from wenbi.utils import download_audio, extract_audio_segment, parse_timestamp


logger = logging.getLogger(__name__)


@dataclass
class EnZhResult:
    """Output paths produced by the en-zh workflow."""

    english_vtt: str
    english_md: str
    bilingual_md: str
    diagnostics_json: str | None
    provider: str
    kept_segments: int
    dropped_segments: int


def seconds_to_vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp."""
    seconds = max(float(seconds or 0), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def seconds_to_display_time(seconds: float) -> str:
    """Format seconds as a human-readable timestamp without milliseconds."""
    seconds = max(float(seconds or 0), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def likely_language_from_text(text: str) -> str:
    """Best-effort text language classifier for EN/ZH fallback paths."""
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin_count = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if cjk_count > latin_count:
        return "zh"
    if latin_count > 0:
        return "en"
    return "unknown"


def filter_source_segments(
    segments: Iterable[dict[str, Any]],
    source_lang: str = "en",
    min_language_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split normalized ASR segments into kept source-language and dropped segments."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    source_lang = source_lang.lower()

    for segment in segments:
        language = str(segment.get("language") or "").lower()
        confidence = segment.get("language_confidence")
        if confidence is None:
            confidence = 1.0 if language else 0.0

        if language == source_lang and float(confidence) >= min_language_confidence:
            kept.append(segment)
        else:
            dropped.append(segment)

    return kept, dropped


def merge_adjacent_segments(
    segments: list[dict[str, Any]],
    max_gap: float = 1.0,
    max_chars: int = 900,
) -> list[dict[str, Any]]:
    """Merge adjacent same-speaker/source-language segments for readable output."""
    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("start") or 0)):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        if not merged:
            merged.append({**segment, "text": text})
            continue

        previous = merged[-1]
        same_speaker = (previous.get("speaker") or "") == (segment.get("speaker") or "")
        same_language = (previous.get("language") or "") == (segment.get("language") or "")
        gap = float(segment.get("start") or 0) - float(previous.get("end") or 0)
        combined_len = len(str(previous.get("text") or "")) + len(text)

        if same_speaker and same_language and gap <= max_gap and combined_len <= max_chars:
            previous["end"] = segment.get("end", previous.get("end"))
            previous["text"] = f"{previous.get('text', '').rstrip()} {text}".strip()
            previous_conf = previous.get("confidence")
            segment_conf = segment.get("confidence")
            if previous_conf is not None and segment_conf is not None:
                previous["confidence"] = min(float(previous_conf), float(segment_conf))
        else:
            merged.append({**segment, "text": text})

    return merged


def write_vtt(segments: list[dict[str, Any]], output_path: str) -> None:
    """Write normalized segments to WebVTT."""
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = seconds_to_vtt_time(float(segment.get("start") or 0))
        end = seconds_to_vtt_time(float(segment.get("end") or 0))
        text = str(segment.get("text") or "").strip()
        speaker = segment.get("speaker")
        lines.append(f"{start} --> {end}")
        if speaker:
            lines.append(f"<v {speaker}>{text}</v>")
        else:
            lines.append(text)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_english_markdown(segments: list[dict[str, Any]], output_path: str) -> None:
    """Write English-only transcript markdown."""
    parts: list[str] = []
    for segment in segments:
        start = seconds_to_display_time(float(segment.get("start") or 0))
        end = seconds_to_display_time(float(segment.get("end") or 0))
        speaker = segment.get("speaker")
        header = f"### **{start} - {end}**"
        if speaker:
            header += f" · {speaker}"
        parts.append(f"{header}\n\n{str(segment.get('text') or '').strip()}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))


def write_bilingual_markdown(
    segments: list[dict[str, Any]], translations: list[str], output_path: str
) -> None:
    """Write side-by-side English/Chinese markdown."""
    parts: list[str] = []
    for segment, translation in zip(segments, translations):
        start = seconds_to_display_time(float(segment.get("start") or 0))
        end = seconds_to_display_time(float(segment.get("end") or 0))
        speaker = segment.get("speaker")
        header = f"### **{start} - {end}**"
        if speaker:
            header += f" · {speaker}"
        english = str(segment.get("text") or "").strip()
        parts.append(
            f"{header}\n\n"
            f"**[English]**\n{english}\n\n"
            f"**[中文]**\n{translation.strip()}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(parts))


def translate_chunks(
    chunks: list[str],
    target_language: str = "Chinese",
    llm: str = "ollama/qwen3.5:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    use_deepl: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Translate chunks using DeepL first, with the existing LLM stack as fallback."""
    deepl_translator = None
    deepl_available = False
    if use_deepl:
        try:
            from wenbi.llm.deepl import configure_deepl, is_deepl_available

            if is_deepl_available(deepl_key, verbose=verbose):
                deepl_translator = configure_deepl(deepl_key, verbose=verbose)
                deepl_available = True
        except Exception as e:
            if verbose:
                logger.debug(f"DeepL initialization failed: {e}")

    translate_module = None

    def translate_with_llm(text: str) -> str:
        nonlocal translate_module
        if translate_module is None:
            from wenbi.model import _import_dspy, configure_lm

            configure_lm(
                llm or "ollama/qwen3.5:cloud",
                verbose=verbose,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
            )
            dspy = _import_dspy()

            class TranslateSignature(dspy.Signature):
                """Translate text to the target language while preserving meaning and style."""

                text_to_translate = dspy.InputField(desc="Text content to translate")
                target_language = dspy.InputField(desc="Target language")
                translated_text = dspy.OutputField(desc="Translated text")

            translate_module = dspy.Predict(TranslateSignature)
        return translate_module(
            text_to_translate=text, target_language=target_language
        ).translated_text

    translations: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        translated_text = None
        if deepl_available:
            try:
                from wenbi.llm.deepl import translate_with_deepl

                translated_text = translate_with_deepl(
                    deepl_translator, chunk, target_language, verbose=verbose
                )
                if verbose:
                    logger.debug(f"Translated chunk {index} with DeepL")
            except Exception as e:
                if verbose:
                    logger.debug(f"DeepL failed for chunk {index}: {e}")

        if translated_text is None:
            try:
                translated_text = translate_with_llm(chunk)
                if verbose:
                    logger.debug(f"Translated chunk {index} with LLM fallback")
            except Exception as e:
                logger.warning(f"Failed to translate chunk {index}: {e}")
                translated_text = f"[Translation Error: {chunk}]"

        translations.append(translated_text)

    return translations


def normalize_gladia_utterances(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Gladia pre-recorded result into normalized segments."""
    transcription = (response.get("result") or {}).get("transcription") or {}
    segments: list[dict[str, Any]] = []
    for utterance in transcription.get("utterances") or []:
        language = utterance.get("language") or likely_language_from_text(
            utterance.get("text") or ""
        )
        speaker = utterance.get("speaker")
        segments.append(
            {
                "start": float(utterance.get("start") or 0),
                "end": float(utterance.get("end") or 0),
                "text": str(utterance.get("text") or "").strip(),
                "language": language,
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
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transcribe with Gladia pre-recorded code switching."""
    import requests

    headers = {"x-gladia-key": api_key}
    if verbose:
        logger.debug("Uploading audio to Gladia")
    with open(audio_path, "rb") as audio_file:
        upload_response = requests.post(
            "https://api.gladia.io/v2/upload",
            headers=headers,
            files={"audio": (os.path.basename(audio_path), audio_file)},
            timeout=120,
        )
    upload_response.raise_for_status()
    audio_url = upload_response.json()["audio_url"]

    payload: dict[str, Any] = {
        "audio_url": audio_url,
        "language_config": {
            "languages": [source_lang, interpreter_lang],
            "code_switching": True,
        },
        "sentences": True,
    }
    if speaker_labels:
        payload["diarization"] = True
        payload["diarization_config"] = {"min_speakers": 1, "max_speakers": 6}

    if verbose:
        logger.debug("Creating Gladia transcription job")
    job_response = requests.post(
        "https://api.gladia.io/v2/pre-recorded",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    job_response.raise_for_status()
    job = job_response.json()
    result_url = job.get("result_url") or f"https://api.gladia.io/v2/pre-recorded/{job['id']}"

    deadline = time.time() + timeout_seconds
    result: dict[str, Any] = {}
    while time.time() < deadline:
        poll_response = requests.get(result_url, headers=headers, timeout=60)
        poll_response.raise_for_status()
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


def transcribe_with_sensevoice(
    audio_path: str,
    speaker_labels: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe with local FunASR SenseVoiceSmall."""
    from funasr import AutoModel

    model_kwargs: dict[str, Any] = {
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

    segments: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        language = row.get("language") or likely_language_from_text(text)
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
                "language": language,
                "language_confidence": row.get("language_confidence"),
                "speaker": f"Speaker {speaker}" if speaker is not None else None,
                "confidence": row.get("confidence"),
                "provider": "sensevoice",
            }
        )
    return segments


def transcribe_with_whisper_chunks(
    audio_path: str,
    model_size: str = "large-v3-turbo",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fallback Whisper transcription with text-based language tagging."""
    import whisper

    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(audio_path, fp16=False, verbose=verbose)
    segments: list[dict[str, Any]] = []
    for row in result.get("segments") or []:
        text = str(row.get("text") or "").strip()
        segments.append(
            {
                "start": float(row.get("start") or 0),
                "end": float(row.get("end") or 0),
                "text": text,
                "language": likely_language_from_text(text),
                "language_confidence": None,
                "speaker": None,
                "confidence": None,
                "provider": "whisper",
            }
        )
    return segments


def prepare_audio(
    input_path: str,
    output_dir: str,
    start_time: str | None = None,
    end_time: str | None = None,
    output_wav: str = "",
    verbose: bool = False,
) -> str:
    """Convert local or URL input into a WAV file, optionally clipped."""
    timestamp = parse_timestamp(start_time, end_time, verbose=verbose)
    is_url = input_path.startswith(("http://", "https://", "www."))
    if is_url:
        return download_audio(
            input_path,
            output_dir=output_dir,
            timestamp=timestamp,
            output_wav=output_wav,
            verbose=verbose,
        )

    if input_path.lower().endswith(".wav") and not timestamp:
        return input_path
    return extract_audio_segment(
        input_path,
        timestamp=timestamp,
        output_dir=output_dir,
        output_wav=output_wav,
        verbose=verbose,
    )


def process_en_zh(
    input_path: str,
    output_dir: str = "",
    start_time: str = "",
    end_time: str = "",
    asr_provider: str = "auto",
    transcribe_model: str = "large-v3-turbo",
    source_lang: str = "en",
    interpreter_lang: str = "zh",
    target_language: str = "Chinese",
    llm: str = "ollama/qwen3.5:cloud",
    chunk_length: int = 20,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    gladia_key: str | None = None,
    speaker_labels: bool = True,
    save_json: bool = False,
    verbose: bool = False,
) -> EnZhResult:
    """Run the full English-source to Chinese bilingual workflow."""
    if not input_path:
        raise ValueError("input_path is required")
    out_dir = output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if start_time and end_time:
        suffix = f"_{start_time.replace(':', '')}_{end_time.replace(':', '')}"
    else:
        suffix = ""
    output_wav = f"{base_name}{suffix}.wav" if suffix else ""
    audio_path = prepare_audio(
        input_path,
        out_dir,
        start_time=start_time or None,
        end_time=end_time or None,
        output_wav=output_wav,
        verbose=verbose,
    )

    raw_json_path = os.path.join(out_dir, f"{base_name}{suffix}_gladia_raw.json") if save_json else None
    diagnostics_path = os.path.join(out_dir, f"{base_name}{suffix}_en_zh_segments.json") if save_json else None
    provider = asr_provider
    segments: list[dict[str, Any]]

    if asr_provider == "auto":
        if gladia_key or os.getenv("GLADIA_API_KEY"):
            provider = "gladia"
        else:
            provider = "sensevoice"

    if provider == "gladia":
        key = gladia_key or os.getenv("GLADIA_API_KEY")
        if not key:
            raise ValueError("Gladia provider requires --gladia-key or GLADIA_API_KEY")
        segments, _ = transcribe_with_gladia(
            audio_path,
            key,
            source_lang=source_lang,
            interpreter_lang=interpreter_lang,
            speaker_labels=speaker_labels,
            save_raw_json=raw_json_path,
            verbose=verbose,
        )
    elif provider == "sensevoice":
        try:
            segments = transcribe_with_sensevoice(
                audio_path, speaker_labels=speaker_labels, verbose=verbose
            )
        except Exception as e:
            if asr_provider == "auto":
                logger.warning(f"SenseVoice failed ({e}), falling back to Whisper")
                provider = "whisper"
                segments = transcribe_with_whisper_chunks(
                    audio_path, model_size=transcribe_model, verbose=verbose
                )
            else:
                raise
    elif provider == "whisper":
        segments = transcribe_with_whisper_chunks(
            audio_path, model_size=transcribe_model, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown ASR provider: {asr_provider}")

    # ASR is run on the clipped WAV, so shift timestamps back to the original
    # media timeline when --start-time/--end-time were used.
    timestamp = parse_timestamp(start_time or None, end_time or None)
    offset = float((timestamp or {}).get("start") or 0)
    if offset:
        for segment in segments:
            segment["start"] = float(segment.get("start") or 0) + offset
            segment["end"] = float(segment.get("end") or 0) + offset

    kept, dropped = filter_source_segments(segments, source_lang=source_lang)
    merged = merge_adjacent_segments(kept)
    translations = translate_chunks(
        [str(segment.get("text") or "") for segment in merged],
        target_language=target_language,
        llm=llm or "ollama/qwen3.5:cloud",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        deepl_key=deepl_key,
        verbose=verbose,
    )

    english_vtt = os.path.join(out_dir, f"{base_name}{suffix}_en.vtt")
    english_md = os.path.join(out_dir, f"{base_name}{suffix}_en.md")
    bilingual_md = os.path.join(out_dir, f"{base_name}{suffix}_en_zh.md")
    write_vtt(merged, english_vtt)
    write_english_markdown(merged, english_md)
    write_bilingual_markdown(merged, translations, bilingual_md)

    if diagnostics_path:
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": provider,
                    "source_lang": source_lang,
                    "interpreter_lang": interpreter_lang,
                    "audio_path": audio_path,
                    "kept_segments": merged,
                    "dropped_segments": dropped,
                    "raw_segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return EnZhResult(
        english_vtt=english_vtt,
        english_md=english_md,
        bilingual_md=bilingual_md,
        diagnostics_json=diagnostics_path,
        provider=provider,
        kept_segments=len(merged),
        dropped_segments=len(dropped),
    )
