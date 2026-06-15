"""English-to-Chinese bilingual audio workflow.

This module keeps provider-specific ASR code behind a small normalized segment
schema so the CLI, filtering, formatting, and translation steps can stay stable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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
    gladia_vtt: str | None = None
    english_rewritten_md: str | None = None


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


def write_rewritten_markdown(paragraphs: list[str], output_path: str) -> None:
    """Write rewritten English paragraphs as clean prose markdown."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(p.strip() for p in paragraphs if p.strip()))


def write_bilingual_markdown(
    paragraphs: list[str], translations: list[str], output_path: str
) -> None:
    """Write side-by-side English/Chinese markdown as clean prose.

    No timestamps, no speaker labels. Topic paragraphs separated by ---.
    """
    parts: list[str] = []
    for english, translation in zip(paragraphs, translations):
        english_text = english.strip() if isinstance(english, str) else str(english).strip()
        parts.append(
            f"**[English]**\n{english_text}\n\n"
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


def group_into_topics(
    segments: list[dict[str, Any]],
    llm: str = "ollama/qwen3.5:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Group merged transcript segments into topic-coherent paragraphs via LLM.

    Preserves 100% of original text — only groups adjacent segments by topic
    and marks paragraph breaks with '---'.
    """
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )
    dspy = _import_dspy()

    class TopicGroupSignature(dspy.Signature):
        """Group transcript segments into topic-coherent paragraphs.

        Preserve ALL original text exactly. Only group adjacent segments by
        topic coherence. Mark paragraph breaks between topic groups with '---'
        on its own line. Do NOT add, remove, or rephrase any text.
        """

        transcript = dspy.InputField(
            desc="Segmented transcript with numbered segment markers"
        )
        grouped = dspy.OutputField(
            desc="Topic-grouped paragraphs using same text, paragraph breaks marked with ---"
        )

    # Build the transcript input with segment markers
    indexed_lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        indexed_lines.append(f"[{i}] {str(seg.get('text') or '').strip()}")
    transcript_text = "\n".join(indexed_lines)

    if verbose:
        logger.debug("Grouping %d segments into topic paragraphs", len(segments))

    module = dspy.Predict(TopicGroupSignature)
    result = module(transcript=transcript_text)

    if verbose:
        logger.debug("Topic grouping complete")

    # Split on '---' to get topic paragraphs
    paragraphs = [p.strip() for p in result.grouped.split("---") if p.strip()]
    return paragraphs


def rewrite_english(
    paragraphs: list[str],
    llm: str = "ollama/qwen3.5:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Remove oral fillers and fix grammar conservatively via LLM.

    Removes: um, uh, you know, like (filler), false starts, obvious fragments.
    Fixes: grammar, unify broken sentences.
    Keeps 97% original wording — does NOT restructure or rephrase.
    """
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )
    dspy = _import_dspy()

    class CleanEnglishSignature(dspy.Signature):
        """Remove oral fillers and fix grammar. Keep 97% original wording.

        Remove: um, uh, you know, like (as filler), false starts, obvious fragments.
        Fix: grammar, unify broken sentences.
        Do NOT restructure or rephrase. Preserve the original voice and style.
        """

        english_text = dspy.InputField(desc="Raw English transcript paragraph")
        cleaned = dspy.OutputField(
            desc="Cleaned English preserving 97% original wording"
        )

    module = dspy.Predict(CleanEnglishSignature)

    rewritten: list[str] = []
    for i, paragraph in enumerate(paragraphs, 1):
        if verbose:
            logger.debug("Rewriting paragraph %d/%d", i, len(paragraphs))
        try:
            result = module(english_text=paragraph)
            rewritten.append(result.cleaned.strip())
        except Exception as e:
            logger.warning("Rewrite failed for paragraph %d: %s", i, e)
            rewritten.append(paragraph)

    return rewritten


def format_speaker_turns_for_rewrite(
    segments: list[dict[str, Any]],
    max_chars: int = 6000,
) -> list[str]:
    """Create speaker-labeled chunks suitable for interview rewriting."""
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        speaker = str(segment.get("speaker") or "Speaker").strip()
        line = f"【{speaker}】{text}"
        if current_lines and current_len + len(line) > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def rewrite_chinese_interview(
    speaker_chunks: list[str],
    llm: str = "ollama/qwen3.5:cloud",
    expected_speakers: int = 2,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Polish a Chinese interview transcript while preserving speaker turns."""
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm or "ollama/qwen3.5:cloud",
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )
    dspy = _import_dspy()

    class ChineseInterviewRewriteSignature(dspy.Signature):
        """
        Rewrite a Chinese interview transcript into polished written Chinese.

        Follow these rules strictly:
        1. Preserve speaker separation and turn order. Keep every paragraph prefixed with a speaker label such as 【主持人】, 【受访者】, or the original 【Speaker 0】/【Speaker 1】 label when roles are unclear.
        2. Assume the interview has {expected_speakers} speakers unless the transcript clearly shows otherwise.
        3. Remove Chinese oral fillers and repetitions such as 嗯、啊、这个、那个、就是说、然后、对吧、是吧、什么的、之类的, while preserving meaning.
        4. Convert fragmented spoken Chinese into fluent written interview prose. Correct punctuation, grammar, names, terms, and obvious ASR mistakes only when context makes the correction clear.
        5. Do not add arguments, facts, citations, or examples that are not present in the transcript.
        6. If a speaker role, proper noun, technical term, timeline, or ambiguous ASR phrase needs human confirmation, append a short section titled ## 需要确认的问题 with concrete questions. Omit that section if nothing needs confirmation.
        """

        speaker_text = dspy.InputField(desc="Chinese speaker-labeled interview transcript")
        expected_speakers = dspy.InputField(desc="Expected number of speakers")
        written_text = dspy.OutputField(desc="Polished Chinese interview transcript with speaker labels")

    module = dspy.Predict(ChineseInterviewRewriteSignature)

    rewritten: list[str] = []
    for i, chunk in enumerate(speaker_chunks, 1):
        if verbose:
            logger.debug("Rewriting Chinese interview chunk %d/%d", i, len(speaker_chunks))
        try:
            result = module(speaker_text=chunk, expected_speakers=str(expected_speakers))
            rewritten.append(result.written_text.strip())
        except Exception as e:
            logger.warning("Chinese interview rewrite failed for chunk %d: %s", i, e)
            rewritten.append(chunk)

    return rewritten


def rewrite_english_interview(
    speaker_chunks: list[str],
    llm: str = "ollama/qwen3.5:cloud",
    expected_speakers: int = 2,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Polish an English interview transcript while preserving speaker turns."""
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm or "ollama/qwen3.5:cloud",
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )
    dspy = _import_dspy()

    class EnglishInterviewRewriteSignature(dspy.Signature):
        """
        Rewrite an English interview transcript into polished written English.

        Follow these rules strictly:
        1. Preserve speaker separation and turn order. Keep every paragraph prefixed with a speaker label such as [Host], [Guest], or the original [Speaker 0]/[Speaker 1] label when roles are unclear.
        2. Assume the interview has {expected_speakers} speakers unless the transcript clearly shows otherwise.
        3. Remove oral fillers and repetitions such as um, uh, you know, like, I mean, sort of, kind of, right, okay, false starts, and repeated fragments, while preserving meaning.
        4. Convert fragmented spoken English into fluent written interview prose. Correct punctuation, grammar, names, terms, and obvious ASR mistakes only when context makes the correction clear.
        5. Do not add arguments, facts, citations, or examples that are not present in the transcript.
        6. If a speaker role, proper noun, technical term, timeline, or ambiguous ASR phrase needs human confirmation, append a short section titled ## Questions for Clarification with concrete questions. Omit that section if nothing needs confirmation.
        """

        speaker_text = dspy.InputField(desc="English speaker-labeled interview transcript")
        expected_speakers = dspy.InputField(desc="Expected number of speakers")
        written_text = dspy.OutputField(desc="Polished English interview transcript with speaker labels")

    module = dspy.Predict(EnglishInterviewRewriteSignature)

    rewritten: list[str] = []
    for i, chunk in enumerate(speaker_chunks, 1):
        if verbose:
            logger.debug("Rewriting English interview chunk %d/%d", i, len(speaker_chunks))
        try:
            result = module(speaker_text=chunk, expected_speakers=str(expected_speakers))
            rewritten.append(result.written_text.strip())
        except Exception as e:
            logger.warning("English interview rewrite failed for chunk %d: %s", i, e)
            rewritten.append(chunk)

    return rewritten


def write_gladia_vtt(raw_result: dict[str, Any], output_path: str) -> None:
    """Write the raw Gladia API response as a WebVTT file (cue-level, no merging)."""
    segments = normalize_gladia_utterances(raw_result)
    write_vtt(segments, output_path)


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


def gladia_upload_mime_type(audio_path: str) -> str:
    """Return a reasonable upload MIME type for Gladia-supported audio files."""
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
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-acodec",
        "aac",
        "-b:a",
        "64k",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_output = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "ffmpeg failed while compressing Gladia upload: "
            f"{stderr_output[-500:]}"
        )
    return output_path


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

    When code_switching=True, enables bilingual code-switching mode (original
    behaviour for en-zh).  When code_switching=False, sends a single-language
    config which is appropriate for mono-lingual multi-speaker audio.
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
                    attempt,
                    effective_upload_retries,
                    exc,
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
            "language_config": {
                "languages": [source_lang],
            },
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
    asr_provider: str = "gladia",
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
    gladia_raw_result: dict[str, Any] | None = None

    if asr_provider == "auto":
        if gladia_key or os.getenv("GLADIA_API_KEY"):
            provider = "gladia"
        else:
            provider = "sensevoice"

    if provider == "gladia":
        key = gladia_key or os.getenv("GLADIA_API_KEY")
        if not key:
            raise ValueError("Gladia provider requires --gladia-key or GLADIA_API_KEY")
        segments, gladia_raw_result = transcribe_with_gladia(
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

    # --- New pipeline: topic grouping → rewrite → translate ---

    # 1. Save Gladia raw VTT alongside (if using Gladia)
    gladia_vtt_path: str | None = None
    if provider == "gladia" and gladia_raw_result is not None:
        gladia_vtt_path = os.path.join(out_dir, f"{base_name}{suffix}_gladia.vtt")
        write_gladia_vtt(gladia_raw_result, gladia_vtt_path)
        if verbose:
            logger.debug("Saved raw Gladia VTT: %s", gladia_vtt_path)

    # 2. Group merged segments into topic paragraphs via LLM
    topic_paragraphs = group_into_topics(
        merged,
        llm=llm or "ollama/qwen3.5:cloud",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        verbose=verbose,
    )
    if verbose:
        logger.debug("Grouped into %d topic paragraphs", len(topic_paragraphs))

    # 3. Rewrite English: remove oral fillers, conservative cleanup
    rewritten_paragraphs = rewrite_english(
        topic_paragraphs,
        llm=llm or "ollama/qwen3.5:cloud",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        verbose=verbose,
    )

    # 4. Save rewritten English markdown
    english_rewritten_md = os.path.join(out_dir, f"{base_name}{suffix}_en_rewritten.md")
    write_rewritten_markdown(rewritten_paragraphs, english_rewritten_md)

    # 5. Translate rewritten paragraphs
    translations = translate_chunks(
        rewritten_paragraphs,
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
    write_bilingual_markdown(rewritten_paragraphs, translations, bilingual_md)

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
        gladia_vtt=gladia_vtt_path,
        english_rewritten_md=english_rewritten_md,
    )


# ---------------------------------------------------------------------------
# Speaker-aware single-language workflow (e.g. English interview with 2+
# speakers, no interpreter).  Transcribes with diarization, groups into
# topics, rewrites, and optionally translates to a target language.
# ---------------------------------------------------------------------------


@dataclass
class SpeakerResult:
    """Output paths produced by the speaker workflow."""

    transcript_vtt: str
    transcript_md: str
    rewritten_md: str
    bilingual_md: str | None
    diagnostics_json: str | None
    provider: str
    num_speakers: int
    total_segments: int
    gladia_vtt: str | None = None


def process_speaker(
    input_path: str,
    output_dir: str = "",
    start_time: str = "",
    end_time: str = "",
    asr_provider: str = "gladia",
    transcribe_model: str = "large-v3-turbo",
    source_lang: str = "en",
    target_language: str = "Chinese",
    llm: str = "ollama/qwen3.5:cloud",
    chunk_length: int = 20,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    gladia_key: str | None = None,
    speaker_labels: bool = True,
    speaker_count: int | None = None,
    rewrite_mode: str = "english",
    save_json: bool = False,
    verbose: bool = False,
) -> SpeakerResult:
    """Run the full speaker-aware single-language workflow.

    1. Transcribe with diarization (Gladia, SenseVoice, or Whisper).
    2. Merge adjacent same-speaker segments.
    3. Group into topic paragraphs via LLM.
    4. Rewrite (remove oral fillers).
    5. Optionally translate to *target_language*.
    """
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
    diagnostics_path = os.path.join(out_dir, f"{base_name}{suffix}_speaker_segments.json") if save_json else None
    provider = asr_provider
    segments: list[dict[str, Any]]
    gladia_raw_result: dict[str, Any] | None = None

    if asr_provider == "auto":
        if gladia_key or os.getenv("GLADIA_API_KEY"):
            provider = "gladia"
        else:
            provider = "sensevoice"

    if provider == "gladia":
        key = gladia_key or os.getenv("GLADIA_API_KEY")
        if not key:
            raise ValueError("Gladia provider requires --gladia-key or GLADIA_API_KEY")
        segments, gladia_raw_result = transcribe_with_gladia(
            audio_path,
            key,
            source_lang=source_lang,
            interpreter_lang=source_lang,  # same language — no code switching
            speaker_labels=speaker_labels,
            speaker_count=speaker_count,
            save_raw_json=raw_json_path,
            verbose=verbose,
            code_switching=False,
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

    # Shift timestamps back when --start-time/--end-time were used.
    timestamp = parse_timestamp(start_time or None, end_time or None)
    offset = float((timestamp or {}).get("start") or 0)
    if offset:
        for segment in segments:
            segment["start"] = float(segment.get("start") or 0) + offset
            segment["end"] = float(segment.get("end") or 0) + offset

    # Label all segments with the source language (no filtering needed)
    for seg in segments:
        if not seg.get("language"):
            seg["language"] = source_lang

    merged = merge_adjacent_segments(segments)

    # Count distinct speakers
    speakers_seen = {
        seg.get("speaker") for seg in merged if seg.get("speaker")
    }
    num_speakers = len(speakers_seen) or 1

    # --- Topic grouping → rewrite → translate ---

    # 1. Save Gladia raw VTT (if using Gladia)
    gladia_vtt_path: str | None = None
    if provider == "gladia" and gladia_raw_result is not None:
        gladia_vtt_path = os.path.join(out_dir, f"{base_name}{suffix}_gladia.vtt")
        write_gladia_vtt(gladia_raw_result, gladia_vtt_path)
        if verbose:
            logger.debug("Saved raw Gladia VTT: %s", gladia_vtt_path)

    if rewrite_mode == "zh-interview":
        speaker_chunks = format_speaker_turns_for_rewrite(merged)
        if verbose:
            logger.debug("Prepared %d speaker-labeled Chinese chunks", len(speaker_chunks))
        rewritten_paragraphs = rewrite_chinese_interview(
            speaker_chunks,
            llm=llm or "ollama/qwen3.5:cloud",
            expected_speakers=speaker_count or num_speakers or 2,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            verbose=verbose,
        )
    elif rewrite_mode == "en-interview":
        speaker_chunks = format_speaker_turns_for_rewrite(merged)
        if verbose:
            logger.debug("Prepared %d speaker-labeled English chunks", len(speaker_chunks))
        rewritten_paragraphs = rewrite_english_interview(
            speaker_chunks,
            llm=llm or "ollama/qwen3.5:cloud",
            expected_speakers=speaker_count or num_speakers or 2,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            verbose=verbose,
        )
    else:
        # 2. Group merged segments into topic paragraphs via LLM
        topic_paragraphs = group_into_topics(
            merged,
            llm=llm or "ollama/qwen3.5:cloud",
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            verbose=verbose,
        )
        if verbose:
            logger.debug("Grouped into %d topic paragraphs", len(topic_paragraphs))

        # 3. Rewrite English: remove oral fillers, conservative cleanup
        rewritten_paragraphs = rewrite_english(
            topic_paragraphs,
            llm=llm or "ollama/qwen3.5:cloud",
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            verbose=verbose,
        )

    # 4. Save rewritten markdown
    rewritten_md_path = os.path.join(out_dir, f"{base_name}{suffix}_rewritten.md")
    write_rewritten_markdown(rewritten_paragraphs, rewritten_md_path)

    # 5. Translate if target_language differs from source_lang
    bilingual_md_path: str | None = None
    target_lang_lower = (target_language or "").lower()
    source_lower = source_lang.lower()
    # Simple heuristic: skip translation when target matches source
    _skip_translation = (
        target_lang_lower in ("", source_lower)
        or target_lang_lower.startswith(source_lower)
    )
    if not _skip_translation:
        translations = translate_chunks(
            rewritten_paragraphs,
            target_language=target_language,
            llm=llm or "ollama/qwen3.5:cloud",
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            deepl_key=deepl_key,
            verbose=verbose,
        )
        bilingual_md_path = os.path.join(out_dir, f"{base_name}{suffix}_{source_lower}_zh.md")
        write_bilingual_markdown(rewritten_paragraphs, translations, bilingual_md_path)

    # 6. Write transcript VTT + markdown (with speaker labels)
    transcript_vtt = os.path.join(out_dir, f"{base_name}{suffix}_{source_lower}.vtt")
    transcript_md = os.path.join(out_dir, f"{base_name}{suffix}_{source_lower}.md")
    write_vtt(merged, transcript_vtt)
    write_english_markdown(merged, transcript_md)

    if diagnostics_path:
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": provider,
                    "source_lang": source_lang,
                    "speaker_count": speaker_count,
                    "rewrite_mode": rewrite_mode,
                    "audio_path": audio_path,
                    "merged_segments": merged,
                    "raw_segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return SpeakerResult(
        transcript_vtt=transcript_vtt,
        transcript_md=transcript_md,
        rewritten_md=rewritten_md_path,
        bilingual_md=bilingual_md_path,
        diagnostics_json=diagnostics_path,
        provider=provider,
        num_speakers=num_speakers,
        total_segments=len(merged),
        gladia_vtt=gladia_vtt_path,
    )
