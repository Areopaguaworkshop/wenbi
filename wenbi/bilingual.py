"""English-to-Chinese bilingual audio workflow.

This module keeps provider-specific ASR code behind a small normalized segment
schema so the CLI, filtering, formatting, and translation steps can stay stable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterable

from wenbi.asr import (
    normalize_gladia_utterances,
    prepare_gladia_upload_audio,
    gladia_upload_mime_type,
    transcribe_with_gladia,
    transcribe_with_sensevoice,
    transcribe_with_whisper_chunks,
)
from wenbi.utils import download_audio, extract_audio_segment, is_text_file, parse_timestamp


logger = logging.getLogger(__name__)


def _load_glossary_text(glossary_file: str | None) -> str:
    """Load glossary as 'english: chinese\\n' lines for the LLM prompt.

    Uses a user-supplied JSON file ({english: chinese}) when provided, else
    falls back to the built-in patristic glossary. Returns "" if unavailable.
    """
    import json as _json

    try:
        if glossary_file:
            with open(glossary_file, encoding="utf-8") as f:
                pairs = _json.load(f)
            return "\n".join(f"{en}: {zh}" for en, zh in pairs.items())
        from wenbi.patristic_glossary import get_glossary_for_dspy

        return get_glossary_for_dspy()
    except Exception as e:
        logger.debug(f"glossary load failed: {e}")
        return ""


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
        translation_text = translation.strip() if isinstance(translation, str) else str(translation or "").strip()
        parts.append(
            f"**[English]**\n{english_text}\n\n"
            f"**[中文]**\n{translation_text}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(parts))


def translate_chunks(
    chunks: list[str],
    target_language: str = "Chinese",
    llm: str = "ollama/glm-5.2:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    use_deepl: bool = True,
    use_glossary: bool = True,
    glossary_file: str | None = None,
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

    # Determine glossary text for the LLM path (only for Chinese target).
    glossary_text = ""
    if use_glossary and (target_language or "").lower() in ("chinese", "zh"):
        glossary_text = _load_glossary_text(glossary_file)

    def translate_with_llm(text: str) -> str:
        nonlocal translate_module
        if translate_module is None:
            from wenbi.model import _import_dspy, configure_lm

            configure_lm(
                llm or "ollama/glm-5.2:cloud",
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
                glossary = dspy.InputField(desc="Optional term glossary; honor it for consistency", required=False)
                translated_text = dspy.OutputField(desc="Translated text")

            translate_module = dspy.Predict(TranslateSignature)
        kwargs = {"text_to_translate": text, "target_language": target_language}
        if glossary_text:
            kwargs["glossary"] = glossary_text
        return translate_module(**kwargs).translated_text

    translations: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        translated_text = None
        if deepl_available:
            try:
                from wenbi.llm.deepl import translate_with_deepl

                translated_text = translate_with_deepl(
                    deepl_translator, chunk, target_language, verbose=verbose,
                    use_glossary=use_glossary, glossary_file=glossary_file,
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

        # dspy.Predict can return None without raising — guard against that
        if translated_text is None or (isinstance(translated_text, str) and not translated_text.strip()):
            logger.warning(f"Empty translation for chunk {index}, using original text")
            translated_text = f"[Translation Error: {chunk}]"

        translations.append(translated_text)

    return translations


def _strip_topic_segment_markers(text: str) -> str:
    """Remove internal segment markers from topic-grouping output."""
    return re.sub(r"(?m)^\s*\[\d+\]\s*", "", text)


def group_into_topics(
    segments: list[dict[str, Any]],
    llm: str = "ollama/glm-5.2:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Group merged transcript segments into topic-coherent paragraphs via LLM.

    Preserves 100% of original text — only groups adjacent segments by topic
    and marks paragraph breaks with '---'.
    """
    # Build the transcript input with segment markers
    indexed_lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        indexed_lines.append(f"[{i}] {str(seg.get('text') or '').strip()}")
    transcript_text = "\n".join(indexed_lines)

    if verbose:
        logger.debug("Grouping %d segments into topic paragraphs", len(segments))

    if len(transcript_text) > 60_000:
        # ponytail: size-based chunks preserve every segment; replace with validated
        # batch topic grouping if an LLM round-trip can be proved lossless.
        paragraphs: list[str] = []
        current: list[str] = []
        current_size = 0
        for segment in segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            if current and current_size + len(text) + 1 > 6_000:
                paragraphs.append(" ".join(current))
                current = []
                current_size = 0
            current.append(text)
            current_size += len(text) + (1 if current_size else 0)
        if current:
            paragraphs.append(" ".join(current))
        if verbose:
            logger.warning(
                "Transcript is too large for lossless topic grouping; using %d size-based paragraphs",
                len(paragraphs),
            )
        return paragraphs

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

    module = dspy.Predict(TopicGroupSignature)
    result = module(transcript=transcript_text)

    if verbose:
        logger.debug("Topic grouping complete")

    # Split on '---' to get topic paragraphs
    paragraphs = [
        _strip_topic_segment_markers(paragraph).strip()
        for paragraph in result.grouped.split("---")
        if paragraph.strip()
    ]
    return paragraphs


def rewrite_english(
    paragraphs: list[str],
    llm: str = "ollama/glm-5.2:cloud",
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Turn spoken English into a faithful, readable edited transcript."""
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm,
        verbose=verbose,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    )
    dspy = _import_dspy()

    class EditedEnglishTranscriptSignature(dspy.Signature):
        """Edit spoken English into a readable written transcript, not an essay.

        Preserve every claim, argument, qualification, chronology, quotation,
        proper name, technical term, date, number, and intentional repetition.
        Keep hedges such as "I think", "perhaps", and "it seems" when they
        express the speaker's degree of certainty. Preserve speaker labels and
        their order exactly when they occur in the input.

        Remove only semantically empty fillers (for example "um", "uh", and
        "you know"), immediate accidental repetitions, abandoned false starts,
        and discourse-only openers such as "well", "okay", "right", or "so"
        when they do not express a logical relation. Repair punctuation and
        grammar, and combine fragments only when the meaning is unambiguous.
        Do not summarize, reorder, explain, strengthen, weaken,
        add facts, or guess at uncertain ASR wording. Do not emit timestamps,
        internal segment IDs such as "[307]", headings, commentary, or notes.

        Return only the edited transcript.
        """

        spoken_english_transcript: str = dspy.InputField(
            desc="One or more consecutive raw English ASR transcript turns"
        )
        written_transcript: str = dspy.OutputField(
            desc="Faithful edited transcript with source speaker labels retained when present"
        )

    module = dspy.Predict(EditedEnglishTranscriptSignature)

    rewritten: list[str] = []
    for i, paragraph in enumerate(paragraphs, 1):
        if verbose:
            logger.debug("Rewriting paragraph %d/%d", i, len(paragraphs))
        try:
            result = module(spoken_english_transcript=paragraph)
            rewritten.append(result.written_transcript.strip())
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
    llm: str = "ollama/glm-5.2:cloud",
    expected_speakers: int = 2,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Polish a Chinese interview transcript while preserving speaker turns."""
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm or "ollama/glm-5.2:cloud",
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
    llm: str = "ollama/glm-5.2:cloud",
    expected_speakers: int = 2,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    verbose: bool = False,
) -> list[str]:
    """Polish an English interview transcript while preserving speaker turns."""
    from wenbi.model import _import_dspy, configure_lm

    configure_lm(
        llm or "ollama/glm-5.2:cloud",
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
    llm: str = "ollama/glm-5.2:cloud",
    chunk_length: int = 20,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    use_deepl: bool = True,
    gladia_key: str | None = None,
    speaker_labels: bool = True,
    save_json: bool = False,
    verbose: bool = False,
    use_glossary: bool = True,
    glossary_file: str | None = None,
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
            provider = "funasr"

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
    elif provider == "funasr":
        try:
            segments = transcribe_with_sensevoice(
                audio_path, speaker_labels=speaker_labels, verbose=verbose
            )
        except Exception as e:
            if asr_provider == "auto":
                logger.warning(f"FunASR failed ({e}), falling back to Whisper")
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
        llm=llm or "ollama/glm-5.2:cloud",
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
        llm=llm or "ollama/glm-5.2:cloud",
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
        llm=llm or "ollama/glm-5.2:cloud",
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        deepl_key=deepl_key,
        use_deepl=use_deepl,
        use_glossary=use_glossary,
        glossary_file=glossary_file,
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


def _parse_vtt_to_segments(file_path, source_lang="zh"):
    """Parse a VTT file with optional <v Speaker> tags into segment dicts.

    Returns list of dicts with keys: start, end, text, speaker, language
    """
    import re
    segments = []
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Match VTT blocks: timestamp line followed by optional <v> text
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n'
        r'(?:<v\s+([^>]+)>\s*)?(.+?)(?=\n\d{2}:\d{2}:\d{2}|$)',
        re.DOTALL
    )

    def _ts_to_sec(ts):
        h, m, s = ts.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s.replace(',', '.'))

    for m in pattern.finditer(content):
        start = _ts_to_sec(m.group(1))
        end = _ts_to_sec(m.group(2))
        speaker = m.group(3) or None
        text = m.group(4).strip()
        # Remove closing </v> tag if present
        text = re.sub(r'</v>\s*$', '', text)
        if text:
            segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
                "language": source_lang,
            })

    return segments


def process_speaker(
    input_path: str,
    output_dir: str = "",
    start_time: str = "",
    end_time: str = "",
    asr_provider: str = "gladia",
    transcribe_model: str = "large-v3-turbo",
    source_lang: str = "en",
    target_language: str = "Chinese",
    llm: str = "ollama/glm-5.2:cloud",
    chunk_length: int = 20,
    max_tokens: int = 64000,
    timeout: int = 3600,
    temperature: float = 0.1,
    deepl_key: str | None = None,
    use_deepl: bool = True,
    gladia_key: str | None = None,
    speaker_labels: bool = True,
    speaker_count: int | None = None,
    rewrite_mode: str = "english",
    save_json: bool = False,
    verbose: bool = False,
    use_glossary: bool = True,
    glossary_file: str | None = None,
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

    # If input is a text file (VTT, SRT, etc.), parse it directly — no audio extraction needed
    audio_path = input_path  # default; overwritten for audio/video inputs
    if is_text_file(input_path):
        if verbose:
            logger.debug("Input is a text file (%s); skipping audio/ASR pipeline", input_path)
        segments = _parse_vtt_to_segments(input_path, source_lang=source_lang)
        # Label all segments with the source language (no filtering needed)
        for seg in segments:
            if not seg.get("language"):
                seg["language"] = source_lang

        merged = merge_adjacent_segments(segments)
        speakers_seen = {seg.get("speaker") for seg in merged if seg.get("speaker")}
        num_speakers = len(speakers_seen) or 1

        # Skip ASR/provider/vtt-raw sections; jump straight to rewrite/translate
        provider = "text"
        gladia_vtt_path = None
        diagnostics_path = (
            os.path.join(out_dir, f"{base_name}{suffix}_speaker_segments.json")
            if save_json
            else None
        )
        # ... continue with rewrite/translate logic below
    else:
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
                provider = "funasr"

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
        elif provider == "funasr":
            try:
                segments = transcribe_with_sensevoice(
                    audio_path, speaker_labels=speaker_labels, verbose=verbose
                )
            except Exception as e:
                if asr_provider == "auto":
                    logger.warning(f"FunASR failed ({e}), falling back to Whisper")
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

        # --- Save Gladia raw VTT (if using Gladia) ---
        gladia_vtt_path = None
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
            llm=llm or "ollama/glm-5.2:cloud",
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
            llm=llm or "ollama/glm-5.2:cloud",
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
            llm=llm or "ollama/glm-5.2:cloud",
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
            llm=llm or "ollama/glm-5.2:cloud",
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
            llm=llm or "ollama/glm-5.2:cloud",
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            deepl_key=deepl_key,
            use_deepl=use_deepl,
            use_glossary=use_glossary,
            glossary_file=glossary_file,
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
        _diagnostics_audio_path = input_path if is_text_file(input_path) else audio_path
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": provider,
                    "source_lang": source_lang,
                    "speaker_count": speaker_count,
                    "rewrite_mode": rewrite_mode,
                    "audio_path": _diagnostics_audio_path,
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
