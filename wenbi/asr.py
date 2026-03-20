import whisper
import torch
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

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

# FunASR model that supports sentence-level timestamps
FUNASR_ZH_WITH_TS = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FUNASR_VAD_ZH = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
FUNASR_PUNC_ZH = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

FUNASR_ALIASES = {
    # Backward-compatible aliases for old Qwen3 options
    "1.7B": FUNASR_ZH_WITH_TS,
    "0.6B": FUNASR_ZH_WITH_TS,
    # Friendly short names
    "paraformer-zh": FUNASR_ZH_WITH_TS,
}

def _is_whisper_model(model_name: str) -> bool:
    """Check if model name is a Whisper model"""
    return model_name in WHISPER_MODELS


def _normalize_funasr_model(model_name: str) -> str:
    """Map legacy model names to FunASR model identifiers"""
    return FUNASR_ALIASES.get(model_name, model_name)


def transcribe_with_engine(
    audio_path: str,
    model_size: str = "1.7B",
    language: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    Transcribe audio using FunASR with Whisper fallback.

    Args:
        audio_path: Path to audio file
        model_size: FunASR model name (e.g., paraformer-zh) or whisper size
        language: Language code or None for auto-detect
        verbose: Enable logging

    Returns:
        Dict with text, language, segments
    """
    if _is_whisper_model(model_size):
        return _transcribe_whisper(audio_path, model_size, language, verbose)
    return _transcribe_funasr(audio_path, model_size, language, verbose)


def _transcribe_funasr(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    verbose: bool,
) -> Dict:
    """Transcribe using FunASR"""
    try:
        from funasr import AutoModel

        normalized_model = _normalize_funasr_model(model_size)
        if verbose and normalized_model != model_size:
            logger.debug(f"Mapping model '{model_size}' -> '{normalized_model}'")

        if verbose:
            logger.debug(f"Loading FunASR model: {normalized_model}")

        model = AutoModel(
            model=normalized_model,
            vad_model=FUNASR_VAD_ZH,
            punc_model=FUNASR_PUNC_ZH,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

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
        }
    except ImportError as e:
        logger.warning(f"FunASR not available ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, "large-v3", language, verbose)
    except Exception as e:
        logger.warning(f"FunASR failed ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, "large-v3", language, verbose)


def _extract_funasr_segments(result: Any) -> List[Dict]:
    """Extract segments from FunASR result, normalizing to dict format"""
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
        return {"start": start, "end": end, "text": text}

    start = _normalize_funasr_time(getattr(item, "start", 0.0))
    end = _normalize_funasr_time(getattr(item, "end", 0.0))
    text = getattr(item, "text", "")
    return {"start": start, "end": end, "text": text}


def _normalize_funasr_time(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    # Heuristic: FunASR timestamps are often milliseconds. If value is large,
    # treat it as ms and convert to seconds.
    if numeric > 1000.0:
        return numeric / 1000.0
    return numeric


def _transcribe_whisper(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    verbose: bool,
) -> Dict:
    """Transcribe using Whisper"""
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
    }
