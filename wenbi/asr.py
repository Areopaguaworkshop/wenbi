import whisper
import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _is_qwen3_model(model_name: str) -> bool:
    """Check if model name is a Qwen3 model"""
    return model_name in ("0.6B", "1.7B")


def _get_qwen3_model_id(model_name: str) -> str:
    """Get HuggingFace model ID for Qwen3"""
    return f"Qwen/Qwen3-ASR-{model_name}"


def transcribe_with_engine(
    audio_path: str,
    model_size: str = "1.7B",
    language: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    Transcribe audio using Qwen3-ASR with Whisper fallback.
    
    Args:
        audio_path: Path to audio file
        model_size: "1.7B" (Qwen3) or whisper size (tiny, base, large-v3, etc.)
        language: Language code or None for auto-detect
        verbose: Enable logging
    
    Returns:
        Dict with text, language, segments
    """
    if _is_qwen3_model(model_size):
        return _transcribe_qwen3(audio_path, model_size, language, verbose)
    else:
        return _transcribe_whisper(audio_path, model_size, language, verbose)


def _transcribe_qwen3(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    verbose: bool,
) -> Dict:
    """Transcribe using Qwen3-ASR"""
    try:
        from qwen_asr import Qwen3ASRModel
        
        if verbose:
            logger.debug(f"Loading Qwen3-ASR-{model_size} model...")
        
        model = Qwen3ASRModel.from_pretrained(
            _get_qwen3_model_id(model_size),
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            max_new_tokens=256,
        )
        
        if verbose:
            logger.debug("Transcribing with Qwen3-ASR...")
        
        result = model.transcribe(
            audio=audio_path,
            language=language or "auto",
        )
        
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            return {
                "text": getattr(item, "text", ""),
                "language": getattr(item, "language", language or "unknown"),
                "segments": _extract_segments(getattr(item, "segments", [])),
            }
        elif isinstance(result, dict):
            return {
                "text": result.get("text", ""),
                "language": result.get("language", language or "unknown"),
                "segments": _extract_segments(result.get("segments", [])),
            }
        else:
            raise ValueError(f"Unexpected Qwen3-ASR result type: {type(result)}")
            
    except ImportError as e:
        logger.warning(f"Qwen3-ASR not available ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, "large-v3", language, verbose)
    except Exception as e:
        logger.warning(f"Qwen3-ASR failed ({e}), falling back to Whisper")
        return _transcribe_whisper(audio_path, "large-v3", language, verbose)


def _extract_segments(segments) -> list:
    """Extract segments from Qwen3-ASR result, normalizing to dict format"""
    result = []
    for seg in segments:
        if isinstance(seg, dict):
            result.append(seg)
        else:
            result.append({
                "start": getattr(seg, "start", 0),
                "end": getattr(seg, "end", 0),
                "text": getattr(seg, "text", ""),
            })
    return result


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
