from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import os

# Ensure this helper function is defined before any calls


def format_timestamp(seconds):
    """Convert seconds to VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


# Debug: print type of format_timestamp to be sure it is callable
print("format_timestamp type:", type(format_timestamp))


def separate_speakers(audio_path, auth_token=None):
    """
    Separate audio file by speakers using pyannote.audio.

    Args:
        audio_path (str): Path to the WAV audio file
        auth_token (str): HuggingFace authentication token for pyannote.audio

    Returns:
        dict: Dictionary mapping speaker IDs to list of time segments
    """
    # Removed min_duration argument (no longer supported)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token,
    )

    # Run diarization
    diarization = pipeline(audio_path)

    # Group segments by speaker
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append({"start": turn.start, "end": turn.end})

    return speakers


def extract_speaker_segments(audio_path, speaker_segments):
    """
    Extract audio segments for each speaker and save as separate files.

    Args:
        audio_path (str): Path to the original audio file
        speaker_segments (dict): Dictionary of speaker segments from separate_speakers()

    Returns:
        dict: Dictionary mapping speaker IDs to their audio file paths
    """
    audio = AudioSegment.from_wav(audio_path)
    speaker_files = {}

    for speaker, segments in speaker_segments.items():
        # Combine all segments for this speaker
        speaker_audio = AudioSegment.empty()
        for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            speaker_audio += audio[start_ms:end_ms]

        # Save speaker's audio
        output_path = f"{os.path.splitext(audio_path)[0]}_{speaker}.wav"
        speaker_audio.export(output_path, format="wav")
        speaker_files[speaker] = output_path

    return speaker_files


def transcribe_multi_speaker(audio_path, language_hints=None, model_size="large-v3"):
    """
    Transcribe multi-speaker audio with language detection per speaker.

    Args:
        audio_path (str): Path to the WAV audio file
        language_hints (dict, optional): Dictionary mapping speaker IDs to language hints
        model_size (str): Whisper model size to use
    """
    # Get your HuggingFace token from environment variable
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    if not auth_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")

    # Step 1: Separate speakers
    print("Separating speakers...")
    speaker_segments = separate_speakers(audio_path, auth_token)

    # Step 2: Extract audio for each speaker
    print("Extracting speaker segments...")
    speaker_files = extract_speaker_segments(audio_path, speaker_segments)

    # Step 3: Transcribe each speaker's audio
    print("Transcribing speaker segments...")
    model = whisper.load_model(model_size)
    transcriptions = {}

    for spk, speaker_path in extract_speaker_segments(
        audio_path, separate_speakers(
            audio_path, os.getenv("HUGGINGFACE_TOKEN"))
    ).items():
        lang_hint = language_hints.get(spk) if language_hints else None
        result = model.transcribe(
            speaker_path, language=lang_hint, fp16=False, verbose=True
        )
        lang = result["language"]
        if lang in transcriptions:
            transcriptions[lang]["segments"].extend(result["segments"])
        else:
            transcriptions[lang] = {
                "detected_language": lang,
                "segments": result["segments"],
            }

    return transcriptions


def format_transcription(transcriptions):
    """
    Format transcriptions into VTT format without language tags.

    Args:
        transcriptions (dict): Output from transcribe_multi_speaker()

    Returns:
        str: VTT formatted transcription without language tag lines.
    """
    vtt_lines = ["WEBVTT\n"]

    for speaker, data in transcriptions.items():
        # Remove language tag; do not extract 'detected_language'
        for segment in data["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()

            vtt_lines.append(f"\n{start} --> {end}")
            # Removed the line that adds language tag
            vtt_lines.append(f"{text}\n")

    return "\n".join(vtt_lines)


def speaker_vtt(transcriptions, output_dir=None, base_filename=""):
    """
    Create separate VTT files for each speaker from the multi-speaker transcriptions.

    Args:
        transcriptions (dict): Mapping of speaker IDs (or languages) to their transcription data.
        output_dir (str, optional): Directory in which to save the VTT files. Defaults to current directory.
        base_filename (str, optional): Base filename to prepend.

    Returns:
        dict: Mapping of speaker IDs (or languages) to their generated VTT file paths.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)
    vtt_files = {}

    for lang, data in transcriptions.items():
        vtt_lines = ["WEBVTT\n"]
        for segment in data["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            vtt_lines.append(f"\n{start} --> {end}")
            vtt_lines.append(f"[{lang}]")
            vtt_lines.append(f"{text}\n")
        vtt_content = "\n".join(vtt_lines)
        filename = f"{base_filename}_{
            lang}.vtt" if base_filename else f"{lang}.vtt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(vtt_content)
        vtt_files[lang] = filepath

    return vtt_files
