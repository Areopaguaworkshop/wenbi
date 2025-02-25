import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
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
    # Initialize speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    )
    
    # Run diarization
    diarization = pipeline(audio_path)
    
    # Group segments by speaker
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append({
            'start': turn.start,
            'end': turn.end
        })
    
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
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            speaker_audio += audio[start_ms:end_ms]
        
        # Save speaker's audio
        output_path = f"{os.path.splitext(audio_path)[0]}_{speaker}.wav"
        speaker_audio.export(output_path, format="wav")
        speaker_files[speaker] = output_path
    
    return speaker_files

def transcribe_multi_speaker(audio_path, language_hints=None):
    """
    Transcribe multi-speaker audio with language detection per speaker.
    
    Args:
        audio_path (str): Path to the WAV audio file
        language_hints (dict, optional): Dictionary mapping speaker IDs to language hints
        
    Returns:
        dict: Dictionary containing speaker transcriptions and detected languages
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
    model = whisper.load_model("large-v3-turbo")
    transcriptions = {}
    
    for speaker, speaker_path in speaker_files.items():
        # Get language hint for this speaker if provided
        language = language_hints.get(speaker) if language_hints else None
        
        # Transcribe with language hint
        result = model.transcribe(
            speaker_path,
            language=language,
            fp16=False,
            verbose=True
        )
        
        transcriptions[speaker] = {
            'detected_language': result['language'],
            'segments': result['segments']
        }
    
    return transcriptions

def format_transcription(transcriptions):
    """
    Format transcriptions into VTT format with speaker labels.
    
    Args:
        transcriptions (dict): Output from transcribe_multi_speaker()
        
    Returns:
        str: VTT formatted transcription
    """
    vtt_lines = ["WEBVTT\n"]
    
    for speaker, data in transcriptions.items():
        language = data['detected_language']
        for segment in data['segments']:
            # Debug: confirm format_timestamp is callable
            if not callable(format_timestamp):
                print("Error: format_timestamp is not callable!")
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            vtt_lines.append(f"\n{start} --> {end}")
            vtt_lines.append(f"[{speaker} - {language}]")
            vtt_lines.append(f"{text}\n")
    
    return "\n".join(vtt_lines)

def speaker_vtt(transcriptions, output_dir=None, base_filename=""):
    """
    Create separate VTT files for each speaker from the multi-speaker transcriptions.
    
    Args:
        transcriptions (dict): Mapping of speaker IDs to their transcription data.
        output_dir (str, optional): Directory in which to save the VTT files. Defaults to current directory.
        base_filename (str, optional): Base filename to prepend. If empty, speaker IDs are used as filenames.
        
    Returns:
        dict: Mapping of speaker IDs to their generated VTT file paths.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    speaker_files = {}
    
    for speaker, data in transcriptions.items():
        vtt_lines = ["WEBVTT\n"]
        language = data.get('detected_language', 'unknown')
        for segment in data['segments']:
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            vtt_lines.append(f"\n{start} --> {end}")
            vtt_lines.append(f"[{speaker} - {language}]")
            vtt_lines.append(f"{text}\n")
        vtt_content = "\n".join(vtt_lines)
        filename = f"{base_filename}_{speaker}.vtt" if base_filename else f"{speaker}.vtt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(vtt_content)
        speaker_files[speaker] = filepath
        
    return speaker_files

if __name__ == "__main__":
    # Example usage
    audio_file = "path/to/your/audio.wav"
    
    # Optional: Provide language hints for speakers
    language_hints = {
        "SPEAKER_00": "en",
        "SPEAKER_01": "zh"
    }
    
    try:
        transcriptions = transcribe_multi_speaker(audio_file, language_hints)
        vtt_content = format_transcription(transcriptions)
        
        # Save VTT file
        output_path = os.path.splitext(audio_file)[0] + "_multi.vtt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)
            
        print(f"Transcription saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
