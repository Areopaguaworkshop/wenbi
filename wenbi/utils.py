import os
import whisper
import re
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from spacy.lang.zh import Chinese
from spacy.lang.en import English
import spacy
from langdetect import detect, detect_langs, LangDetectException
from pydub import AudioSegment


def parse_subtitle(file_path, vtt_file=None):
    """
    Parses various subtitle formats (.ass, .sub, .srt, .txt, .vtt) into a DataFrame.
    If vtt_file is provided, it will be used directly as the content.
    """
    if vtt_file is None:
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as file:
                lines = file.readlines()
        except FileNotFoundError:
            return pd.DataFrame(columns=["Timestamps", "Content"])
        except ImportError:
            print("pysrt library not found. Falling back to less robust parsing.")
    else:
        lines = vtt_file.splitlines()

    timestamps = []
    contents = []
    current_content = []
    if file_path.lower().endswith(".txt") or (
        vtt_file is not None and file_path.lower().endswith(".txt")
    ):
        contents = lines
        timestamps = [""] * len(contents)
    else:
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check for timestamp line
            if "-->" in line or re.match(
                r"\d{2}:\d{2}:\d{2}[,\.]\d{3} --> \d{2}:\d{2}:\d{2}[,\.]\d{3}", line
            ):
                timestamps.append(line)
                i += 1
                current_content = []
                # Skip any empty lines and collect text until a new timestamp is detected.
                while i < len(lines) and not re.match(
                    r"\d{2}:\d{2}:\d{2}[,\.]\d{3} --> \d{2}:\d{2}:\d{2}[,\.]\d{3}",
                    lines[i].strip(),
                ):
                    stripped = lines[i].strip()
                    if stripped:  # only add non-empty text lines
                        current_content.append(stripped)
                    i += 1
                contents.append(" ".join(current_content))
            # Handle other subtitle formats (Dialogue or similar)
            elif "Dialogue:" in line or re.match(r"{\d+}{\d+}.*", line):
                timestamps.append(line)
                i += 1
                current_content = []
                while i < len(lines) and not lines[i].strip().isdigit():
                    stripped = lines[i].strip()
                    if stripped:
                        current_content.append(stripped)
                    i += 1
                contents.append(" ".join(current_content))
            else:
                i += 1

    return pd.DataFrame({"Timestamps": timestamps, "Content": contents})


def transcribe(file_path, language=None, output_dir=None, model_size="large-v3"):
    """
    Transcribes an audio file to a WebVTT file with proper timestamps.

    Args:
        file_path (str): Path to the audio file
        language (str, optional): Language code for transcription
        output_dir (str, optional): Directory to save the VTT file
        model_size (str, optional): Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
    """
    model = whisper.load_model(f"{model_size}", device="cpu")
    result = model.transcribe(
        file_path, fp16=False, verbose=True, language=language if language else None
    )
    detected_language = result.get(
        "language", language if language else "unknown")

    # Create VTT content with proper timestamps
    vtt_content = ["WEBVTT\n"]
    for segment in result["segments"]:
        # ...existing timestamp formatting...
        hours = int(segment["start"] // 3600)
        minutes = int((segment["start"] % 3600) // 60)
        start_seconds = segment["start"] % 60
        end_hours = int(segment["end"] // 3600)
        end_minutes = int((segment["end"] % 3600) // 60)
        end_seconds = segment["end"] % 60

        start_time = f"{hours:02d}:{minutes:02d}:{start_seconds:06.3f}"
        end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:06.3f}"
        text = segment["text"].strip()
        vtt_content.append(f"\n{start_time} --> {end_time}\n{text}")

    # Use provided output_dir or default to the base file's directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))
    else:
        os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(output_dir, base_name + ".vtt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(" ".join(vtt_content))

    return out_file, detected_language


def segment(file_path, sentence_count=20):
    """Segments a text file into paragraphs by grouping every N sentences."""
    try:
        # Handle docx files
        if file_path.lower().endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        else:
            vtt_df = parse_subtitle(file_path)
            text = "。".join(vtt_df["Content"])

        # Directly use basic language classes
        if any(char in text for char in "，。？！"):
            nlp = Chinese()
        else:
            nlp = English()

        # Add the sentencizer component to the pipeline
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        doc = nlp(text)

        paragraphs = []
        current_paragraph = []
        current_count = 0
        for sent in doc.sents:
            # Add Chinese comma if needed
            sent_text = sent.text.strip()
            if not any(sent_text.endswith(p) for p in "，。？！,.!?"):
                sent_text += "，"
            current_paragraph.append(sent_text)
            current_count += 1
            if current_count >= sentence_count:
                paragraphs.append("".join(current_paragraph))
                current_paragraph = []
                current_count = 0

        if current_paragraph:
            paragraphs.append("".join(current_paragraph))

        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"Error in segment: {e}")
        return text


def download_audio(url, output_dir=None, timestamp=None, output_wav=None):
    """
    Download audio from a URL and convert it to WAV format.

    Args:
        url (str): URL of the video/audio to download
        output_dir (str, optional): Directory to save the downloaded file
        timestamp (tuple, optional): (start_seconds, end_seconds) for extraction
        output_wav (str, optional): Custom filename for the output WAV file

    Returns:
        str: Path to the downloaded WAV file
    """
    import yt_dlp

    if output_dir is None:
        output_dir = os.getcwd()

    # If output_wav is provided, use it as the output filename (without extension)
    if output_wav:
        output_wav = os.path.splitext(output_wav)[0]  # Remove extension if present
        outtmpl = os.path.join(output_dir, f"{output_wav}.%(ext)s")
    else:
        outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if output_wav:
                output_file = os.path.join(output_dir, f"{output_wav}.wav")
            else:
                output_file = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".wav"

            if timestamp:
                # Extract the specified segment
                return extract_audio_segment(output_file, timestamp, output_dir)
            return output_file
    except Exception as e:
        raise Exception(f"Error downloading audio: {str(e)}")


def language_detect(file_path, detected_lang=None):
    """
    Detects the language of a text file using langdetect.
    Returns language code (e.g., 'zh', 'en', etc.).
    """
    try:
        df = parse_subtitle(file_path)
        sample_content = " ".join(df["Content"].head(20))
        if not sample_content.strip():
            # Fallback if file content is empty or only whitespace
            return "en"
        languages = detect_langs(sample_content)
        if languages:
            detected = languages[0].lang
            return "zh" if detected.startswith("zh") else detected
    except Exception as e:
        print(f"Language detection error: {e}")
    return "en"


def parse_timestamp(start_time=None, end_time=None):
    """Parse start and end times in HH:MM:SS format to seconds tuple."""
    if not start_time or not end_time:
        return None
    try:
        def time_to_seconds(time_str):
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

        return (time_to_seconds(start_time), time_to_seconds(end_time))
    except:
        raise ValueError("Invalid time format. Use HH:MM:SS")


def extract_audio_segment(audio_path, timestamp=None, output_dir=None, output_wav=""):
    """
    Extract full audio or segment using moviepy.

    Args:
        audio_path (str): Path to input audio/video file
        timestamp (dict, optional): Dictionary with 'start' and 'end' times in HH:MM:SS format
        output_dir (str, optional): Output directory for the extracted audio
        output_wav (str, optional): Custom filename for the output WAV file
    """
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    try:
        # Use VideoFileClip first to handle both video and audio files
        try:
            clip = VideoFileClip(audio_path)
            audio = clip.audio
        except:
            # If not video, try loading as audio
            audio = AudioFileClip(audio_path)

        if timestamp:
            # Convert HH:MM:SS to seconds
            start = sum(x * int(t) for x, t in zip([3600, 60, 1], timestamp['start'].split(':')))
            end = sum(x * int(t) for x, t in zip([3600, 60, 1], timestamp['end'].split(':')))

            # Extract segment
            audio = audio.subclipped(start, end)
            if output_wav:
                # Remove .wav extension if present in output_wav
                output_wav = os.path.splitext(output_wav)[0]
                output_path = os.path.join(output_dir, f"{output_wav}.wav")
            else:
                # Use default timestamp-based filename
                output_path = os.path.join(output_dir, f"{base_name}_{timestamp['start']}-{timestamp['end']}.wav")
        else:
            output_path = os.path.join(output_dir, f"{base_name}.wav")

        # Write WAV file
        audio.write_audiofile(output_path, codec='pcm_s16le')

        # Clean up
        audio.close()
        if 'clip' in locals():
            clip.close()

        return output_path

    except Exception as e:
        raise Exception(f"Error extracting audio: {e}")
