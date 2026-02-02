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
import logging


def parse_subtitle(file_path, vtt_file=None, verbose=False):
    """
    Parses various subtitle formats (.ass, .sub, .srt, .txt, .vtt) into a DataFrame.
    If vtt_file is provided, it will be used directly as the content.
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Parsing subtitle file: {file_path}")
    
    if vtt_file is None:
        try:
            with open(file_path, "r", encoding="utf-8-sig", errors="replace") as file:
                lines = file.readlines()
            if verbose:
                logger.debug(f"Read {len(lines)} lines from file")
        except FileNotFoundError:
            if verbose:
                logger.debug(f"File not found: {file_path}")
            return pd.DataFrame(columns=["Timestamps", "Content"])
        except ImportError:
            print("pysrt library not found. Falling back to less robust parsing.")
    else:
        lines = vtt_file.splitlines()
        if verbose:
            logger.debug(f"Processing {len(lines)} lines from VTT content")

    timestamps = []
    contents = []
    current_content = []
    
    if file_path.lower().endswith(".txt") or (
        vtt_file is not None and file_path.lower().endswith(".txt")
    ):
        contents = lines
        timestamps = [""] * len(contents)
        if verbose:
            logger.debug(f"Processed as plain text file with {len(contents)} entries")
    else:
        i = 0
        segment_count = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check for timestamp line
            if "-->" in line or re.match(
                r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> \d{2}:\d{2}:\d{2}[,.]\d{3}", line
            ):
                timestamps.append(line)
                segment_count += 1
                i += 1
                current_content = []
                # Skip any empty lines and collect text until a new timestamp is detected.
                while i < len(lines) and not re.match(
                    r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> \d{2}:\d{2}:\d{2}[,.]\d{3}",
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
                segment_count += 1
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
        
        if verbose:
            logger.debug(f"Parsed {segment_count} subtitle segments")

    result_df = pd.DataFrame({"Timestamps": timestamps, "Content": contents})
    if verbose:
        logger.debug(f"Created DataFrame with {len(result_df)} rows")
    
    return result_df


def transcribe(file_path, language=None, output_dir=None, model_size="large-v3", verbose=False):
    """
    Transcribes an audio file to a WebVTT file with proper timestamps.

    Args:
        file_path (str): Path to the audio file
        language (str, optional): Language code for transcription
        output_dir (str, optional): Directory to save the VTT file
        model_size (str, optional): Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
        verbose (bool): Enable verbose logging
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Starting transcription of: {file_path}")
        logger.debug(f"Model size: {model_size}")
        logger.debug(f"Language: {language or 'auto-detect'}")
    
    model = whisper.load_model(f"{model_size}", device="cpu")
    if verbose:
        logger.debug(f"Whisper model loaded: {model_size}")
    
    result = model.transcribe(
        file_path, fp16=False, verbose=verbose, language=language if language else None
    )
    detected_language = result.get(
        "language", language if language else "unknown")

    if verbose:
        logger.debug(f"Transcription completed. Detected language: {detected_language}")
        logger.debug(f"Number of segments: {len(result['segments'])}")

    # Create VTT content with proper timestamps
    vtt_content = ["WEBVTT\n"]
    for i, segment in enumerate(result["segments"]):
        if verbose and i % 50 == 0:  # Log progress every 50 segments
            logger.debug(f"Processing segment {i+1}/{len(result['segments'])}")
        
        # Format timestamps
        hours = int(segment["start"] // 3600)
        minutes = int((segment["start"] % 3600) // 60)
        start_seconds = segment["start"] % 60
        end_hours = int(segment["end"] // 3600)
        end_minutes = int((segment["end"] % 3600) // 60)
        end_seconds = segment["end"] % 60

        start_time = f"{hours:02d}:{minutes:02d}:{start_seconds:06.3f}"
        end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:06.3f}"

        vtt_content.append(f"{start_time} --> {end_time}\n")
        vtt_content.append(f"{segment['text'].strip()}\n\n")

    # Determine output file path
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    vtt_file_path = os.path.join(output_dir, f"{base_name}.vtt")

    # Write VTT file
    with open(vtt_file_path, "w", encoding="utf-8") as f:
        f.writelines(vtt_content)

    if verbose:
        logger.debug(f"VTT file saved: {vtt_file_path}")

    # Create CSV file
    csv_data = []
    for segment in result["segments"]:
        start_time = f"{int(segment['start'] // 3600):02d}:{int((segment['start'] % 3600) // 60):02d}:{segment['start'] % 60:06.3f}"
        end_time = f"{int(segment['end'] // 3600):02d}:{int((segment['end'] % 3600) // 60):02d}:{segment['end'] % 60:06.3f}"
        csv_data.append({
            "Timestamps": f"{start_time} --> {end_time}",
            "Content": segment["text"].strip()
        })

    csv_df = pd.DataFrame(csv_data)
    csv_file_path = os.path.join(output_dir, f"{base_name}.csv")
    csv_df.to_csv(csv_file_path, index=False, encoding="utf-8")

    if verbose:
        logger.debug(f"CSV file saved: {csv_file_path}")
        logger.debug(f"Transcription process completed successfully")

    return vtt_file_path, csv_file_path


def segment(file_path, sentence_count=20, cite_timestamps=False, verbose=False):
    """
    Segments text into paragraphs with a fixed number of sentences.
    
    Args:
        file_path (str): Path to the input file
        sentence_count (int): Number of sentences per paragraph
        cite_timestamps (bool): Whether to include timestamp headers
        verbose (bool): Enable verbose logging
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Starting text segmentation: {file_path}")
        logger.debug(f"Sentences per paragraph: {sentence_count}")
        logger.debug(f"Include timestamps: {cite_timestamps}")
    
    try:
        vtt_df = parse_subtitle(file_path, verbose=verbose)
        
        if verbose:
            logger.debug(f"Parsed {len(vtt_df)} subtitle segments")
        
        if cite_timestamps and not vtt_df.empty and vtt_df['Timestamps'].notna().any():
            if verbose:
                logger.debug("Using timestamp-aware segmentation")
            return _segment_with_timestamps(vtt_df, sentence_count, verbose=verbose)
        
        # Regular segmentation without timestamps
        all_content = "。".join(vtt_df["Content"]) if not vtt_df.empty else ""
        
        if verbose:
            logger.debug(f"Total content length: {len(all_content)} characters")
        
        # Remove repeated patterns
        pattern = r"(([\\u4e00-\\u9fa5。！？；：""（）【】《》、]{1,5}))(\\s?\\1)+"
        cleaned_content = re.sub(pattern, r"\\1", all_content)
        
        if verbose and len(cleaned_content) != len(all_content):
            logger.debug(f"Removed repetitions, new length: {len(cleaned_content)} characters")

        # Detect language and initialize appropriate model
        if any(char in cleaned_content for char in "，。？！"):
            nlp = Chinese()
            if verbose:
                logger.debug("Detected Chinese text, using Chinese NLP model")
        else:
            nlp = English()
            if verbose:
                logger.debug("Detected non-Chinese text, using English NLP model")

        # Add sentencizer if not present
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        doc = nlp(cleaned_content)
        
        # Process sentences into paragraphs
        paragraphs = []
        current_sentences = []
        count = 0
        total_sentences = len(list(doc.sents))
        
        if verbose:
            logger.debug(f"Processing {total_sentences} sentences")
        
        for i, sent in enumerate(doc.sents):
            if verbose and i % 100 == 0:  # Log progress every 100 sentences
                logger.debug(f"Processing sentence {i+1}/{total_sentences}")
            
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Add Chinese comma if needed
            if not any(sent_text.endswith(p) for p in "，。？！,.!?"):
                sent_text += "，"
                
            current_sentences.append(sent_text)
            count += 1
            
            # Create new paragraph when reaching sentence count
            if count >= sentence_count:
                paragraphs.append("".join(current_sentences))
                current_sentences = []
                count = 0
        
        # Add remaining sentences if any
        if current_sentences:
            paragraphs.append("".join(current_sentences))
        
        result = "\n\n".join(paragraphs)
        
        if verbose:
            logger.debug(f"Created {len(paragraphs)} paragraphs")
            logger.debug(f"Final text length: {len(result)} characters")
        
        return result
        
    except Exception as e:
        error_msg = f"An error occurred during segmentation: {e}"
        if verbose:
            logger.debug(error_msg)
        return error_msg


def _segment_with_timestamps(vtt_df, sentence_count, verbose=False):
    """Helper function for timestamp-aware segmentation"""
    logger = logging.getLogger(__name__)
    
    def format_timestamp(timestamp_str):
        """Format timestamp string to remove milliseconds for cleaner display"""
        if "-->" in timestamp_str:
            parts = timestamp_str.split(" --> ")
            if len(parts) == 2:
                start = parts[0].split(".")[0] if "." in parts[0] else parts[0].split(",")[0]
                end = parts[1].split(".")[0] if "." in parts[1] else parts[1].split(",")[0]
                return f"{start} - {end}"
        return timestamp_str
    
    paragraphs = []
    current_sentences = []
    current_timestamps = []
    count = 0
    
    if verbose:
        logger.debug(f"Processing {len(vtt_df)} segments with timestamps")
    
    for idx, row in vtt_df.iterrows():
        content = str(row["Content"]).strip()
        timestamp = str(row["Timestamps"]).strip()
        
        if not content:
            continue
        
        # Add sentences from this segment
        sentences = re.split(r'[。！？.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            if not any(sentence.endswith(p) for p in "，。？！,.!?"):
                sentence += "，"
            
            current_sentences.append(sentence)
            current_timestamps.append(timestamp)
            count += 1
            
            # Create new paragraph when reaching sentence count
            if count >= sentence_count:
                if current_timestamps:
                    first_timestamp = format_timestamp(current_timestamps[0])
                    last_timestamp = format_timestamp(current_timestamps[-1])
                    
                    # Create timestamp header
                    if first_timestamp == last_timestamp:
                        header = f"### **{first_timestamp}**\n\n"
                    else:
                        # Extract time ranges
                        first_time = first_timestamp.split(" - ")[0] if " - " in first_timestamp else first_timestamp
                        last_time = last_timestamp.split(" - ")[1] if " - " in last_timestamp else last_timestamp
                        header = f"### **{first_time} - {last_time}**\n\n"
                    
                    paragraph_content = "".join(current_sentences)
                    paragraphs.append(header + paragraph_content)
                else:
                    paragraphs.append("".join(current_sentences))
                
                current_sentences = []
                current_timestamps = []
                count = 0
    
    # Add remaining sentences if any
    if current_sentences:
        if current_timestamps:
            first_timestamp = format_timestamp(current_timestamps[0])
            last_timestamp = format_timestamp(current_timestamps[-1])
            
            if first_timestamp == last_timestamp:
                header = f"### **{first_timestamp}**\n\n"
            else:
                first_time = first_timestamp.split(" - ")[0] if " - " in first_timestamp else first_timestamp
                last_time = last_timestamp.split(" - ")[1] if " - " in last_timestamp else last_timestamp
                header = f"### **{first_time} - {last_time}**\n\n"
            
            paragraph_content = "".join(current_sentences)
            paragraphs.append(header + paragraph_content)
        else:
            paragraphs.append("".join(current_sentences))
    
    result = "\n\n".join(paragraphs)
    
    if verbose:
        logger.debug(f"Created {len(paragraphs)} timestamped paragraphs")
    
    return result


def _sanitize_filename(filename):
    """Sanitize a string to be a valid filename."""
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Truncate to a reasonable length
    return sanitized[:100]


def download_audio(url, output_dir=None, timestamp=None, output_wav=None, verbose=False):
    """
    Downloads audio from a URL using yt-dlp and converts it to WAV format.
    
    Args:
        url (str): URL to download audio from
        output_dir (str): Directory to save the downloaded audio
        timestamp (dict): Optional start/end time for extraction
        output_wav (str): Optional custom filename for WAV output
        verbose (bool): Enable verbose logging
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Starting audio download from URL: {url}")
    
    import subprocess
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    if verbose:
        logger.debug(f"Output directory: {output_dir}")
    
    try:
        # Get video title using yt-dlp
        get_title_cmd = ["yt-dlp", "--get-title", "--no-warnings", url]
        if verbose:
            logger.debug(f"Fetching title with command: {' '.join(get_title_cmd)}")
        
        title_result = subprocess.run(get_title_cmd, capture_output=True, text=True, check=True)
        video_title = title_result.stdout.strip()
        base_filename = _sanitize_filename(video_title)
        
        if verbose:
            logger.debug(f"Sanitized filename: {base_filename}")

        # Set the output path for the downloaded file
        temp_path = os.path.join(output_dir, base_filename)

        # Download audio using yt-dlp
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--output", f"{temp_path}.%(ext)s",
            url
        ]
        
        if verbose:
            logger.debug(f"Running yt-dlp command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if verbose:
            logger.debug("yt-dlp download completed successfully")
        
        # Find the downloaded file
        downloaded_file = f"{temp_path}.wav"
        if not os.path.exists(downloaded_file):
            # Look for other possible extensions
            for ext in ['.wav', '.mp3', '.m4a', '.webm']:
                if os.path.exists(f"{temp_path}{ext}"):
                    downloaded_file = f"{temp_path}{ext}"
                    break
        
        if not os.path.exists(downloaded_file):
            raise FileNotFoundError("Downloaded file not found")
        
        # Extract segment if timestamp is provided
        if timestamp:
            if verbose:
                logger.debug(f"Extracting segment: {timestamp['start']} - {timestamp['end']}")
            final_path = extract_audio_segment(downloaded_file, timestamp, output_dir, output_wav, verbose=verbose)
            # Clean up temporary file
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
        else:
            # Rename to final filename if output_wav is specified
            if output_wav:
                final_path = os.path.join(output_dir, output_wav)
                if not final_path.endswith('.wav'):
                    final_path += '.wav'
                os.rename(downloaded_file, final_path)
            else:
                final_path = downloaded_file
        
        if verbose:
            logger.debug(f"Audio download completed: {final_path}")
        
        return final_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error downloading audio: {e.stderr}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Error in download_audio: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def language_detect(file_path, detected_lang=None, verbose=False):
    """Detect language of the content in a file"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Detecting language for: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if verbose:
            logger.debug(f"File content length: {len(content)} characters")
        
        # Use provided language if available
        if detected_lang:
            if verbose:
                logger.debug(f"Using provided language: {detected_lang}")
            return detected_lang
        
        # Auto-detect language
        detected = detect(content)
        if verbose:
            logger.debug(f"Auto-detected language: {detected}")
        
        return detected
        
    except (LangDetectException, Exception) as e:
        if verbose:
            logger.debug(f"Language detection failed: {e}, defaulting to 'en'")
        return "en"


def parse_timestamp(start_time=None, end_time=None, verbose=False):
    """Parse start and end time strings and return in seconds"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Parsing timestamps: {start_time} - {end_time}")
    
    if not start_time or not end_time:
        return None
        
    def time_to_seconds(time_str):
        """Convert HH:MM:SS format to seconds"""
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        return 0
    
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    if verbose:
        logger.debug(f"Converted to seconds: {start_seconds} - {end_seconds}")
    
    return {'start': start_seconds, 'end': end_seconds}


def extract_audio_segment(audio_path, timestamp=None, output_dir=None, output_wav="", verbose=False):
    """
    Extract audio segment from video/audio file and convert to WAV format.
    
    Args:
        audio_path (str): Path to the input audio/video file
        timestamp (dict): Optional dict with 'start' and 'end' times in seconds
        output_dir (str): Directory to save the output
        output_wav (str): Custom filename for the output WAV file
        verbose (bool): Enable verbose logging
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Extracting audio from: {audio_path}")
        if timestamp:
            logger.debug(f"Segment: {timestamp['start']}s - {timestamp['end']}s")
    
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Determine output filename
    if output_wav:
        if not output_wav.endswith('.wav'):
            output_wav += '.wav'
        output_path = os.path.join(output_dir, output_wav)
    else:
        if timestamp:
            start_str = f"{int(timestamp['start']//3600):02d}{int((timestamp['start']%3600)//60):02d}{int(timestamp['start']%60):02d}"
            end_str = f"{int(timestamp['end']//3600):02d}{int((timestamp['end']%3600)//60):02d}{int(timestamp['end']%60):02d}"
            output_path = os.path.join(output_dir, f"{base_name}_{start_str}_{end_str}.wav")
        else:
            output_path = os.path.join(output_dir, f"{base_name}.wav")
    
    if verbose:
        logger.debug(f"Output path: {output_path}")
    
    try:
        # Check if input is video or audio
        if audio_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm')):
            if verbose:
                logger.debug("Processing as video file")
            with VideoFileClip(audio_path) as video:
                audio_clip = video.audio
                if timestamp:
                    audio_clip = audio_clip.subclip(timestamp['start'], timestamp['end'])
                audio_clip.write_audiofile(output_path, logger=None)
        else:
            if verbose:
                logger.debug("Processing as audio file")
            with AudioFileClip(audio_path) as audio:
                if timestamp:
                    audio = audio.subclip(timestamp['start'], timestamp['end'])
                audio.write_audiofile(output_path, logger=None)
        
        if verbose:
            logger.debug(f"Audio extraction completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        error_msg = f"Error extracting audio: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def save_markdown(content, output_path, verbose=False):
    """Save content to a markdown file"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug(f"Saving markdown to: {output_path}")
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        if verbose:
            logger.debug("Markdown file saved successfully")
    except Exception as e:
        if verbose:
            logger.debug(f"Error saving markdown: {e}")
        raise Exception(f"Error saving markdown: {e}")