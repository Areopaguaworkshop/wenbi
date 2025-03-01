import os
import json
import subprocess

# Optional: to extract URL metadata using yt-dlp
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

def get_url_metadata(url):
    """
    Extract metadata from a URL using yt-dlp.
    Returns a dictionary of metadata.
    """
    if yt_dlp is None:
        return {"error": "yt-dlp not installed"}
    try:
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'forcejson': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info
    except Exception as e:
        return {"error": str(e)}

def get_media_metadata(file_path):
    """
    Use ffprobe to extract metadata from a video or audio file.
    Returns a dictionary of metadata.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams', file_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(output)
        return info
    except Exception as e:
        return {"error": str(e)}

def get_subtitle_metadata(file_path):
    """
    Extract simple metadata from a subtitle file.
    For example, count the number of lines.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        lines = content.splitlines()
        return {"lines": len(lines)}
    except Exception as e:
        return {"error": str(e)}

def format_metadata(metadata):
    """
    Format a metadata dictionary into a text header.
    This header can be prepended to CSV or Markdown output.
    """
    lines = []
    for key, value in metadata.items():
        # If the value is a dict, convert to a sorted JSON string.
        if isinstance(value, dict):
            try:
                value = json.dumps(value, indent=2, sort_keys=True)
            except Exception:
                value = str(value)
        else:
            value = str(value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

def add_metadata_header(output_path, metadata_text):
    """
    Prepend the metadata text as a header to the file at output_path.
    Existing file contents are preserved after the header.
    """
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            original = f.read()
        new_content = metadata_text + "\n\n" + original
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        print(f"Error adding metadata header: {e}")

# Example wrapper that extracts metadata based on file type and returns a formatted header.
def extract_metadata_header(source):
    """
    Given a source (URL or local file path), detect the type based on its extension or prefix,
    extract metadata accordingly, and return a formatted header string.
    """
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        meta = get_url_metadata(source)
        header = format_metadata({"Source URL": source, "URL_Metadata": meta})
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]:
            meta = get_media_metadata(source)
            header = format_metadata({"Video File": source, "Media_Metadata": meta})
        elif ext in [".mp3", ".wav", ".flac", ".aac", ".ogg"]:
            meta = get_media_metadata(source)
            header = format_metadata({"Audio File": source, "Media_Metadata": meta})
        elif ext in [".vtt", ".srt", ".ass", ".sub", ".txt"]:
            meta = get_subtitle_metadata(source)
            header = format_metadata({"Subtitle File": source, "Subtitle_Metadata": meta})
        else:
            header = f"Source: {source}"
    return header
