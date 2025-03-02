import os
import json
import subprocess
from datetime import datetime
import requests
import extruct
from w3lib.html import get_base_url

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

def format_author(author_str):
    """Format author name as 'LastName, FirstName'"""
    if not author_str:
        return "Unknown Author"
    parts = author_str.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return author_str

def get_url_metadata(url):
    """Extract metadata from URL using both extruct and yt-dlp"""
    metadata = {}
    
    # First try to get structured metadata using extruct
    try:
        r = requests.get(url)
        base_url = get_base_url(r.text, r.url)
        structured_data = extruct.extract(
            r.text,
            base_url=base_url,
            uniform=True,
            syntaxes=['json-ld', 'opengraph', 'microdata', 'dublin']
        )
        
        # Try to find author and publication info from structured data
        for syntax in structured_data.values():
            if not syntax:
                continue
            for item in syntax:
                if isinstance(item, dict):
                    # Look for common metadata fields
                    if not metadata.get("Title"):
                        metadata["Title"] = (
                            item.get("headline") or 
                            item.get("name") or 
                            item.get("title")
                        )
                    if not metadata.get("Author"):
                        author = (
                            item.get("author", {}).get("name") or
                            item.get("creator") or
                            item.get("author")
                        )
                        if author:
                            metadata["Author"] = format_author(author)
                    if not metadata.get("Date"):
                        metadata["Date"] = (
                            item.get("datePublished") or
                            item.get("publishedDate") or
                            item.get("date")
                        )
                    if not metadata.get("Publisher"):
                        metadata["Publisher"] = (
                            item.get("publisher", {}).get("name") or
                            item.get("site_name") or
                            item.get("publisher")
                        )
    except Exception as e:
        print(f"Error extracting structured metadata: {e}")

    # If it's a video platform URL, also try yt-dlp
    if yt_dlp and any(platform in url.lower() for platform in ['youtube', 'vimeo', 'dailymotion']):
        try:
            ydl_opts = {
                'skip_download': True,
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Update metadata with yt-dlp info if missing
                if not metadata.get("Title"):
                    metadata["Title"] = info.get('title')
                if not metadata.get("Author"):
                    metadata["Author"] = format_author(info.get('uploader', ''))
                if not metadata.get("Date"):
                    metadata["Date"] = info.get('upload_date', '')[:10]
                if not metadata.get("Publisher"):
                    metadata["Publisher"] = info.get('extractor')
        except Exception as e:
            print(f"Error extracting video metadata: {e}")

    # Ensure we have at least basic metadata
    if not metadata:
        metadata = {
            "Title": "Unknown Title",
            "URL": url
        }
    
    # Always include the source URL
    metadata["URL"] = url
    
    return metadata

def get_media_metadata(file_path):
    """Extract essential metadata from video/audio file"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            '-show_chapters', file_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(output)
        
        # Extract creation date
        format_tags = info.get('format', {}).get('tags', {})
        creation_time = format_tags.get('creation_time', '')
        if creation_time:
            creation_time = creation_time[:4]  # Just get the year
            
        return {
            "Title": format_tags.get('title', os.path.basename(file_path)),
            "Author": format_author(format_tags.get('artist', '')),
            "Year": creation_time,
            "Location": format_tags.get('location', ''),
            "Publisher": format_tags.get('publisher', '')
        }
    except Exception as e:
        return {
            "Title": os.path.basename(file_path),
            "Year": datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y')
        }

def format_metadata(metadata):
    """Format metadata as a clean header with only essential fields"""
    ordered_keys = ["Title", "Author", "Year", "Date", "Location", "Publisher", "URL"]
    lines = []
    
    for key in ordered_keys:
        if key in metadata and metadata[key]:
            lines.append(f"{key}: {metadata[key]}")
    
    return "---\n" + "\n".join(lines) + "\n---"

def add_metadata_header(output_path, metadata_text):
    """Add metadata header to file"""
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            original = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{metadata_text}\n\n{original}")
    except Exception as e:
        print(f"Error adding metadata header: {e}")

def extract_metadata_header(source):
    """Extract and format metadata based on source type"""
    source = source.strip()
    if source.startswith(("http://", "https://")):
        meta = get_url_metadata(source)
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mp3", ".wav", ".flac", ".aac", ".ogg"]:
            meta = get_media_metadata(source)
        else:
            # For subtitle files or unknown types, just use the filename as title
            meta = {
                "Title": os.path.basename(source),
                "Year": datetime.fromtimestamp(os.path.getctime(source)).strftime('%Y')
            }
    
    return format_metadata(meta)
