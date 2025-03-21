#!/usr/bin/env python
import sys
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


def parse_time(time_str):
    """
    Converts a time string in the format hh:mm:ss into seconds.
    """
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time format must be hh:mm:ss")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_segment(
    video_path, start_time_sec, end_time_sec, output_path, title_text=None
):
    """
    Extracts a video segment from start_time_sec to end_time_sec and, if provided,
    overlays a title on the segment.
    """
    with VideoFileClip(video_path) as clip:
        subclip = clip.subclipped(start_time_sec, end_time_sec)

        if title_text:
            # Create a TextClip for the title. Adjust fontsize, color, and position as needed.
            title_clip = TextClip(
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                title_text,
                font_size=70,
                color="red",
                duration=65,
                text_align="center",
                margin=(4, 4),
                bg_color="black",
                transparent=False,
            )
            # Composite the text clip over the subclip.
            final_clip = CompositeVideoClip([subclip, title_clip])
        else:
            final_clip = subclip

        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    if len(sys.argv) not in [5, 6]:
        print(
            "Usage: python extract_segment.py <input_video> <start_time (hh:mm:ss)> <end_time (hh:mm:ss)> <output_video> [title]"
        )
        print(
            'Example: python extract_segment.py myvideo.mp4 00:01:30 00:02:45 output.mp4 "My Segment Title"'
        )
        sys.exit(1)

    video_path = sys.argv[1]
    try:
        start_time_sec = parse_time(sys.argv[2])
        end_time_sec = parse_time(sys.argv[3])
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)
    output_path = sys.argv[4]
    title_text = sys.argv[5] if len(sys.argv) == 6 else None

    extract_segment(video_path, start_time_sec, end_time_sec, output_path, title_text)
