import sys
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import argparse


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
    video_path, start_time_sec, end_time_sec, output_path, 
    title_text=None, duration=65, font_size=70, text_color="red"
):
    """
    Extracts a video segment with customizable text overlay options.
    """
    with VideoFileClip(video_path) as clip:
        subclip = clip.subclipped(start_time_sec, end_time_sec)

        if title_text:
            # Create TextClip with customizable parameters
            title_clip = TextClip(
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                title_text,
                font_size=font_size,
                color=text_color,
                duration=duration,
                text_align="center",
                margin=(4, 4),
                bg_color="black",
                transparent=True,
            )
            final_clip = CompositeVideoClip([subclip, title_clip])
        else:
            final_clip = subclip

        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def main():
    parser = argparse.ArgumentParser(description='Extract video segment with optional title overlay')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('start_time', help='Start time (hh:mm:ss)')
    parser.add_argument('end_time', help='End time (hh:mm:ss)')
    parser.add_argument('output_video', help='Output video file path')
    parser.add_argument('--title', help='Optional title text overlay')
    parser.add_argument('--duration', type=int, default=65, help='Title overlay duration (seconds)')
    parser.add_argument('--font-size', type=int, default=70, help='Title font size')
    parser.add_argument('--color', default='red', help='Title text color')

    args = parser.parse_args()

    try:
        start_time_sec = parse_time(args.start_time)
        end_time_sec = parse_time(args.end_time)
    except ValueError as e:
        print(f"Error parsing time: {e}")
        sys.exit(1)

    extract_segment(
        args.input_video,
        start_time_sec,
        end_time_sec,
        args.output_video,
        args.title,
        duration=args.duration,
        font_size=args.font_size,
        text_color=args.color
    )

if __name__ == "__main__":
    main()
