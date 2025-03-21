import sys
from moviepy.video.io.VideoFileClip import VideoFileClip


def parse_time(time_str):
    """
    Converts a time string in the format hh:mm:ss into seconds.
    """
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time format must be hh:mm:ss")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_segment(video_path, start_time_sec, end_time_sec, output_path):
    """
    Extracts a video segment from start_time_sec to end_time_sec.
    """
    with VideoFileClip(video_path) as clip:
        subclip = clip.subclipped(start_time_sec, end_time_sec)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python extract_segment.py <input_video> <start_time (hh:mm:ss)> <end_time (hh:mm:ss)> <output_video>"
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

    extract_segment(video_path, start_time_sec, end_time_sec, output_path)
