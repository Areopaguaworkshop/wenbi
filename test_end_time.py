#!/usr/bin/env python3
"""Debug script to test end_time parameter parsing"""

import sys
from wenbi.video_slides import parse_time_to_seconds

# Simulate what happens in cli.py
class Args:
    start_time = ""
    end_time = "01:12:06"
    verbose = True

args = Args()

# Parse time range (default to first hour)
start_time = getattr(args, "start_time", "00:00:00") or "00:00:00"
end_time = getattr(args, "end_time", "01:00:00") or "01:00:00"

print(f"After parsing:")
print(f"  start_time = {repr(start_time)}")
print(f"  end_time = {repr(end_time)}")

# Convert to seconds
start_seconds = parse_time_to_seconds(start_time)
end_seconds = parse_time_to_seconds(end_time)

print(f"\nIn seconds:")
print(f"  start_seconds = {start_seconds}")
print(f"  end_seconds = {end_seconds}")
print(f"  Duration = {end_seconds - start_seconds} seconds = {(end_seconds - start_seconds) / 60} minutes")

# With frame interval of 60 seconds, how many frames?
frame_interval = 60
expected_frames = (end_seconds - start_seconds) // frame_interval
print(f"\nWith frame_interval={frame_interval}s:")
print(f"  Expected ~{expected_frames} frames")
