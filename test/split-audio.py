from pydub import AudioSegment
import os
import math

def split_audio(audio_path, num_parts=10):
    """
    Split an audio file into equal parts.
    
    Args:
        audio_path (str): Path to the audio file
        num_parts (int): Number of parts to split into (default: 10)
    
    Returns:
        list: List of paths to the split audio files
    """
    # Get file extension and create output directory
    base_path, ext = os.path.splitext(audio_path)
    output_dir = os.path.dirname(base_path)
    
    # Load audio file
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    # Calculate length of each part (in milliseconds)
    total_length = len(audio)
    part_length = math.ceil(total_length / num_parts)
    
    # Split and export parts
    output_files = []
    for i in range(num_parts):
        # Calculate start and end times
        start_time = i * part_length
        end_time = min((i + 1) * part_length, total_length)
        
        # Extract the segment
        segment = audio[start_time:end_time]
        
        # Generate output filename
        output_path = f"{base_path}_part{i+1:02d}.wav"
        
        # Export segment
        try:
            segment.export(output_path, format="wav")
            output_files.append(output_path)
            print(f"Created part {i+1}: {output_path}")
        except Exception as e:
            print(f"Error exporting part {i+1}: {e}")
    
    return output_files

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python split-audio.py <audio_file> [num_parts]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    num_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Splitting {audio_file} into {num_parts} parts...")
    output_files = split_audio(audio_file, num_parts)
    print(f"Created {len(output_files)} parts successfully")
