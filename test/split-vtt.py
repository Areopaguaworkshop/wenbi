import os

def split_vtt(vtt_path, num_parts=10):
    """
    Split a VTT file into a specified number of equal parts by cues.
    
    Args:
        vtt_path (str): Path to the input VTT file.
        num_parts (int): Number of parts to split into (default: 10).
    
    Returns:
        list: List of paths to the generated split VTT files.
    """
    # Read the entire VTT file lines
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Assume first non-empty line is header ("WEBVTT")
    header = ""
    cues = []
    current_cue = []

    # Parse header and cues separated by blank lines.
    # Skip any empty lines at the beginning.
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not header_found and stripped:
            header = line  # Save header (should be "WEBVTT")
            header_found = True
            continue
        # If header is read, cues start after an empty line.
        if header_found:
            if stripped == "":
                if current_cue:
                    cues.append("".join(current_cue))
                    current_cue = []
            else:
                current_cue.append(line)
    # Append last cue if any
    if current_cue:
        cues.append("".join(current_cue))
    
    # Determine number of cues per part (ceiling division)
    total_cues = len(cues)
    cues_per_part = -(-total_cues // num_parts)  # ceiling division
    
    output_files = []
    base_name = os.path.splitext(os.path.basename(vtt_path))[0]
    output_dir = os.path.dirname(vtt_path)
    
    for i in range(num_parts):
        part_cues = cues[i * cues_per_part : (i + 1) * cues_per_part]
        if not part_cues:
            break  # no more cues to split
        part_filename = os.path.join(output_dir, f"{base_name}_part{i+1:02d}.vtt")
        with open(part_filename, "w", encoding="utf-8") as f_out:
            # Write header and a blank line before cues
            f_out.write(header.rstrip() + "\n\n")
            # Write cues. Maintain original cue formatting.
            for cue in part_cues:
                f_out.write(cue.strip() + "\n\n")
        output_files.append(part_filename)
        print(f"Created: {part_filename}")
    
    return output_files

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split-vtt.py <vtt_file> [num_parts]")
        sys.exit(1)
    vtt_file = sys.argv[1]
    num_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    parts = split_vtt(vtt_file, num_parts)
    print(f"Split into {len(parts)} files.")
