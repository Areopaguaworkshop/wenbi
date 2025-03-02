# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-generativeai==0.8.4",
#     "gradio==5.16.0",
#     "marimo",
#     "pandas==2.2.3",
#     "spacy==3.8.4",
# ]
# ///

import spacy
import os
import re
import pandas as pd
import argparse
from spacy.lang.zh import Chinese
from spacy.lang.en import English

def parse_subtitle(file_path):
    """Parses various subtitle formats (.ass, .sub, .srt, .txt, .vtt) into a DataFrame."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            lines = file.readlines()
    except FileNotFoundError:
        return pd.DataFrame(columns=["Timestamps", "Content"])

    timestamps = []
    contents = []
    current_content = []

    if file_path.lower().endswith(".txt"):
        # ...existing txt handling...
        pass
    else:
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines when not collecting content
            if not line and not current_content:
                i += 1
                continue

            # Check for both timestamp formats with optional whitespace
            timestamp_pattern = r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}"
            if "-->" in line or re.match(timestamp_pattern, line):
                timestamps.append(line)
                i += 1
                current_content = []
                
                # Collect content until next timestamp, handling empty lines
                while i < len(lines):
                    content_line = lines[i].strip()
                    
                    # Skip empty lines but keep collecting
                    if not content_line:
                        i += 1
                        continue
                        
                    # Stop if we hit next timestamp or index number
                    if re.match(timestamp_pattern, content_line) or content_line.isdigit():
                        break
                        
                    current_content.append(content_line)
                    i += 1
                
                # Join collected content, handling empty content case
                if current_content:
                    contents.append(" ".join(current_content))
                else:
                    contents.append("")
            
            elif "Dialogue:" in line:
                # ...existing .ass handling...
                pass
            else:
                i += 1

    return pd.DataFrame({"Timestamps": timestamps, "Content": contents})

def rm_rep(file_path):
    """Removes repeated words/phrases from a file."""
    try:
        vtt_df = parse_subtitle(file_path)
        all_content = "。".join(vtt_df["Content"])
        pattern = (
            r"(([\u4e00-\u9fa5。！？；：“”（）【】《》、]{1,5}))(\s?\1)+"
        )
        return re.sub(pattern, r"\1", all_content)
    except Exception as e:
        return f"An error occurred: {e}"

def segment(file_path, sentence_count=10):
    """Segments text into paragraphs with fixed sentence count."""
    text = rm_rep(file_path)
    
    # Detect language and initialize appropriate model
    if any(char in text for char in "，。？！"):
        nlp = Chinese()
    else:
        nlp = English()

    # Add sentencizer if not present
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    
    # Process sentences into paragraphs
    paragraphs = []
    current_sentences = []
    count = 0
    
    for sent in doc.sents:
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
    
    # Join paragraphs with double newlines
    return "\n\n".join(paragraphs)

def process_subtitle(file_path, sentence_count=10):
    """Process subtitle file and generate markdown and CSV outputs."""
    try:
        base_name, _ = os.path.splitext(file_path)
        csv_file_path = f"{base_name}.csv"
        markdown_file_path = f"{base_name}.md"

        # Export to CSV
        vtt_df = parse_subtitle(file_path)
        vtt_df.to_csv(csv_file_path, index=False, encoding="utf-8")
        print(f"CSV file '{csv_file_path}' created successfully.")

        # Create Markdown with sentence count parameter
        markdown_output = segment(file_path, sentence_count)
        
        # Write markdown to file
        with open(markdown_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"Markdown file '{markdown_file_path}' created successfully.")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert subtitle files to markdown and CSV')
    parser.add_argument('file', help='Path to subtitle file')
    parser.add_argument('-s', '--sentences', type=int, default=10,
                      help='Maximum sentences per paragraph (default: 10)')
    
    args = parser.parse_args()
    process_subtitle(args.file, args.sentences)

if __name__ == "__main__":
    main()
