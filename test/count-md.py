#!/usr/bin/env python3
import re
import argparse
import os

def count_words(text):
    """Count Chinese characters and English words in text."""
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`.*?`', '', text)
    
    # Count Chinese characters (including punctuation)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_count = len(chinese_chars)
    
    # Count English words
    # First remove Chinese characters and punctuation
    english_only = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    english_only = re.sub(r'[^\w\s]', ' ', english_only)
    english_words = english_only.split()
    english_count = len(english_words)
    
    return chinese_count, english_count

def process_markdown_file(file_path):
    """Process a markdown file and count words."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        chinese_count, english_count = count_words(content)
        
        print(f"\nStatistics for {os.path.basename(file_path)}:")
        print(f"Chinese characters: {chinese_count}")
        print(f"English words: {english_count}")
        print(f"Total: {chinese_count + english_count}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Count Chinese characters and English words in markdown files."
    )
    parser.add_argument(
        "input", help="Path to markdown file or directory containing markdown files"
    )
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        if args.input.endswith(('.md', '.markdown')):
            process_markdown_file(args.input)
        else:
            print("Error: Input file must be a markdown file (.md or .markdown)")
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith(('.md', '.markdown')):
                    file_path = os.path.join(root, file)
                    process_markdown_file(file_path)
    else:
        print("Error: Input path does not exist")

if __name__ == "__main__":
    main()
