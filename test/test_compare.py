import os
import argparse
from docx import Document
from datetime import datetime
from redlines import Redlines

def docx_to_text(path):
    """Extract text from docx file"""
    doc = Document(path)
    return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())

def main():
    parser = argparse.ArgumentParser(description='Compare two docx files and output diff results')
    parser.add_argument('original', help='Path to original docx file')
    parser.add_argument('modified', help='Path to modified docx file')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory for diff files (default: current directory)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract text from both documents
    original_text = docx_to_text(args.original)
    modified_text = docx_to_text(args.modified)

    # Generate diff using Redlines
    diff = Redlines(original_text, modified_text)
    marked_text = diff.output_markdown

    # Save as markdown
    base_name = os.path.splitext(os.path.basename(args.original))[0]
    md_out = os.path.join(args.output_dir, f"{base_name}_compare.md")
    
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write(f"# Document Comparison\n\n")
        f.write(f"**Original:** {args.original}\n")
        f.write(f"**Modified:** {args.modified}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Changes\n\n")
        f.write(marked_text)  # Redlines provides markdown with ~~deletions~~ and {+additions+}

    print(f"Generated redline markdown: {md_out}")

if __name__ == '__main__':
    main()