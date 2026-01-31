#!/usr/bin/env python3

import sys
sys.path.append('/home/ajiap/project/wenbi')

from wenbi.cli import handle_rewrite_command

class Args:
    def __init__(self):
        self.input = "https://www.youtube.com/watch?v=rL9_N98yrLU"
        self.config = ""
        self.output_dir = ""
        self.llm = ""
        self.chunk_length = 20
        self.max_tokens = 130000
        self.timeout = 3600
        self.temperature = 0.1
        self.lang = "Chinese"
        self.subcommand = "rewrite"
        self.transcribe_model = "large-v3"
        self.multi_language = False
        self.transcribe_lang = ""
        self.output_wav = ""
        self.cite_timestamps = True
        self.verbose = True
        self.start_time = ""
        self.end_time = ""

if __name__ == "__main__":
    args = Args()
    result = handle_rewrite_command(args)
    print("Result:", result)