# Wenbi

A simple tool to make the video, audio, subtitle and video-url (especially youtube) content into a written markdown files with the ability to rewritten the oral expression into written ones, or translating the content into a target language by using LLM. 

Initally, this porject is just serving to my website [GCDFL](https://www.gcdfl.org/). We do a service to turn its lectures into a written files for easier further editing. 

Note: LLM can make mistakes and cannot be fully trusted. LLM can only be used for preliminary processing of data, some elementary work, and in this sense, LLM does greatly improve editing efficiency. 


### you can try the [demo](https://archive.gcdfl.org/), right now only remove the timestamps and joining the lines. 

## Features

- Accept most popular audio, video, subtitle files and url--mainly using yt-dlp as input. 

- Editing the files by using LLM to rewriting and translating the content into a readable written markdown files. 

- Support input with multiple languages.

- offer an commandline and gradio GUI with multiple options for further personal setting

## Install

### prerequest
- install [rye](https://rye.astral.sh/)

### first step clone this repository

`
git clone https://github.com/Areopaguaworkshop/wenbi.git
` 

### second step 

```
cd wenbi 

mv pyproject.toml pyproject-bk.toml

rye init 

```

### third step

`
copy whole content of the pyproject-bk.toml into pyproject.toml
` 

Then run

`source .venv/bin/activate` 

`rye pin 3.12` 

`rye sync`

### four step

You can choose commandline or webGUI through gradio.

- gradio

`python main.py`

Then go to http://localhost:7860. 

- commandline 

'python cli.py --help'

usage: cli.py [-h] [--language LANGUAGE] [--llm LLM] [--multi-language] [--translate-lang TRANSLATE_LANG] [--output-dir OUTPUT_DIR] input

wenbi: Convert video, audio, url or subtitle files to CSV and written Markdown outputs.

positional arguments:
  input                 Path to input file or URL

options:
  -h, --help            show this help message and exit
  --language LANGUAGE   Transcribe Language (optional)
  --llm LLM             Large Language Model identifier (optional)
  --multi-language      Enable multi-language processing (default: False)
  --translate-lang TRANSLATE_LANG
                        Target translation language (default: Chinese)
  --output-dir OUTPUT_DIR
                        Output directory (optional)



Note: if you want to convert the audio file of multi-language, you should set multi-language as True. for commandline is --multi-language. you nedd a HUGGINGFACE_TOKEN in you environment. by `export HUGGINGFACE_TOKEN="you HUGGINGFACE_TOKEN here"`. 


Enjoy! 

### Buy me a Cofee. 

## License:
AI-Subtitle-Editor is licensed under the Apache License 2.0 found in the [LICENSE](https://github.com/Areopaguaworkshop/AI-Subtitle-Editor/blob/main/license.md) file in the root directory of this repository.

## Citation:
```@article{areopagus/wenbi
  title = {wenbi},
  author = {Yuan, Yongjia},
  year = {2024},
}

```

