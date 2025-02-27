# Wenbi

A simple tool to make the video, audio, subtitle and video-url (especially youtube) content into a written markdown files with the ability to rewritten the oral expression into written ones, or translating the content into a target language by using LLM. 

Initally, this porject is just serving to my website [GCDFL](https://www.gcdfl.org/). We do a service to turn its lectures into a written files for easier further editing. 

Wenbi is a Chinese name 文笔, meaning a good writting. 

:warning: LLM can make mistakes (of course, human make mistakes too) and cannot be fully trusted. LLM can only be used for preliminary processing of data, some elementary work, and in this sense, LLM does greatly improve editing efficiency. 


### you can try the [demo](https://archive.gcdfl.org/). 

## Features

- **100% Open source and totally free of use**. I love open source, I learned a lot from it. 

- :100: Accept most popular audio, video, subtitle files and url--mainly using yt-dlp as input. 

- :100: Editing the files by using LLM to rewriting and translating the content into a readable written markdown files. 

- :100: Support input with multiple languages.

- :100: offer an commandline and gradio GUI with multiple options for further personal settings 

- :100: The support provider is Ollama, you can use most of the models from ollama. 

- :construction: other provider supporting, such as OpenAi, Google and Others. 

- :construction: fine-tuned model for specific job, for example for my personal project from [GCDFL](https://www.gcdfl.org/), introducing the eastern churches to Chinese audience through academic lectures; [CTCFOL](https://www.ctcfol.org/), The Chinese Translation of Church Fathers from Original Languages. 

:warning: the default translating and rewritten language are Chinese, however, you can choose other Ollama models, and languages through our commandline options. 

## Install
- You can install through pip (or other tools as uv or rye) and from source. 

### prerequest
- Install [Ollama](https://ollama.com/) and dowload a model. The default model for this project is qwen2.5. 

### Install through pip

1. build a virtue environment through [uv](https://docs.astral.sh/uv/guides/install-python/)--recommened or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

-for uv: `uv venv --python 3.12`

2. `uv pip install wenbi` or `uv add wenbi`

:warning: due to some package issues, it is better to install llvmlite and numba first by `uv add llvmlite numba` , then install wenbi. 

After install, you can simply using wenbi commandline. if you want a gradio GUI, you can run `wenbi --gui`. the commandline options as follow: 
```
wenbi: Convert video, audio, URL, or subtitle files to CSV and Markdown outputs.

positional arguments:
  input                 Path to input file or URL

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Output directory (optional)
  --gui                 Launch Gradio GUI
  --rewrite-llm REWRITE_LLM
                        Rewrite LLM model identifier (optional)
  --translate-llm TRANSLATE_LLM
                        Translation LLM model identifier (optional)
  --transcribe-lang TRANSCRIBE_LANG
                        Transcribe language (optional)
  --translate-lang TRANSLATE_LANG
                        Target translation language (default: Chinese)
  --rewrite-lang REWRITE_LANG
                        Target language for rewriting (default: Chinese)
  --multi-language      Enable multi-language processing
  --chunk-length CHUNK_LENGTH
                        the chunk of Number of sentences per paragraph for llm to tranlsate or rewrite.
                        (default: 8)
  --max-tokens MAX_TOKENS
                        Maximum tokens for LLM output (default: 50000)
  --timeout TIMEOUT     LLM request timeout in seconds (default: 3600)
  --temperature TEMPERATURE
                        LLM temperature parameter (default: 0.1)
  --base-url BASE_URL   Base URL for LLM API (default: http://localhost:11434)

```

### Install from Source

1. install [rye](https://rye.astral.sh/)

2. `
git clone https://github.com/Areopaguaworkshop/wenbi.git
` 
3. 
```
cd wenbi 

mv pyproject.toml pyproject-bk.toml

rye init 

```

4. `
copy whole content of the pyproject-bk.toml into pyproject.toml
` 
5. 
`source .venv/bin/activate` 

`rye pin 3.12` 

`rye sync`

:warning: Again, you may face some package issues. you can either install the depencies one by one in the pyproject-bk.toml, or  add llvmlite and numba first by `rye add llvmlite numba` , then install wenbi. 

6. You can choose commandline or webGUI through gradio.

- gradio

`python wenbi/main.py`

Then go to http://localhost:7860. 

- commandline 

'python wenbi/cli.py --help'

:warning: if you want to convert the audio file of multi-language, you should set multi-language as True. for commandline is --multi-language. you nedd a HUGGINGFACE_TOKEN in you environment. by `export HUGGINGFACE_TOKEN="you HUGGINGFACE_TOKEN here"`. 


Enjoy! 

### Buy me a [Cofee](https://www.gcdfl.org/donate/). 

## License:
AI-Subtitle-Editor is licensed under the Apache License 2.0 found in the [LICENSE](https://github.com/Areopaguaworkshop/AI-Subtitle-Editor/blob/main/license.md) file in the root directory of this repository.

## Citation:
```@article{Areopaguaworkshop/wenbi
  title = {wenbi},
  author = {Ephrem, Yuan},
  year = {2024},
}

```

