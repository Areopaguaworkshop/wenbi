# AI subtitle Editor

The final goal of this project is turn the subtitle into a readable written file with slids by using AI, that is Large Language Model (LLM) promts. 

This is my personal project for [GCDFL](https://www.gcdfl.org/). The lectures will generate the cc.vtt or srt files. However, I want to make these files into readable markdown file for Chinese audience. 

## you can try the [demo](https://archive.gcdfl.org/), right now only remove the timestamps and joining the lines. 

# Content

### [Features](https://github.com/Areopaguaworkshop/AI-Subtitle-Editor?tab=readme-ov-file#features)

### [Install](https://github.com/Areopaguaworkshop/AI-Subtitle-Editor?tab=readme-ov-file#install-1)

## Features:

- Remove transcript timestamps and joining the lines. [Done] 

- delete the oral words, such hi, a, ha etc. 

- delete repeated words, phrase and sentences.

- more. 

## Install

### prerequest
- install [rye](https://rye.astral.sh/)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (optional, if you want use the a_w_whisper-cpp.py, recommend for cpu user.)
- [whisper](https://github.com/openai/whisper) (optional, if you want use the a_w-whisper.py)

### first step clone this repository

`
git clone https://github.com/Areopaguaworkshop/AI-Subtitle-Editor.git
` 

### second step 

```
cd AI-Subtitle-Editor

mv pyproject.toml pyproject-bk.toml

rye init 

```

### third step

`
copy whole content of the pyproject-bk.toml into pyproject.toml
` 

Then run 

`rye sync`

### four step

`python src/rm_time_join_line.py`

- if you want local whisper with audio transcribe, you can run 

`python src/a_w-whisper`

or 

`python src/a_w_cpp.py` # you will need to dowload the [whispercpp.bin](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

or, if you have the GROQ_API_KEY, you can first 

`export GROQ_API_KEY="put your GROQ_API_KEY here"` 

then run 

`python src/a_w_groq.py` 

- Then you can access the from http://localhost:7860. 

Enjoy! 

### Buy me a Cofee. 

## License:
AI-Subtitle-Editor is licensed under the Apache License 2.0 found in the [LICENSE](https://github.com/Areopaguaworkshop/AI-Subtitle-Editor/blob/main/license.md) file in the root directory of this repository.

## Citation:
```@article{areopagus/AI-Subtitle-Editor,
  title = {AI-Subtitle-Editor},
  author = {Yuan, Yongjia},
  year = {2024},
}

```

