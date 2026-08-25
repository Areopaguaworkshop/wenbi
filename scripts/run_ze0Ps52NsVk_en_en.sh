#!/usr/bin/env bash
set -euo pipefail

cd /home/ajiap/project/wenbi

# Load keys from both .env files (wenbi has GLADIA, videoedit has DEEPL)
set -a
source /home/ajiap/project/wenbi/.env
source /home/ajiap/project/videoedit/.env
set +a

.venv/bin/wenbi en-en /home/ajiap/project/videoedit/downloads/ze0Ps52NsVk.mp4 \
  --asr-provider gladia \
  --lang Chinese \
  --speaker-count 2 \
  --llm ollama/glm-5.2:cloud \
  --save-json \
  -v
