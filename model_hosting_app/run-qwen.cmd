@echo off
setlocal

set "BASE_MODEL_NAME=Qwen/Qwen3.5-0.8B"
set "ADAPTER_PATH=D:\models\qwen3-5-0-8b-poetry-mercury-qlora-8bit-March21-002"

python "%~dp0app.py"
