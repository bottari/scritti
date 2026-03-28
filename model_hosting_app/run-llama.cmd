@echo off
setlocal

set "BASE_MODEL_NAME=meta-llama/Llama-3.1-8B"
set "ADAPTER_PATH=D:\models\llama31-8b-poetry-mercury-qlora-4bit-March-21-2026v1"

python "%~dp0app.py"
