@echo off
setlocal

set "SCRITTI_BASE_MODEL_NAME=Qwen/Qwen3.5-0.8B"
set "SCRITTI_ADAPTER_PATH=D:\models\qwen3-5-0-8b-poetry-mercury-qlora-8bit-March21-002"
set "BASE_MODEL_NAME="
set "ADAPTER_PATH="

python "%~dp0app.py"
