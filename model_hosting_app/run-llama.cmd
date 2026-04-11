@echo off
setlocal

set "SCRITTI_BASE_MODEL_NAME=meta-llama/Llama-3.1-8B"
set "SCRITTI_ADAPTER_PATH=D:\models\choice-models\llama3-8b-poetry-mercury-26-qlora-8bit-019\final_model"
set "BASE_MODEL_NAME="
set "ADAPTER_PATH="

python "%~dp0app.py"
