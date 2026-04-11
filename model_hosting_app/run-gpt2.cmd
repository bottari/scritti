@echo off
setlocal

set "SCRITTI_BASE_MODEL_NAME=D:\models\choice-models\gpt2-finetuned-poetry-mercury-04\final_model"
set "SCRITTI_ADAPTER_PATH= "
set "BASE_MODEL_NAME="
set "ADAPTER_PATH="

python "%~dp0app.py"
