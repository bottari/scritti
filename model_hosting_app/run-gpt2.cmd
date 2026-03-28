@echo off
setlocal

set "BASE_MODEL_NAME=C:\Users\micha\Desktop\projects\scritti\artifacts\gpt2-whitman\final_model"
set "ADAPTER_PATH= "

python "%~dp0app.py"
