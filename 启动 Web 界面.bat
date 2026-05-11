@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -q gradio 2>nul

echo.
echo Starting Web interface...
echo.
python app.py
