@echo off
REM One-time setup. Double-click or run from the project folder (e.g. D:\finance_agent).
cd /d "%~dp0"
echo === Creating virtual environment ===
python -m venv .venv
if errorlevel 1 (echo Python not found. Install Python 3.10+ first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo pip install failed. Check your internet connection. & pause & exit /b 1)
echo === Loading sample data ===
python seed_data.py --load --reset
echo === Checking Ollama + model (Ollama app must be running) ===
python check_setup.py
pause
