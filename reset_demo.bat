@echo off
REM Resets the database to fresh sample data so every demo / video starts from the same numbers.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python seed_data.py --load --reset
pause
