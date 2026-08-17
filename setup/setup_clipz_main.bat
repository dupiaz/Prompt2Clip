@echo off
REM ============================================================
REM  clipz_main — Orchestrator + UI Environment
REM  Lightweight: PyQt5, scipy, matplotlib, openai (NO GPU)
REM ============================================================
echo.
echo ============================================================
echo  SETTING UP: clipz_main (Orchestrator + UI)
echo ============================================================

REM --- Create conda env ---
echo [1/4] Creating conda environment: clipz_main (Python 3.10)...
conda create -n clipz_main python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)

REM --- Activate ---
call conda activate clipz_main

REM --- Upgrade pip ---
echo [2/4] Upgrading pip, setuptools, wheel...
pip install --upgrade pip setuptools wheel

REM --- Install FFmpeg via conda (required by moviepy/ffmpeg-python) ---
echo [3/4] Installing FFmpeg binary...
conda install -c conda-forge ffmpeg -y

REM --- Install Python packages ---
echo [4/4] Installing Python dependencies...
pip install "numpy>=1.23.5,<2.0.0"
pip install scipy matplotlib tqdm
pip install requests python-dotenv openai
pip install PyQt5
pip install moviepy ffmpeg-python
pip install Pillow

REM --- Final numpy lock ---
pip install "numpy>=1.23.5,<2.0.0"

echo.
echo ============================================================
echo  clipz_main SETUP COMPLETE!
echo  Verify: conda run -n clipz_main python -c "from PyQt5.QtWidgets import QApplication; print('OK')"
echo ============================================================
pause
