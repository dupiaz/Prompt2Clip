@echo off
REM ============================================================
REM  clipz_whisper — Transcription GPU Environment
REM  faster-whisper + CTranslate2 on CUDA (Blackwell sm_120)
REM ============================================================
echo.
echo ============================================================
echo  SETTING UP: clipz_whisper (Transcription GPU)
echo ============================================================

REM --- Create conda env ---
echo [1/4] Creating conda environment: clipz_whisper (Python 3.10)...
conda create -n clipz_whisper python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)

REM --- Activate ---
call conda activate clipz_whisper

REM --- Upgrade pip ---
echo [2/4] Upgrading pip, setuptools, wheel...
pip install --upgrade pip setuptools wheel

REM --- Install numpy first (conflict prevention) ---
echo [3/4] Pinning numpy...
pip install "numpy>=1.23.5,<2.0.0"

REM --- Install faster-whisper + CTranslate2 (latest for Blackwell fix) ---
echo [4/4] Installing faster-whisper + CTranslate2...
pip install ctranslate2
pip install faster-whisper
pip install python-dotenv

REM --- Install NVIDIA CUDA libraries for CTranslate2 ---
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

REM --- Final numpy lock ---
pip install "numpy>=1.23.5,<2.0.0"

echo.
echo ============================================================
echo  clipz_whisper SETUP COMPLETE!
echo  Verify: conda run -n clipz_whisper python -c "from faster_whisper import WhisperModel; print('OK')"
echo ============================================================
pause
