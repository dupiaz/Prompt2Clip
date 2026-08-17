@echo off
REM ============================================================
REM  clipz_audio — Audio Analysis GPU Environment
REM  PyTorch cu128 + librosa + torch-vggish-yamnet (Blackwell)
REM  NOTE: TensorFlow replaced by PyTorch YAMNet
REM ============================================================
echo.
echo ============================================================
echo  SETTING UP: clipz_audio (Audio Analysis GPU)
echo ============================================================

REM --- Create conda env ---
echo [1/5] Creating conda environment: clipz_audio (Python 3.10)...
conda create -n clipz_audio python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)

REM --- Activate ---
call conda activate clipz_audio

REM --- Upgrade pip ---
echo [2/5] Upgrading pip, setuptools, wheel...
pip install --upgrade pip setuptools wheel

REM --- Pin numpy FIRST ---
echo [3/5] Pinning numpy...
pip install "numpy>=1.23.5,<2.0.0"

REM --- CUDA LOCK: PyTorch cu128 for Blackwell sm_120 ---
echo [4/5] Installing PyTorch + TorchAudio (CUDA 12.8 for Blackwell)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)

REM --- Audio stack ---
echo [5/5] Installing Audio dependencies...
REM Core audio processing
pip install librosa soundfile
pip install scipy

REM YAMNet via PyTorch (replaces TensorFlow + TF-Hub)
pip install torch-vggish-yamnet

REM Additional audio analysis
pip install praat-parselmouth ruptures

REM Utilities
pip install matplotlib tqdm python-dotenv

REM --- Final numpy lock ---
pip install "numpy>=1.23.5,<2.0.0"

echo.
echo ============================================================
echo  clipz_audio SETUP COMPLETE!
echo  Verify: conda run -n clipz_audio python -c "import torch; print('CUDA:', torch.cuda.is_available()); import librosa; print('librosa OK')"
echo ============================================================
pause
