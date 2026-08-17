@echo off
REM ============================================================
REM  clipz_vision — Vision GPU Environment
REM  PyTorch cu128 + YOLO + CLIP + dlib (Blackwell sm_120)
REM ============================================================
echo.
echo ============================================================
echo  SETTING UP: clipz_vision (Vision GPU)
echo ============================================================

REM --- Create conda env ---
echo [1/6] Creating conda environment: clipz_vision (Python 3.10)...
conda create -n clipz_vision python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)

REM --- Activate ---
call conda activate clipz_vision

REM --- Upgrade pip ---
echo [2/6] Upgrading pip, setuptools, wheel...
pip install --upgrade pip setuptools wheel

REM --- Pin numpy FIRST ---
echo [3/6] Pinning numpy...
pip install "numpy>=1.23.5,<2.0.0"

REM --- CUDA LOCK: PyTorch cu128 for Blackwell sm_120 ---
echo [4/6] Installing PyTorch + TorchVision (CUDA 12.8 for Blackwell)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed!
    pause
    exit /b 1
)

REM --- Vision stack ---
echo [5/6] Installing Vision dependencies (YOLO, CLIP, OpenCV, dlib)...
pip install ultralytics
pip install transformers
pip install opencv-python Pillow
pip install scipy matplotlib tqdm
pip install python-dotenv

REM --- dlib via conda-forge (avoids C++ build issues on Windows) ---
echo [6/6] Installing dlib via conda-forge...
conda install -c conda-forge dlib -y

REM --- Final numpy lock ---
pip install "numpy>=1.23.5,<2.0.0"

echo.
echo ============================================================
echo  clipz_vision SETUP COMPLETE!
echo  Verify: conda run -n clipz_vision python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Arch:', torch.cuda.get_arch_list())"
echo ============================================================
pause
