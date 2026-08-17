@echo off
REM ============================================================
REM  Prompt2Clip — Health Check Script
REM  Verifies all 4 environments are correctly set up
REM ============================================================
echo.
echo ============================================================
echo  PROMPT2CLIP — ENVIRONMENT HEALTH CHECK
echo ============================================================

set PASS=0
set FAIL=0

REM === clipz_main ===
echo.
echo [CHECK] clipz_main...
conda run -n clipz_main --no-banner python -c "from PyQt5.QtWidgets import QApplication; import numpy; print('  clipz_main    OK | numpy=' + numpy.__version__)"
if errorlevel 1 (
    echo   [FAIL] clipz_main
    set /a FAIL+=1
) else (
    set /a PASS+=1
)

REM === clipz_whisper ===
echo.
echo [CHECK] clipz_whisper...
conda run -n clipz_whisper --no-banner python -c "from faster_whisper import WhisperModel; import numpy; print('  clipz_whisper  OK | numpy=' + numpy.__version__)"
if errorlevel 1 (
    echo   [FAIL] clipz_whisper
    set /a FAIL+=1
) else (
    set /a PASS+=1
)

REM === clipz_vision ===
echo.
echo [CHECK] clipz_vision...
conda run -n clipz_vision --no-banner python -c "import torch; from ultralytics import YOLO; from transformers import CLIPProcessor; import numpy; print('  clipz_vision   OK | numpy=' + numpy.__version__ + ' | CUDA=' + str(torch.cuda.is_available()) + ' | GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))"
if errorlevel 1 (
    echo   [FAIL] clipz_vision
    set /a FAIL+=1
) else (
    set /a PASS+=1
)

REM === clipz_audio ===
echo.
echo [CHECK] clipz_audio...
conda run -n clipz_audio --no-banner python -c "import torch; import librosa; import numpy; print('  clipz_audio    OK | numpy=' + numpy.__version__ + ' | CUDA=' + str(torch.cuda.is_available()) + ' | GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))"
if errorlevel 1 (
    echo   [FAIL] clipz_audio
    set /a FAIL+=1
) else (
    set /a PASS+=1
)

echo.
echo ============================================================
echo  RESULTS: %PASS% passed, %FAIL% failed (out of 4)
echo ============================================================

if %FAIL% GTR 0 (
    echo  [WARNING] Some environments failed. Check errors above.
) else (
    echo  [SUCCESS] All environments are healthy!
)
echo.
pause
