@echo off
REM ============================================================
REM  Prompt2Clip — Master Setup Script
REM  Creates all 4 isolated conda environments
REM  Target: Windows 11, RTX 5060 Ti (Blackwell sm_120)
REM ============================================================
echo.
echo ============================================================
echo  PROMPT2CLIP — MULTI-ENVIRONMENT SETUP
echo  GPU: RTX 5060 Ti (Blackwell sm_120)
echo  CUDA: 12.8 (cu128) for PyTorch
echo ============================================================
echo.
echo This will create 4 conda environments:
echo   1. clipz_main    — Orchestrator + UI (no GPU)
echo   2. clipz_whisper  — Transcription (faster-whisper GPU)
echo   3. clipz_vision   — YOLO + CLIP (PyTorch GPU)
echo   4. clipz_audio    — Audio Analysis + YAMNet (PyTorch GPU)
echo.
echo Estimated time: 20-40 minutes (depends on network speed)
echo.
pause

REM --- Get script directory ---
set SETUP_DIR=%~dp0

REM === Environment 1: clipz_main ===
echo.
echo ################################################################
echo  [1/4] Setting up clipz_main...
echo ################################################################
call "%SETUP_DIR%setup_clipz_main.bat"

REM === Environment 2: clipz_whisper ===
echo.
echo ################################################################
echo  [2/4] Setting up clipz_whisper...
echo ################################################################
call "%SETUP_DIR%setup_clipz_whisper.bat"

REM === Environment 3: clipz_vision ===
echo.
echo ################################################################
echo  [3/4] Setting up clipz_vision...
echo ################################################################
call "%SETUP_DIR%setup_clipz_vision.bat"

REM === Environment 4: clipz_audio ===
echo.
echo ################################################################
echo  [4/4] Setting up clipz_audio...
echo ################################################################
call "%SETUP_DIR%setup_clipz_audio.bat"

REM === Run health check ===
echo.
echo ################################################################
echo  Running health check on all environments...
echo ################################################################
call "%SETUP_DIR%healthcheck.bat"

echo.
echo ============================================================
echo  ALL ENVIRONMENTS CREATED SUCCESSFULLY!
echo ============================================================
pause
