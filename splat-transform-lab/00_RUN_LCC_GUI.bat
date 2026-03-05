@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
  echo Python not found in PATH.
  echo Install Python 3 and try again.
  pause
  exit /b 1
)

python lcc_to_ply_gui.py
if %errorlevel% neq 0 (
  echo.
  echo GUI exited with error code %errorlevel%.
  pause
)
