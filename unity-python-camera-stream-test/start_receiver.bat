@echo off
setlocal

cd /d "%~dp0"

echo ======================================
echo Unity -> Python dual camera receiver
echo ======================================
echo.

where py >nul 2>nul
if %errorlevel% neq 0 (
  echo Python launcher (py) not found in PATH.
  echo Install Python 3 and make sure "py" works in cmd.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo [1/4] Creating virtual environment...
  py -3 -m venv .venv
  if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo [2/4] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
  echo Failed to activate virtual environment.
  pause
  exit /b 1
)

echo [3/4] Installing/updating dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo [4/4] Starting receiver...
echo Press Q or ESC in the OpenCV window to quit.
echo.
python receive_two_cams.py

echo.
echo Receiver stopped.
pause
endlocal
