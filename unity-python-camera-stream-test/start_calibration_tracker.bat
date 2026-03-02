@echo off
setlocal

cd /d "%~dp0"

echo =============================================
echo Unity dual cam calibration + 3D tracker start
echo =============================================
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

echo [2/4] Installing dependencies in .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo [3/4] Starting calibration/tracker UI...
echo OpenCV windows + Tkinter panel will open.
echo Press Q or ESC in an OpenCV window to quit.
echo.
".venv\Scripts\python.exe" -u calibrate_and_track_3d.py

echo.
echo App stopped.
pause
endlocal
