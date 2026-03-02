@echo off
setlocal

cd /d "%~dp0"

echo ======================================
echo Unity -> Python dual camera receiver
echo ======================================
echo.

where py >nul 2>nul
if errorlevel 1 goto :NO_PY

if not exist ".venv" (
  echo [1/4] Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :VENV_FAIL
)

echo [2/4] Installing/updating dependencies in .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :PIP_FAIL

echo [3/4] Starting receiver...
echo Press Q or ESC in the OpenCV window to quit.
echo.
".venv\Scripts\python.exe" -u receive_two_cams.py

echo.
echo Receiver stopped.
pause
endlocal
exit /b 0

:NO_PY
echo Python launcher py not found in PATH.
echo Install Python 3 and make sure py works in cmd.
pause
endlocal
exit /b 1

:VENV_FAIL
echo Failed to create virtual environment.
pause
endlocal
exit /b 1

:PIP_FAIL
echo Failed to install dependencies.
pause
endlocal
exit /b 1
