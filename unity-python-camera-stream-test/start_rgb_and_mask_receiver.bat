@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 goto :NO_PY

if not exist ".venv" (
  py -3 -m venv .venv
  if errorlevel 1 goto :VENV_FAIL
)

".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :PIP_FAIL

".venv\Scripts\python.exe" -u receive_rgb_and_mask.py

pause
endlocal
exit /b 0

:NO_PY
echo Python launcher py not found in PATH.
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
