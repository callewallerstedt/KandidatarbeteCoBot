@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating venv...
  py -3.11 -m venv .venv || goto :err
)
echo Installing requirements...
.\.venv\Scripts\python.exe -m pip install --upgrade pip || goto :err
.\.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
echo Launching GUI...
.\.venv\Scripts\python.exe app.py
pause
exit /b 0
:err
echo Failed. Check output above.
pause
exit /b 1
