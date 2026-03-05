@echo off
setlocal

if "%~1"=="" (
  echo Usage: convert_lcc.bat ^<input.lcc^> ^<output-prefix^>
  echo Example: convert_lcc.bat scene.lcc scene_out
  exit /b 1
)

set INPUT=%~1
set OUT=%~2
if "%OUT%"=="" set OUT=output

echo [1/3] LCC -^> HTML
splat-transform -w "%INPUT%" "%OUT%.html"

echo [2/3] LCC -^> PLY
splat-transform -w "%INPUT%" "%OUT%.ply"

echo [3/3] LCC -^> SOG
splat-transform -w "%INPUT%" "%OUT%.sog"

echo Done.
