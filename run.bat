@echo off
rem Double-click this file in Explorer to start tensorMerger.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 merge_safetensors.py) || (python merge_safetensors.py)
echo.
pause
