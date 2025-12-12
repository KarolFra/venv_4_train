@echo off
setlocal
REM One-click launcher for cycling YOLO models on laptop camera

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=python"
)

set "M1=%ROOT%runs\train\train3_connector\weights\best.pt"
set "M2=%ROOT%runs\train\train9_KICAD\weights\best.pt"
set "M3=%ROOT%runs\train\train3\weights\best.pt"

echo Using Python: %PY%
echo Model 1: %M1%
echo Model 2: %M2%
echo Model 3: %M3%

"%PY%" "%ROOT%camera_demo.py" ^
  --models "%M1%" "%M2%" "%M3%" ^
  --switch-sec 10 ^
  --source 0 ^
  --imgsz 640 ^
  --conf 0.25 ^
  --device cpu %*

echo.
echo Finished. Press any key to exit.
pause >nul
endlocal

