@echo off
REM One-click PTL smoke test for the torch-xpu 2.12.1 fix (Windows).
REM Creates .venv-backend, installs torch 2.12.1+xpu + backend deps, then
REM trains YOLO11n on XPU (CI settings) and exports the FP16 OpenVINO IR.
setlocal
cd /d "%~dp0"

if not exist .venv-backend\Scripts\python.exe (
  echo [smoke] creating venv
  py -3.12 -m venv .venv-backend 2>nul || python -m venv .venv-backend || goto :fail
)
set PY=.venv-backend\Scripts\python.exe

echo [smoke] installing torch 2.12.1+xpu
%PY% -m pip install -q --upgrade pip wheel || goto :fail
%PY% -m pip install -q --index-url https://download.pytorch.org/whl/xpu torch==2.12.1 torchvision==0.27.1 || goto :fail
echo [smoke] installing backend requirements
%PY% -m pip install -q -r backend\requirements.txt || goto :fail

set APP=%CD%
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=15
set NTRAIN=%2
if "%NTRAIN%"=="" set NTRAIN=1600
echo [smoke] training %EPOCHS% epochs on %NTRAIN% synthetic images
%PY% smoke_win.py %EPOCHS% %NTRAIN% || goto :fail
echo.
echo [smoke] PASS
exit /b 0

:fail
echo.
echo [smoke] FAIL - send the last 30 lines above
exit /b 1
