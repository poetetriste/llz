@echo off
chcp 65001 >nul
rem Cross-machine determinism test for llz.
cd /d %~dp0

set "PY="

rem 1) py launcher (never a Store stub)
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"

rem 2) real python (runtime check, Store stub fails here)
if defined PY goto havepy
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"

:havepy
if not defined PY goto install
echo Using: %PY%
goto venv

:install
echo Python not found. Installing Python 3.12 via winget...
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY%" goto venv
set "PY=C:\Program Files\Python312\python.exe"
if exist "%PY%" goto venv
set "PY="
echo PYTHON NOT FOUND. Install from python.org (check "Add python.exe to PATH") and rerun.
pause
exit /b 1

:venv
if not exist .venv %PY% -m venv .venv
set "VPY=.venv\Scripts\python.exe"

echo Installing torch CPU + transformers (first run downloads ~400 MB)...
"%VPY%" -m pip install --quiet torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
"%VPY%" -m pip install --quiet transformers==5.22.1 numpy==2.5.4

echo === TEST START ===
"%VPY%" decode_test.py onegin12k.llz
echo === TEST END ===
pause
exit /b 0