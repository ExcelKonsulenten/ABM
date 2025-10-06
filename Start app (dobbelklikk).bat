@echo off
setlocal
pushd "%~dp0"
if not exist ".venv" (
  py -3 -m venv .venv
)
call ".venv\Scripts\activate"
python -m pip install --upgrade pip
python start_app.py
pause
