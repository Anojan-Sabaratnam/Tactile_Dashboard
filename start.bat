@echo off
echo Starting AI Tactile Dashboard...
set PYTHONPATH=%cd%\Codebase

:: Start the python server in the background
start /B .venv\Scripts\python.exe Codebase\dashboard\app.py

:: Wait 3 seconds for the server to start
timeout /t 1 /nobreak > nul

:: Open the default web browser directly to the Live Demo page
start http://127.0.0.1:8050

echo Dashboard is running. Close this window to stop the server.
:: Wait for user input so the window doesn't close immediately (which would kill the background process if it were attached differently, but here it keeps the console open)
:: Actually, since we used start /B, the python process is tied to this console.
pause
