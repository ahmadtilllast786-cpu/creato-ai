@echo off
title Creato - AI Short Video Studio
echo ============================================================
echo   Launching Creato - AI Short Video Studio
echo   Theme: Cyber-Violet Obsidian Studio
echo   Overlay: Black Card with White Writing
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found in .venv.
    echo Please make sure the setup completed.
    pause
    exit /b 1
)

echo [2/3] Starting Backend Server (FastAPI on http://localhost:8000)...
start "AI Shorts Backend" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000"

echo [3/3] Starting Frontend Dashboard (Vite on http://localhost:5173)...
start "AI Shorts Frontend" cmd /k "cd /d "%~dp0dashboard" && npm.cmd run dev"

echo.
echo All services launched!
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
timeout /t 3 >nul
start http://localhost:5173
echo Press any key to exit this launcher window (services will stay running).
pause >nul
