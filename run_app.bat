@echo off
title Creato AI - Studio Launcher
setlocal EnableDelayedExpansion

echo ============================================================
echo            CREATO AI - AUTO SHORT VIDEO STUDIO
echo ============================================================
echo.

:: Ensure working directory is the project directory
cd /d "%~dp0"
set "PROJECT_DIR=%CD%"

:: Ensure Node and FFmpeg are in PATH even after system reboot
set "PATH=C:\ffmpeg\bin;C:\Program Files\nodejs;C:\Program Files (x86)\nodejs;%APPDATA%\npm;%PATH%"

:: 1. Validate environment
echo [1/4] Checking environment...
if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at:
    echo "%PROJECT_DIR%\.venv\Scripts\python.exe"
    echo Please make sure the virtual environment exists.
    pause
    exit /b 1
)

:: 2. Free up ports 8000 and 5173 if already occupied
echo [2/4] Checking network ports 8000 and 5173...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    if "%%a" neq "0" (
        echo [*] Freeing port 8000 PID %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173 "') do (
    if "%%a" neq "0" (
        echo [*] Freeing port 5173 PID %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)

:: 3. Start Backend Server
echo [3/4] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Creato AI - Backend" /d "%PROJECT_DIR%" /min cmd /k ".venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"

:: 4. Start Frontend Dashboard
echo [4/4] Starting Vite Dashboard on http://localhost:5173 ...
start "Creato AI - Frontend" /d "%PROJECT_DIR%\dashboard" /min cmd /k "npm.cmd run dev"

:: 5. Wait for servers to initialize
echo.
echo Waiting for servers to initialize...
powershell -NoProfile -Command "Start-Sleep -Seconds 3"

:: Open browser
echo.
echo ============================================================
echo   Creato AI is now running!
echo   Frontend Dashboard: http://localhost:5173
echo   Backend Server:     http://127.0.0.1:8000
echo ============================================================
echo.
echo Opening browser...
start http://localhost:5173

echo.
echo Servers are running in the background (minimized in taskbar).
echo To stop Creato AI anytime, double-click "Stop Creato AI" on your Desktop.
echo.
echo Closing launcher in 4 seconds...
powershell -NoProfile -Command "Start-Sleep -Seconds 4"
exit /b 0
