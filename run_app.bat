@echo off
title Creato AI - Studio Launcher
setlocal EnableDelayedExpansion

echo ============================================================
echo            CREATO AI - AUTO SHORT VIDEO STUDIO
echo ============================================================
echo.

:: Ensure working directory is the project directory
cd /d "%~dp0"

:: 1. Validate environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please make sure the setup completed.
    pause
    exit /b 1
)

:: 2. Free up ports 8000 and 5173 if already occupied
echo [*] Checking network ports 8000 and 5173...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    if "%%a" neq "0" (
        echo [*] Freeing port 8000 (PID %%a)...
        taskkill /F /PID %%a >nul 2>&1
    )
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173 "') do (
    if "%%a" neq "0" (
        echo [*] Freeing port 5173 (PID %%a)...
        taskkill /F /PID %%a >nul 2>&1
    )
)

:: 3. Start Backend Server
echo [1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Creato AI - Backend" cmd /c "title Creato AI - Backend && .venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"

:: 4. Start Frontend Dashboard
echo [2/3] Starting Vite Dashboard on http://localhost:5173 ...
start "Creato AI - Frontend" cmd /c "title Creato AI - Frontend && cd dashboard && npm.cmd run dev"

:: 5. Wait for servers to initialize
echo [3/3] Waiting for servers to start...
timeout /t 3 /nobreak >nul

:: Open browser
echo.
echo ============================================================
echo   Creato AI is now running!
echo   Frontend Dashboard: http://localhost:5173
echo   Backend Server:     http://127.0.0.1:8000
echo ============================================================
echo Opening browser...
start http://localhost:5173

echo.
echo This launcher will automatically close in 5 seconds.
echo Creato AI will continue running in the background.
echo (To shut down both servers anytime, double-click "Stop Creato AI" on your Desktop)
echo.
timeout /t 5 >nul
exit /b 0
