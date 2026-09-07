@echo off
title Creato AI - Stopper
echo ============================================================
echo             STOPPING CREATO AI STUDIO SERVERS
echo ============================================================
echo.

echo [*] Stopping Backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    if "%%a" neq "0" (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo [*] Stopping Frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173 "') do (
    if "%%a" neq "0" (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo.
echo [DONE] Creato AI servers stopped successfully.
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
exit /b 0
