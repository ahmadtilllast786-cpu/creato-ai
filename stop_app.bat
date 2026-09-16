@echo off
title Creato AI - Stopper
echo ============================================================
echo             STOPPING CREATO AI STUDIO SERVERS
echo ============================================================
echo.

echo [*] Stopping Backend (port 8000) and Frontend (port 5173)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000, 5173 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo.
echo [DONE] Creato AI servers stopped successfully.
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
exit /b 0
