@echo off
cd /d "%~dp0"

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

if not exist .env (
    echo.
    echo  ERROR: .env file not found.
    echo  Copy .env.example to .env and add your API keys.
    echo.
    pause
    exit /b 1
)

echo.
echo  ========================================
echo   Zettai Ittchi v0.1.0
echo  ========================================
echo.
echo  Endpoint:     http://127.0.0.1:8000/v1
echo  Health check: http://127.0.0.1:8000/health
echo.
echo  Press Ctrl+C to stop.
echo.

python -m uvicorn zettai_ittchi.app:app --host 127.0.0.1 --port 8000
pause
