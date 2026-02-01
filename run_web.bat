@echo off
REM Trilogy System - Web Interface Launcher
REM Automatically installs dependencies, frees port if needed, and launches Gradio web UI

echo ============================================================
echo Trilogy System - Web Interface Launcher
echo ============================================================
echo.

REM Step 1: Install/Update Dependencies
echo [1/4] Installing dependencies from requirements.txt...
echo.
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please run: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Step 2: Check if port 7860 (Gradio default) is in use
echo [2/4] Checking if port 7860 is available...
echo.

REM Find process using port 7860
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do set PID=%%a

if defined PID (
    echo Port 7860 is in use by process ID: %PID%
    echo Attempting to free port...

    REM Kill the process
    taskkill /F /PID %PID% >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not kill process %PID%
        echo You may need to run this script as Administrator
        echo.
        choice /C YN /M "Continue anyway"
        if errorlevel 2 exit /b 1
    ) else (
        echo Port 7860 freed successfully!
        echo Waiting 2 seconds for port to be released...
        timeout /t 2 /nobreak >nul
    )
) else (
    echo Port 7860 is available!
)
echo.

REM Step 3: Check for API key
echo [3/4] Checking API key configuration...
echo.

if exist .env (
    echo Found .env file
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if "%%a"=="ANTHROPIC_API_KEY" (
            if "%%b"=="your-api-key-here" (
                echo WARNING: .env file contains placeholder API key
                echo Please edit .env and add your actual Anthropic API key
                pause
                exit /b 1
            )
            echo API key configured in .env file
        )
    )
) else (
    if not defined ANTHROPIC_API_KEY (
        echo WARNING: No API key found!
        echo.
        echo Please either:
        echo   1. Create .env file with ANTHROPIC_API_KEY=your-key-here
        echo   2. Set environment variable ANTHROPIC_API_KEY
        echo.
        pause
        exit /b 1
    )
    echo API key found in environment variable
)
echo.

REM Step 4: Launch web interface
echo [4/4] Launching Trilogy Web Interface...
echo.
echo The web interface will open at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

python trilogy_web.py

REM If script exits
echo.
echo ============================================================
echo Web interface stopped
echo ============================================================
pause
