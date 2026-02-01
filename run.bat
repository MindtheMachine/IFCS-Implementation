@echo off
REM Trilogy System - Main Launcher
REM Interactive menu for running different components

:menu
cls
echo ============================================================
echo          TRILOGY SYSTEM - ECR-CONTROL PROBE-IFCS
echo ============================================================
echo.
echo Select what you want to run:
echo.
echo 1. Web Interface (Gradio UI)
echo 2. Benchmark Evaluation (TruthfulQA / ASQA)
echo 3. Single Query (Command Line)
echo 4. Test Suite (36 taxonomy cases)
echo 5. Install/Update Dependencies
echo 6. Setup API Key (.env configuration)
echo 7. Exit
echo.

choice /C 1234567 /N /M "Enter your choice (1-7): "
set CHOICE=%ERRORLEVEL%

echo.

if "%CHOICE%"=="1" goto :web
if "%CHOICE%"=="2" goto :benchmark
if "%CHOICE%"=="3" goto :single
if "%CHOICE%"=="4" goto :testsuite
if "%CHOICE%"=="5" goto :install
if "%CHOICE%"=="6" goto :setup_key
if "%CHOICE%"=="7" goto :exit

:web
cls
echo ============================================================
echo Launching Web Interface...
echo ============================================================
echo.
call run_web.bat
goto :menu

:benchmark
cls
echo ============================================================
echo Launching Benchmark Evaluation...
echo ============================================================
echo.
call run_benchmark.bat
goto :menu

:single
cls
echo ============================================================
echo Single Query Processing
echo ============================================================
echo.
echo Enter your query (or press Enter to cancel):
set /p QUERY="Query: "
if "%QUERY%"=="" goto :menu

echo.
echo Processing query through trilogy system...
echo.
python trilogy_app.py --prompt "%QUERY%"

echo.
echo Results saved to:
echo   - baseline_output.txt
echo   - regulated_output.txt
echo   - comparison_analysis.txt
echo.
pause
goto :menu

:testsuite
cls
echo ============================================================
echo Running Test Suite (36 Taxonomy Cases)
echo ============================================================
echo.
echo This will run the first 10 test cases from the taxonomy.
echo.
choice /C YN /M "Continue"
if errorlevel 2 goto :menu

echo.
python trilogy_app.py --test-suite

echo.
echo Test results saved to: test_results.json
echo.
pause
goto :menu

:install
cls
echo ============================================================
echo Installing/Updating Dependencies
echo ============================================================
echo.
echo Installing packages from requirements.txt...
echo.
python -m pip install -r requirements.txt
echo.
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    goto :menu
)

echo.
echo ============================================================
echo Dependencies installed successfully!
echo ============================================================
echo.
echo Installed packages:
python -m pip list | findstr /C:"anthropic" /C:"gradio" /C:"numpy" /C:"datasets" /C:"rouge" /C:"tqdm" /C:"pandas"
echo.
pause
goto :menu

:setup_key
cls
echo ============================================================
echo API Key Setup
echo ============================================================
echo.

if exist .env (
    echo Current .env file found:
    type .env
    echo.
    choice /C YN /M "Do you want to recreate the .env file"
    if errorlevel 2 goto :menu
)

if not exist .env.template (
    echo Creating .env.template...
    echo # Environment Variables for Trilogy System > .env.template
    echo # Copy this file to .env and fill in your actual API key >> .env.template
    echo. >> .env.template
    echo # Anthropic API Key (required^) >> .env.template
    echo # Get your API key from: https://console.anthropic.com/ >> .env.template
    echo ANTHROPIC_API_KEY=your-api-key-here >> .env.template
)

echo.
echo Creating .env file...
copy .env.template .env >nul

echo.
echo Please enter your Anthropic API key:
echo (Get it from: https://console.anthropic.com/)
echo.
set /p API_KEY="API Key: "

if "%API_KEY%"=="" (
    echo No API key entered. Using placeholder.
    echo Please edit .env manually and add your API key.
) else (
    echo # Environment Variables for Trilogy System > .env
    echo # Anthropic API Key >> .env
    echo ANTHROPIC_API_KEY=%API_KEY% >> .env
    echo.
    echo API key saved to .env file!
)

echo.
pause
goto :menu

:exit
cls
echo ============================================================
echo Thank you for using Trilogy System!
echo ============================================================
echo.
exit /b 0
