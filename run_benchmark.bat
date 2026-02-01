@echo off
REM Trilogy System - Benchmark Evaluation Launcher
REM Runs TruthfulQA or ASQA benchmark evaluation

echo ============================================================
echo Trilogy System - Benchmark Evaluation Launcher
echo ============================================================
echo.

REM Step 1: Install/Update Dependencies
echo [1/3] Installing dependencies from requirements.txt...
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

REM Step 2: Check for API key
echo [2/3] Checking API key configuration...
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

REM Step 3: Prompt for benchmark selection
echo [3/3] Select benchmark to run:
echo.
echo 1. TruthfulQA (Test - 5 examples)
echo 2. TruthfulQA (Small - 50 examples)
echo 3. TruthfulQA (Full - 817 examples)
echo 4. ASQA (Test - 5 examples)
echo 5. ASQA (Small - 50 examples)
echo 6. ASQA (Medium - 100 examples)
echo 7. Custom command
echo.

choice /C 1234567 /N /M "Enter your choice (1-7): "
set CHOICE=%ERRORLEVEL%

echo.
echo ============================================================
echo Starting benchmark evaluation...
echo ============================================================
echo.

if "%CHOICE%"=="1" (
    echo Running: TruthfulQA Test - 5 examples
    echo.
    python trilogy_app.py --benchmark truthfulqa --batch-size 5
)

if "%CHOICE%"=="2" (
    echo Running: TruthfulQA Small - 50 examples
    echo This will take approximately 8-10 minutes
    echo.
    python trilogy_app.py --benchmark truthfulqa --batch-size 50
)

if "%CHOICE%"=="3" (
    echo Running: TruthfulQA Full - 817 examples
    echo This will take approximately 2-3 hours
    echo.
    choice /C YN /M "Are you sure you want to continue"
    if errorlevel 2 goto :end
    python trilogy_app.py --benchmark truthfulqa
)

if "%CHOICE%"=="4" (
    echo Running: ASQA Test - 5 examples
    echo.
    python trilogy_app.py --benchmark asqa --batch-size 5
)

if "%CHOICE%"=="5" (
    echo Running: ASQA Small - 50 examples
    echo This will take approximately 8-10 minutes
    echo.
    python trilogy_app.py --benchmark asqa --batch-size 50
)

if "%CHOICE%"=="6" (
    echo Running: ASQA Medium - 100 examples
    echo This will take approximately 15-20 minutes
    echo.
    python trilogy_app.py --benchmark asqa --batch-size 100
)

if "%CHOICE%"=="7" (
    echo Custom command mode
    echo.
    echo Available options:
    echo   --benchmark truthfulqa^|asqa
    echo   --batch-size N           Number of examples to evaluate
    echo   --batch-start N          Starting index (for resuming)
    echo   --rate-limit N.N         Delay between API calls (seconds)
    echo.
    echo Example: --benchmark truthfulqa --batch-size 100 --rate-limit 2.0
    echo.
    set /p CUSTOM_CMD="Enter your command (or press Enter to cancel): "
    if not "%CUSTOM_CMD%"=="" (
        python trilogy_app.py %CUSTOM_CMD%
    )
)

:end
echo.
echo ============================================================
echo Benchmark evaluation complete!
echo ============================================================
echo.
echo Results saved in current directory:
echo   - *_results.csv        : Per-example results
echo   - *_summary.json       : Aggregated statistics
echo   - *_comparison.txt     : Baseline vs regulated comparison
echo   - *_report.html        : Visual report (open in browser)
echo.
pause
