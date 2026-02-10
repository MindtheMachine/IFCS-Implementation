@echo off
setlocal EnableExtensions
REM Trilogy System - Benchmark Evaluation Launcher
REM Runs TruthfulQA or ASQA benchmark evaluation in an isolated venv to avoid torch issues

set "VENV_DIR=.venv_bench"
set "REQ_FILE=requirements-benchmark.txt"

echo ============================================================
echo Trilogy System - Benchmark Evaluation Launcher
echo ============================================================
echo.

REM Step 1: Ensure virtual environment exists
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Creating isolated Python environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [1/3] Using existing virtual environment at %VENV_DIR%
)
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"

REM Step 2: Install/Update minimal benchmark dependencies (skip with SKIP_PIP_INSTALL=1)
if defined SKIP_PIP_INSTALL (
    echo [2/3] Skipping dependency installation (SKIP_PIP_INSTALL=1)
) else (
    if not exist "%REQ_FILE%" (
        echo ERROR: %REQ_FILE% not found. Please ensure it exists.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies from %REQ_FILE% into %VENV_DIR% ...
    "%PY_EXE%" -m pip install --upgrade pip --quiet
    "%PY_EXE%" -m pip install -r "%REQ_FILE%" --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [3/3] Select benchmark to run:
echo   (Note: .env will be loaded by the app. Set LLM_PROVIDER/LLM_MODEL there.)
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
    "%PY_EXE%" trilogy_app.py --benchmark truthfulqa --batch-size 5
)

if "%CHOICE%"=="2" (
    echo Running: TruthfulQA Small - 50 examples
    echo This will take approximately 8-10 minutes
    echo.
    "%PY_EXE%" trilogy_app.py --benchmark truthfulqa --batch-size 50
)

if "%CHOICE%"=="3" (
    echo Running: TruthfulQA Full - 817 examples
    echo This will take approximately 2-3 hours
    echo.
    choice /C YN /M "Are you sure you want to continue"
    if errorlevel 2 goto :end
    "%PY_EXE%" trilogy_app.py --benchmark truthfulqa
)

if "%CHOICE%"=="4" (
    echo Running: ASQA Test - 5 examples
    echo.
    "%PY_EXE%" trilogy_app.py --benchmark asqa --batch-size 5
)

if "%CHOICE%"=="5" (
    echo Running: ASQA Small - 50 examples
    echo This will take approximately 8-10 minutes
    echo.
    "%PY_EXE%" trilogy_app.py --benchmark asqa --batch-size 50
)

if "%CHOICE%"=="6" (
    echo Running: ASQA Medium - 100 examples
    echo This will take approximately 15-20 minutes
    echo.
    "%PY_EXE%" trilogy_app.py --benchmark asqa --batch-size 100
)

if "%CHOICE%"=="7" (
    echo Custom command mode
    echo.
    echo Available options:
    echo   --benchmark truthfulqa^|asqa
    echo   --batch-size N           Number of examples to evaluate
    echo   --batch-start N          Starting index (for resuming)
    echo   --rate-limit N.N         Delay between API calls (seconds)
    echo   --include-full-text      Include full text responses in CSV
    echo.
    set /p CUSTOM_CMD="Enter your command (or press Enter to cancel): "
    if not "%CUSTOM_CMD%"=="" (
        "%PY_EXE%" trilogy_app.py %CUSTOM_CMD%
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
