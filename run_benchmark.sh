#!/bin/bash
# Trilogy System - Benchmark Evaluation Launcher
# Runs TruthfulQA or ASQA benchmark evaluation in an isolated venv to avoid torch issues

VENV_DIR=".venv_bench"
REQ_FILE="requirements-benchmark.txt"

echo "============================================================"
echo "Trilogy System - Benchmark Evaluation Launcher"
echo "============================================================"
echo ""

# Step 1: Ensure virtual environment exists
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/3] Creating isolated Python environment at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        read -p "Press Enter to continue..."
        exit 1
    fi
else
    echo "[1/3] Using existing virtual environment at $VENV_DIR"
fi
PY_EXE="$VENV_DIR/bin/python"

# Step 2: Install/Update minimal benchmark dependencies (skip with SKIP_PIP_INSTALL=1)
if [ -n "$SKIP_PIP_INSTALL" ]; then
    echo "[2/3] Skipping dependency installation (SKIP_PIP_INSTALL=1)"
else
    if [ ! -f "$REQ_FILE" ]; then
        echo "ERROR: $REQ_FILE not found. Please ensure it exists."
        read -p "Press Enter to continue..."
        exit 1
    fi
    echo "[2/3] Installing dependencies from $REQ_FILE into $VENV_DIR ..."
    "$PY_EXE" -m pip install --upgrade pip --quiet
    "$PY_EXE" -m pip install -r "$REQ_FILE" --quiet
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        read -p "Press Enter to continue..."
        exit 1
    fi
fi

echo ""
echo "[3/3] Select benchmark to run:"
echo "  (Note: .env will be loaded by the app. Set LLM_PROVIDER/LLM_MODEL there.)"
echo ""
echo "1. TruthfulQA (Test - 5 examples)"
echo "2. TruthfulQA (Small - 50 examples)"
echo "3. TruthfulQA (Full - 817 examples)"
echo "4. ASQA (Test - 5 examples)"
echo "5. ASQA (Small - 50 examples)"
echo "6. ASQA (Medium - 100 examples)"
echo "7. Custom command"
echo ""

read -p "Enter your choice (1-7): " CHOICE

echo ""
echo "============================================================"
echo "Starting benchmark evaluation..."
echo "============================================================"
echo ""

case $CHOICE in
    1)
        echo "Running: TruthfulQA Test - 5 examples"
        echo ""
        "$PY_EXE" trilogy_app.py --benchmark truthfulqa --batch-size 5
        ;;
    2)
        echo "Running: TruthfulQA Small - 50 examples"
        echo "This will take approximately 8-10 minutes"
        echo ""
        "$PY_EXE" trilogy_app.py --benchmark truthfulqa --batch-size 50
        ;;
    3)
        echo "Running: TruthfulQA Full - 817 examples"
        echo "This will take approximately 2-3 hours"
        echo ""
        read -p "Are you sure you want to continue? (y/n): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        "$PY_EXE" trilogy_app.py --benchmark truthfulqa
        ;;
    4)
        echo "Running: ASQA Test - 5 examples"
        echo ""
        "$PY_EXE" trilogy_app.py --benchmark asqa --batch-size 5
        ;;
    5)
        echo "Running: ASQA Small - 50 examples"
        echo "This will take approximately 8-10 minutes"
        echo ""
        "$PY_EXE" trilogy_app.py --benchmark asqa --batch-size 50
        ;;
    6)
        echo "Running: ASQA Medium - 100 examples"
        echo "This will take approximately 15-20 minutes"
        echo ""
        "$PY_EXE" trilogy_app.py --benchmark asqa --batch-size 100
        ;;
    7)
        echo "Custom command mode"
        echo ""
        echo "Available options:"
        echo "  --benchmark truthfulqa|asqa"
        echo "  --batch-size N           Number of examples to evaluate"
        echo "  --batch-start N          Starting index (for resuming)"
        echo "  --rate-limit N.N         Delay between API calls (seconds)"
        echo "  --include-full-text      Include full text responses in CSV"
        echo ""
        read -p "Enter your command (or press Enter to cancel): " CUSTOM_CMD
        if [ ! -z "$CUSTOM_CMD" ]; then
            "$PY_EXE" trilogy_app.py $CUSTOM_CMD
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Benchmark evaluation complete!"
echo "============================================================"
echo ""
echo "Results saved in current directory:"
echo "  - *_results.csv        : Per-example results"
echo "  - *_summary.json       : Aggregated statistics"
echo "  - *_comparison.txt     : Baseline vs regulated comparison"
echo "  - *_report.html        : Visual report (open in browser)"
echo ""
read -p "Press Enter to continue..."
