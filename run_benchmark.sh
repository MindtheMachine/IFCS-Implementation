#!/bin/bash
# Trilogy System - Benchmark Evaluation Launcher
# Runs TruthfulQA or ASQA benchmark evaluation

echo "============================================================"
echo "Trilogy System - Benchmark Evaluation Launcher"
echo "============================================================"
echo ""

# Step 1: Install/Update Dependencies
echo "[1/3] Installing dependencies from requirements.txt..."
echo ""
python -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Please run: python -m pip install -r requirements.txt"
    read -p "Press Enter to continue..."
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

# Step 2: Check for API key
echo "[2/3] Checking API key configuration..."
echo ""

if [ -f .env ]; then
    echo "Found .env file"

    # Check if any provider is configured
    if grep -q "^LLM_PROVIDER=" .env && grep -q "^LLM_API_KEY=" .env; then
        # Check for placeholder
        if grep -q "your-.*-key-here" .env || grep -q "your-actual-key-here" .env; then
            echo "WARNING: .env file contains placeholder API key"
            echo "Please edit .env and add your actual API key"
            read -p "Press Enter to continue..."
            exit 1
        fi
        echo "API key configured in .env file"
    else
        echo "WARNING: .env file exists but LLM_PROVIDER or LLM_API_KEY not set"
        echo "Please edit .env and uncomment ONE provider section"
        read -p "Press Enter to continue..."
        exit 1
    fi
else
    # Check for legacy ANTHROPIC_API_KEY environment variable
    if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$LLM_API_KEY" ]; then
        echo "WARNING: No API key found!"
        echo ""
        echo "Please either:"
        echo "  1. Create .env file from .env.template"
        echo "  2. Set environment variable LLM_API_KEY"
        echo ""
        echo "Run: cp .env.template .env"
        echo "Then edit .env and uncomment ONE provider section"
        echo ""
        read -p "Press Enter to continue..."
        exit 1
    fi
    echo "API key found in environment variable"
fi
echo ""

# Step 3: Prompt for benchmark selection
echo "[3/3] Select benchmark to run:"
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
        python trilogy_app.py --benchmark truthfulqa --batch-size 5
        ;;
    2)
        echo "Running: TruthfulQA Small - 50 examples"
        echo "This will take approximately 8-10 minutes"
        echo ""
        python trilogy_app.py --benchmark truthfulqa --batch-size 50
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
        python trilogy_app.py --benchmark truthfulqa
        ;;
    4)
        echo "Running: ASQA Test - 5 examples"
        echo ""
        python trilogy_app.py --benchmark asqa --batch-size 5
        ;;
    5)
        echo "Running: ASQA Small - 50 examples"
        echo "This will take approximately 8-10 minutes"
        echo ""
        python trilogy_app.py --benchmark asqa --batch-size 50
        ;;
    6)
        echo "Running: ASQA Medium - 100 examples"
        echo "This will take approximately 15-20 minutes"
        echo ""
        python trilogy_app.py --benchmark asqa --batch-size 100
        ;;
    7)
        echo "Custom command mode"
        echo ""
        echo "Available options:"
        echo "  --benchmark truthfulqa|asqa"
        echo "  --batch-size N           Number of examples to evaluate"
        echo "  --batch-start N          Starting index (for resuming)"
        echo "  --rate-limit N.N         Delay between API calls (seconds)"
        echo ""
        echo "Example: --benchmark truthfulqa --batch-size 100 --rate-limit 2.0"
        echo ""
        read -p "Enter your command (or press Enter to cancel): " CUSTOM_CMD
        if [ ! -z "$CUSTOM_CMD" ]; then
            python trilogy_app.py $CUSTOM_CMD
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
echo "Results saved in Results/[model-name]/ directory:"
echo "  - *_results.csv        : Per-example results"
echo "  - *_summary.json       : Aggregated statistics"
echo "  - *_comparison.txt     : Baseline vs regulated comparison"
echo "  - *_report.html        : Visual report (open in browser)"
echo ""
read -p "Press Enter to continue..."
