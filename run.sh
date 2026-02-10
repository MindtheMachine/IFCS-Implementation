#!/bin/bash
# Trilogy System - Main Launcher
# Interactive menu for running different components

show_menu() {
    clear
    echo "============================================================"
    echo "         TRILOGY SYSTEM - ECR-CONTROL PROBE-IFCS"
    echo "============================================================"
    echo ""
    echo "Select what you want to run:"
    echo ""
    echo "1. Web Interface (Gradio UI)"
    echo "2. Benchmark Evaluation (TruthfulQA / ASQA)"
    echo "3. Single Query (Command Line)"
    echo "4. Test Suite (36 taxonomy cases)"
    echo "5. Install/Update Dependencies"
    echo "6. Setup API Key (.env configuration)"
    echo "7. Exit"
    echo ""
    echo "Note: .env is loaded by the Python app at runtime (LLM_PROVIDER/LLM_MODEL)"
}

web_interface() {
    clear
    echo "============================================================"
    echo "Launching Web Interface..."
    echo "============================================================"
    echo ""
    bash run_web.sh
}

benchmark_evaluation() {
    clear
    echo "============================================================"
    echo "Launching Benchmark Evaluation..."
    echo "============================================================"
    echo ""
    bash run_benchmark.sh
}

single_query() {
    clear
    echo "============================================================"
    echo "Single Query Processing"
    echo "============================================================"
    echo ""
    read -p "Enter your query (or press Enter to cancel): " QUERY

    if [ -z "$QUERY" ]; then
        return
    fi

    echo ""
    echo "Processing query through trilogy system..."
    echo ""
    python trilogy_app.py --prompt "$QUERY"

    echo ""
    echo "Results saved to:"
    echo "  - baseline_output.txt"
    echo "  - regulated_output.txt"
    echo "  - comparison_analysis.txt"
    echo ""
    read -p "Press Enter to continue..."
}

test_suite() {
    clear
    echo "============================================================"
    echo "Running Test Suite (36 Taxonomy Cases)"
    echo "============================================================"
    echo ""
    echo "This will run the first 10 test cases from the taxonomy."
    echo ""
    read -p "Continue? (y/n): " confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        return
    fi

    echo ""
    python trilogy_app.py --test-suite

    echo ""
    echo "Test results saved to: test_results.json"
    echo ""
    read -p "Press Enter to continue..."
}

install_dependencies() {
    clear
    echo "============================================================"
    echo "Installing/Updating Dependencies"
    echo "============================================================"
    echo ""
    echo "Installing packages from requirements.txt..."
    echo ""

    python -m pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "ERROR: Installation failed"
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    echo "============================================================"
    echo "Dependencies installed successfully!"
    echo "============================================================"
    echo ""
    echo "Installed packages:"
    python -m pip list | grep -E "anthropic|gradio|numpy|datasets|rouge|tqdm|pandas|python-dotenv|ollama|openai|huggingface"
    echo ""
    read -p "Press Enter to continue..."
}

setup_api_key() {
    clear
    echo "============================================================"
    echo "API Key Setup"
    echo "============================================================"
    echo ""

    if [ -f .env ]; then
        echo "Current .env file found:"
        cat .env
        echo ""
        read -p "Do you want to recreate the .env file? (y/n): " recreate
        if [[ ! "$recreate" =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    if [ ! -f .env.template ]; then
        echo "Creating .env.template..."
        cat > .env.template << 'EOF'
# Environment Variables for Trilogy System
# Copy this file to .env and fill in your actual API key

# Anthropic API Key (required)
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-api-key-here
EOF
    fi

    echo ""
    echo "Creating .env file from template..."
    cp .env.template .env

    echo ""
    echo "Please enter your Anthropic API key:"
    echo "(Get it from: https://console.anthropic.com/)"
    echo ""
    read -p "API Key: " API_KEY

    if [ -z "$API_KEY" ]; then
        echo "No API key entered. Using placeholder."
        echo "Please edit .env manually and add your API key."
    else
        cat > .env << EOF
# Environment Variables for Trilogy System
# Anthropic API Key
ANTHROPIC_API_KEY=$API_KEY
EOF
        echo ""
        echo "API key saved to .env file!"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-7): " choice
    echo ""

    case $choice in
        1) web_interface ;;
        2) benchmark_evaluation ;;
        3) single_query ;;
        4) test_suite ;;
        5) install_dependencies ;;
        6) setup_api_key ;;
        7)
            clear
            echo "============================================================"
            echo "Thank you for using Trilogy System!"
            echo "============================================================"
            echo ""
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1-7."
            sleep 2
            ;;
    esac
done
