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

    pip install -r requirements.txt

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
    pip list | grep -E "anthropic|gradio|numpy|datasets|rouge|tqdm|pandas"
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
        echo "ERROR: .env.template not found!"
        echo "Please ensure .env.template exists in the current directory."
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    echo "Creating .env file from template..."
    cp .env.template .env

    echo ""
    echo "Please edit .env file and uncomment ONE provider section:"
    echo "  - For Anthropic Claude"
    echo "  - For OpenAI GPT-4"
    echo "  - For HuggingFace (FREE tier)"
    echo "  - For Ollama (Local, FREE)"
    echo ""
    echo "Uncomment the 3 lines (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY)"
    echo ""
    read -p "Open .env for editing now? (y/n): " edit_now

    if [[ "$edit_now" =~ ^[Yy]$ ]]; then
        # Try different editors
        if command -v nano &> /dev/null; then
            nano .env
        elif command -v vi &> /dev/null; then
            vi .env
        elif command -v vim &> /dev/null; then
            vim .env
        else
            echo "No text editor found. Please edit .env manually:"
            echo "  nano .env"
            echo "  or"
            echo "  vi .env"
        fi
    else
        echo "Please edit .env manually and add your API key:"
        echo "  nano .env"
        echo "  or"
        echo "  vi .env"
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
