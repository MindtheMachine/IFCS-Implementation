#!/bin/bash
# Trilogy System - Web Interface Launcher
# Automatically installs dependencies, frees port if needed, and launches Gradio web UI

echo "============================================================"
echo "Trilogy System - Web Interface Launcher"
echo "============================================================"
echo ""

# Step 1: Install/Update Dependencies
echo "[1/4] Installing dependencies from requirements.txt..."
echo ""
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Please run: pip install -r requirements.txt"
    read -p "Press Enter to continue..."
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

# Step 2: Check if port 7860 (Gradio default) is in use
echo "[2/4] Checking if port 7860 is available..."
echo ""

# Different commands for Linux/Mac
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    PID=$(lsof -ti:7860)
else
    # Linux
    PID=$(lsof -ti:7860 2>/dev/null || fuser 7860/tcp 2>/dev/null | awk '{print $1}')
fi

if [ ! -z "$PID" ]; then
    echo "Port 7860 is in use by process ID: $PID"
    echo "Attempting to free port..."

    kill -9 $PID 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "WARNING: Could not kill process $PID"
        echo "You may need to run this script with sudo"
        echo ""
        read -p "Continue anyway? (y/n): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Port 7860 freed successfully!"
        echo "Waiting 2 seconds for port to be released..."
        sleep 2
    fi
else
    echo "Port 7860 is available!"
fi
echo ""

# Step 3: Check for API key
echo "[3/4] Checking API key configuration..."
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

# Step 4: Launch web interface
echo "[4/4] Launching Trilogy Web Interface..."
echo ""
echo "The web interface will open at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

python trilogy_web.py

# If script exits
echo ""
echo "============================================================"
echo "Web interface stopped"
echo "============================================================"
read -p "Press Enter to continue..."
