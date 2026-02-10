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
python -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Please run: python -m pip install -r requirements.txt"
    read -p "Press Enter to continue..."
    exit 1
fi
echo "Dependencies installed successfully!"
echo ""

# Step 2: Check if port 7860 (Gradio default) is in use
echo "[2/4] Checking if port 7860 is available..."
echo ""

# Find process using port 7860
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:7860 2>/dev/null)
elif command -v fuser &> /dev/null; then
    PID=$(fuser 7860/tcp 2>/dev/null | awk '{print $1}')
else
    PID=""
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
    
    # Check for placeholder API keys
    if grep -q "your-.*-key-here" .env; then
        echo "WARNING: .env file contains placeholder API key"
        echo "Please edit .env and add your actual Anthropic API key"
        read -p "Press Enter to continue..."
        exit 1
    fi
    echo "API key configured in .env file"
else
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "WARNING: No API key found!"
        echo ""
        echo "Please either:"
        echo "  1. Create .env file with ANTHROPIC_API_KEY=your-key-here"
        echo "  2. Set environment variable ANTHROPIC_API_KEY"
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
