#!/bin/bash
# Trilogy Implementation - GitHub Setup Script for Linux/Mac
# This script automates the process of setting up the GitHub repository

echo "=========================================="
echo "Trilogy Implementation - GitHub Setup"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "trilogy_config.py" ]; then
    echo "ERROR: trilogy_config.py not found!"
    echo "Please run this script from the trilogy-implementation directory"
    read -p "Press Enter to continue..."
    exit 1
fi

# Check for git
if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed"
    echo "Please install git:"
    echo "  - Ubuntu/Debian: sudo apt-get install git"
    echo "  - macOS: brew install git"
    echo "  - Or visit: https://git-scm.com/downloads"
    read -p "Press Enter to continue..."
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "Git initialized"
else
    echo "Git already initialized"
fi

# Add .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
venv/
env/

# API Keys (CRITICAL!)
.env
*.key
secrets.txt

# Output files
*_output.txt
batch_results.json
test_results.json

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Gradio
flagged/
EOF
    echo ".gitignore created"
else
    echo ".gitignore exists"
fi

# Check for API keys in files
echo ""
echo "Checking for API keys in files..."
if grep -r -i "sk-ant-\|sk-proj-\|hf_[A-Za-z0-9]" *.py *.txt *.md 2>/dev/null | grep -v ".git" | grep -v "example" | grep -v "template" | grep -v "your-.*-key"; then
    echo ""
    echo "WARNING: Found potential API key in files!"
    echo "Please remove before committing to GitHub"
    read -p "Continue anyway? (y/n): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "No API keys found in files"
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .
echo "Files added"

# Commit
echo ""
read -p "Commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: ECR-Control Probe-IFCS Trilogy implementation"
fi

git commit -m "$commit_msg" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No changes to commit"
fi

# Ask about GitHub
echo ""
echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""
echo "Choose your setup method:"
echo "1. Use GitHub CLI (gh)"
echo "2. Manual setup (I'll create repo on github.com)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    # Check for gh CLI
    if ! command -v gh &> /dev/null; then
        echo "ERROR: GitHub CLI not installed"
        echo "Install it:"
        echo "  - Ubuntu/Debian: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
        echo "  - macOS: brew install gh"
        echo "  - Or visit: https://cli.github.com/"
        read -p "Press Enter to continue..."
        exit 1
    fi

    echo ""
    read -p "Repository name (default: trilogy-implementation): " repo_name
    if [ -z "$repo_name" ]; then
        repo_name="trilogy-implementation"
    fi

    read -p "Make repository public? (y/n): " is_public
    if [[ "$is_public" =~ ^[Yy]$ ]]; then
        visibility="--public"
    else
        visibility="--private"
    fi

    echo ""
    echo "Creating GitHub repository..."
    gh repo create "MindtheMachine/$repo_name" $visibility --source=. --remote=origin

    echo ""
    echo "Pushing to GitHub..."
    git branch -M main
    git push -u origin main

    echo ""
    echo "SUCCESS!"
    echo "Repository created: https://github.com/MindtheMachine/$repo_name"

elif [ "$choice" = "2" ]; then
    echo ""
    echo "Manual Setup Instructions:"
    echo ""
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: trilogy-implementation"
    echo "3. Description: Implementation of ECR-Control Probe-IFCS inference-time governance trilogy"
    echo "4. Choose Public or Private"
    echo "5. DON'T initialize with README (we have one)"
    echo "6. Click 'Create repository'"
    echo ""
    read -p "Press Enter when you've created the repository..."

    echo ""
    read -p "Enter the repository URL (e.g., https://github.com/MindtheMachine/trilogy-implementation.git): " repo_url

    if [ -z "$repo_url" ]; then
        echo "ERROR: Repository URL required"
        read -p "Press Enter to continue..."
        exit 1
    fi

    echo ""
    echo "Adding remote and pushing..."
    git remote add origin "$repo_url" 2>/dev/null
    if [ $? -ne 0 ]; then
        git remote set-url origin "$repo_url"
    fi
    git branch -M main
    git push -u origin main

    echo ""
    echo "SUCCESS!"
    echo "Repository URL: $repo_url"
else
    echo "ERROR: Invalid choice"
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Visit your repository on GitHub"
echo "2. Add topics: llm-safety, ai-governance, inference-time-control"
echo "3. Enable GitHub Discussions (optional)"
echo "4. Add repository description"
echo "5. Star your own repository!"
echo ""
echo "To continue in Claude Code:"
echo "  cd $(pwd)"
echo "  claude-code"
echo ""
echo "See CLAUDE_CODE_SETUP.md for more details"
echo ""
echo "Setup complete!"
echo ""
read -p "Press Enter to continue..."
