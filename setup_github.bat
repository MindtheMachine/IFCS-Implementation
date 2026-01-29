@echo off
REM Trilogy Implementation - GitHub Setup Script for Windows
REM This script automates the process of setting up the GitHub repository

setlocal enabledelayedexpansion

echo ==========================================
echo Trilogy Implementation - GitHub Setup
echo ==========================================
echo.

REM Check if in correct directory
if not exist "trilogy_config.py" (
    echo ERROR: trilogy_config.py not found!
    echo Please run this script from the trilogy-implementation directory
    pause
    exit /b 1
)

REM Check for git
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: git is not installed
    echo Please install git: https://git-scm.com/downloads
    pause
    exit /b 1
)

REM Initialize git if not already initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    echo Git initialized
) else (
    echo Git already initialized
)

REM Add .gitignore if not exists
if not exist ".gitignore" (
    echo Creating .gitignore...
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo venv/
        echo env/
        echo.
        echo # API Keys ^(CRITICAL!^)
        echo .env
        echo *.key
        echo secrets.txt
        echo.
        echo # Output files
        echo *_output.txt
        echo batch_results.json
        echo test_results.json
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # Gradio
        echo flagged/
    ) > .gitignore
    echo .gitignore created
) else (
    echo .gitignore exists
)

REM Check for API keys in files
echo.
echo Checking for API keys in files...
findstr /S /I /C:"sk-ant-" *.py *.txt *.md >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo WARNING: Found potential API key in files!
    echo Please remove before committing to GitHub
    set /p confirm="Continue anyway? (y/N): "
    if /i not "!confirm!"=="y" (
        exit /b 1
    )
) else (
    echo No API keys found in files
)

REM Add all files
echo.
echo Adding files to git...
git add .
echo Files added

REM Commit
echo.
set /p commit_msg="Commit message (or press Enter for default): "
if "!commit_msg!"=="" (
    set commit_msg=Initial commit: ECR-Control Probe-IFCS Trilogy implementation
)

git commit -m "!commit_msg!" 2>nul
if %ERRORLEVEL% neq 0 (
    echo No changes to commit
)

REM Ask about GitHub
echo.
echo ==========================================
echo GitHub Repository Setup
echo ==========================================
echo.
echo Choose your setup method:
echo 1. Use GitHub CLI (gh)
echo 2. Manual setup (I'll create repo on github.com)
echo.
set /p choice="Enter choice (1 or 2): "

if "!choice!"=="1" (
    REM Check for gh CLI
    where gh >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        echo ERROR: GitHub CLI not installed
        echo Install it: https://cli.github.com/
        pause
        exit /b 1
    )
    
    echo.
    set /p repo_name="Repository name (default: trilogy-implementation): "
    if "!repo_name!"=="" (
        set repo_name=trilogy-implementation
    )
    
    set /p is_public="Make repository public? (y/N): "
    if /i "!is_public!"=="y" (
        set visibility=--public
    ) else (
        set visibility=--private
    )
    
    echo.
    echo Creating GitHub repository...
    gh repo create "MindtheMachine/!repo_name!" !visibility! --source=. --remote=origin
    
    echo.
    echo Pushing to GitHub...
    git branch -M main
    git push -u origin main
    
    echo.
    echo SUCCESS!
    echo Repository created: https://github.com/MindtheMachine/!repo_name!
    
) else if "!choice!"=="2" (
    echo.
    echo Manual Setup Instructions:
    echo.
    echo 1. Go to: https://github.com/new
    echo 2. Repository name: trilogy-implementation
    echo 3. Description: Implementation of ECR-Control Probe-IFCS inference-time governance trilogy
    echo 4. Choose Public or Private
    echo 5. DON'T initialize with README (we have one)
    echo 6. Click 'Create repository'
    echo.
    pause
    
    echo.
    set /p repo_url="Enter the repository URL (e.g., https://github.com/MindtheMachine/trilogy-implementation.git): "
    
    if "!repo_url!"=="" (
        echo ERROR: Repository URL required
        pause
        exit /b 1
    )
    
    echo.
    echo Adding remote and pushing...
    git remote add origin "!repo_url!" 2>nul
    if !ERRORLEVEL! neq 0 (
        git remote set-url origin "!repo_url!"
    )
    git branch -M main
    git push -u origin main
    
    echo.
    echo SUCCESS!
    echo Repository URL: !repo_url!
) else (
    echo ERROR: Invalid choice
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Visit your repository on GitHub
echo 2. Add topics: llm-safety, ai-governance, inference-time-control
echo 3. Enable GitHub Discussions (optional)
echo 4. Add repository description
echo 5. Star your own repository!
echo.
echo To continue in Claude Code:
echo   cd %CD%
echo   claude-code
echo.
echo See CLAUDE_CODE_SETUP.md for more details
echo.
echo Setup complete!
echo.
pause
