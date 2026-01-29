# Windows Setup Guide - Trilogy System

## 🪟 Quick Setup for Windows Users

This guide is specifically for Windows users. All commands use Windows syntax.

---

## 🚀 Method 1: Automated Setup (Recommended)

### Step 1: Download All Files
1. Download all 20 files to a folder (e.g., `C:\Users\YourName\trilogy-implementation`)
2. Open **Command Prompt** or **PowerShell** as Administrator

### Step 2: Navigate to Folder
```cmd
cd C:\Users\YourName\trilogy-implementation
```

### Step 3: Run Setup Script
```cmd
setup_github.bat
```

The script will:
- ✅ Check for git installation
- ✅ Initialize git repository
- ✅ Create .gitignore (protects API keys!)
- ✅ Check for API keys in files
- ✅ Commit all files
- ✅ Help create GitHub repository
- ✅ Push to GitHub

**Result**: Your code is on GitHub! 🎉

---

## 🚀 Method 2: Manual Setup

### Prerequisites
1. **Git for Windows** - Download from: https://git-scm.com/download/win
2. **GitHub account** - Sign up at: https://github.com

### Step 1: Open Command Prompt
Press `Win + R`, type `cmd`, press Enter

### Step 2: Navigate to Your Folder
```cmd
cd C:\Users\YourName\trilogy-implementation
```

### Step 3: Initialize Git
```cmd
git init
```

### Step 4: Create .gitignore (IMPORTANT!)
Create a file named `.gitignore` with this content:
```
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
```

**Windows tip**: To create `.gitignore`, use:
```cmd
notepad .gitignore
```
Then paste the content and save.

### Step 5: Add and Commit Files
```cmd
git add .
git commit -m "Initial commit: ECR-Control Probe-IFCS Trilogy implementation"
```

### Step 6: Create GitHub Repository

**Option A: Using GitHub CLI (if installed)**
```cmd
REM Install GitHub CLI first: https://cli.github.com/
gh repo create MindtheMachine/trilogy-implementation --public --source=. --remote=origin
git branch -M main
git push -u origin main
```

**Option B: Manual (Recommended for Windows)**
1. Go to https://github.com/new
2. Repository name: `trilogy-implementation`
3. Description: `Implementation of ECR-Control Probe-IFCS inference-time governance trilogy`
4. Choose **Public** or **Private**
5. **DON'T** check "Initialize with README" (we already have one)
6. Click **"Create repository"**

### Step 7: Push to GitHub
After creating the repository on GitHub, run these commands:
```cmd
git remote add origin https://github.com/MindtheMachine/trilogy-implementation.git
git branch -M main
git push -u origin main
```

**Done!** Your repository is at:
`https://github.com/MindtheMachine/trilogy-implementation`

---

## 🎯 Testing Locally on Windows

### Step 1: Install Python
Download Python 3.9+ from: https://www.python.org/downloads/

**IMPORTANT**: Check "Add Python to PATH" during installation!

### Step 2: Open Command Prompt
```cmd
cd C:\Users\YourName\trilogy-implementation
```

### Step 3: Create Virtual Environment (Recommended)
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies
```cmd
pip install -r requirements.txt
```

### Step 5: Set API Key

**Option A: Environment Variable (Session-only)**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option B: Permanent Environment Variable**
1. Press `Win + X` → System
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `ANTHROPIC_API_KEY`
6. Variable value: `sk-ant-your-key-here`
7. Click OK

### Step 6: Run the Application

**Web Interface:**
```cmd
python trilogy_web.py
```
Opens at: http://localhost:7860

**Command Line:**
```cmd
python trilogy_app.py --prompt "What is the best programming language?"
```

---

## 🔧 Windows-Specific Tips

### PowerShell vs Command Prompt
Both work! But some differences:

**Command Prompt (cmd):**
```cmd
set ANTHROPIC_API_KEY=your-key
cd C:\trilogy-implementation
```

**PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="your-key"
Set-Location C:\trilogy-implementation
```

### Path with Spaces
If your path has spaces, use quotes:
```cmd
cd "C:\Users\Your Name\trilogy implementation"
```

### File Extensions
Windows hides file extensions by default. To show them:
1. Open File Explorer
2. View tab → Check "File name extensions"

### Creating .gitignore
Windows Explorer won't let you create files starting with `.`

**Solution 1: Command Prompt**
```cmd
notepad .gitignore
```

**Solution 2: PowerShell**
```powershell
New-Item .gitignore -ItemType File
```

**Solution 3: Name it `.gitignore.`**
Windows will automatically remove the trailing dot.

---

## 🚀 Using Git Bash (Alternative)

**Git Bash** is a Linux-like terminal for Windows (comes with Git for Windows).

### Advantages:
- ✅ Use Linux commands (bash, ls, etc.)
- ✅ Better for git operations
- ✅ Can use .sh scripts

### How to Use:
1. Right-click in your folder
2. Select "Git Bash Here"
3. Use Linux commands:
```bash
chmod +x setup_github.sh
./setup_github.sh
```

---

## 🎯 Continuing in Claude Code (Windows)

### Step 1: Install Claude Code
```cmd
pip install claude-code
```

### Step 2: Set API Key (if not already set)
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 3: Navigate to Project
```cmd
cd C:\Users\YourName\trilogy-implementation
```

### Step 4: Start Claude Code
```cmd
claude-code
```

### Step 5: Start Developing!
```
You: "Let's review the codebase"
Claude Code: [provides overview]

You: "Add unit tests for IFCS"
Claude Code: [creates tests]
```

---

## 🐛 Troubleshooting

### "git is not recognized"
**Solution**: Install Git for Windows from https://git-scm.com/download/win
- During installation, select "Use Git from Windows Command Prompt"

### "python is not recognized"
**Solution**: 
1. Reinstall Python and check "Add to PATH"
2. Or manually add Python to PATH:
   - Find Python folder (e.g., `C:\Python39`)
   - Add to System PATH in Environment Variables

### "Permission denied" when running script
**Solution**: Run Command Prompt as Administrator
- Press `Win + X` → "Command Prompt (Admin)"

### API key not working
**Solution**: Check that you've set it correctly:
```cmd
echo %ANTHROPIC_API_KEY%
```
Should show: `sk-ant-...`

### Port 7860 already in use
**Solution**: Change port in `trilogy_web.py`:
```python
interface.launch(server_port=7861)  # Use different port
```

---

## 📋 Quick Reference

### Essential Commands

**Navigate:**
```cmd
cd C:\path\to\folder          REM Change directory
dir                           REM List files
cd ..                         REM Go up one level
```

**Git:**
```cmd
git init                      REM Initialize repository
git add .                     REM Stage all files
git commit -m "message"       REM Commit changes
git push                      REM Push to GitHub
git status                    REM Check status
```

**Python:**
```cmd
python --version              REM Check Python version
pip install -r requirements.txt    REM Install dependencies
python trilogy_web.py         REM Run web interface
```

**Environment:**
```cmd
set VARNAME=value            REM Set environment variable
echo %VARNAME%               REM Display variable
```

---

## 🎉 Success Checklist

After setup, you should have:
- ✅ Git installed and working
- ✅ Repository on GitHub
- ✅ Code pushed to `MindtheMachine/trilogy-implementation`
- ✅ Python environment set up
- ✅ Dependencies installed
- ✅ API key configured
- ✅ Web interface running locally

**Next**: Try the medical test case to see 87% commitment reduction! 🚀

---

## 📞 Still Need Help?

### Common Resources:
- **Git for Windows**: https://git-scm.com/download/win
- **Python for Windows**: https://www.python.org/downloads/
- **GitHub CLI**: https://cli.github.com/
- **Claude Code Docs**: https://docs.anthropic.com/

### Video Tutorials:
- "Git for Windows Tutorial" on YouTube
- "Python Installation Windows" on YouTube
- "GitHub for Beginners" on YouTube

### Windows-Specific Forums:
- Stack Overflow (tag: windows, git)
- Reddit: r/learnprogramming
- GitHub Community Forum

---

**Ready to go!** Follow the automated setup script for easiest path! 🎊
