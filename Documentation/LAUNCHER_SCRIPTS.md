# Launcher Scripts Guide

This guide explains the launcher scripts available for both Windows (.bat) and Linux/Mac (.sh) users.

## 📋 Available Scripts

| Script | Windows | Linux/Mac | Purpose |
|--------|---------|-----------|---------|
| **Main Menu** | `run.bat` | `./run.sh` | Interactive menu for all functions |
| **Web Interface** | `run_web.bat` | `./run_web.sh` | Launch Gradio web UI |
| **Benchmark Evaluation** | `run_benchmark.bat` | `./run_benchmark.sh` | Run TruthfulQA/ASQA benchmarks |
| **GitHub Setup** | `setup_github.bat` | `./setup_github.sh` | Initialize git and push to GitHub |

## 🚀 Quick Start

### Windows Users

All scripts are ready to use. Simply double-click or run from command prompt:

```batch
# Interactive menu (recommended for beginners)
run.bat

# Direct launch options
run_web.bat              # Launch web interface
run_benchmark.bat        # Benchmark evaluation menu
setup_github.bat         # GitHub repository setup
```

### Linux/Mac Users

Scripts are already executable. Run from terminal:

```bash
# Interactive menu (recommended for beginners)
./run.sh

# Direct launch options
./run_web.sh             # Launch web interface
./run_benchmark.sh       # Benchmark evaluation menu
./setup_github.sh        # GitHub repository setup
```

**Note:** If you get "permission denied", make scripts executable:
```bash
chmod +x *.sh
```

## 📖 Detailed Script Documentation

### 1. Main Menu (`run.bat` / `run.sh`)

**Purpose:** Interactive menu providing access to all system functions.

**Features:**
- Web Interface launch
- Benchmark Evaluation menu
- Single query processing
- Test suite execution (36 taxonomy cases)
- Dependency installation
- API key setup
- Exit option

**Usage:**
```bash
# Windows
run.bat

# Linux/Mac
./run.sh
```

**What it does:**
1. Displays interactive menu
2. Waits for user selection (1-7)
3. Executes chosen function
4. Returns to menu after completion
5. Continues until user selects Exit

**Use this when:**
- You're new to the system
- You want a guided interface
- You're not sure which Python command to run

---

### 2. Web Interface Launcher (`run_web.bat` / `run_web.sh`)

**Purpose:** Launch Gradio web interface with automatic setup.

**Features:**
- Auto-installs dependencies from requirements.txt
- Checks and frees port 7860 if in use
- Validates API key configuration
- Launches trilogy_web.py

**Usage:**
```bash
# Windows
run_web.bat

# Linux/Mac
./run_web.sh
```

**What it does:**
1. **[1/4]** Install/update dependencies (quietly)
2. **[2/4]** Check if port 7860 is available
   - If occupied: Kill process and free port
   - If free: Continue
3. **[3/4]** Validate API key in .env file
   - Check for .env file existence
   - Verify LLM_PROVIDER and LLM_API_KEY are set
   - Warn if placeholder keys detected
4. **[4/4]** Launch web interface

**Opens:** http://localhost:7860

**Use this when:**
- You want the graphical interface
- You're processing single queries interactively
- You want to try test cases visually
- You prefer UI over command line

**Port Management:**
- **Windows:** Uses `netstat` and `taskkill`
- **Linux/Mac:** Uses `lsof` or `fuser` and `kill`

---

### 3. Benchmark Evaluation Launcher (`run_benchmark.bat` / `run_benchmark.sh`)

**Purpose:** Run benchmark evaluations with preset configurations.

**Features:**
- Auto-installs dependencies
- Validates API key configuration
- Interactive benchmark selection menu
- Preset batch sizes for different use cases
- Custom command option

**Usage:**
```bash
# Windows
run_benchmark.bat

# Linux/Mac
./run_benchmark.sh
```

**What it does:**
1. **[1/3]** Install/update dependencies (quietly)
2. **[2/3]** Validate API key in .env file
3. **[3/3]** Display benchmark selection menu:
   - **Option 1:** TruthfulQA Test (5 examples, ~1 min)
   - **Option 2:** TruthfulQA Small (50 examples, ~8-10 min)
   - **Option 3:** TruthfulQA Full (817 examples, ~2-3 hours)
   - **Option 4:** ASQA Test (5 examples, ~1 min)
   - **Option 5:** ASQA Small (50 examples, ~8-10 min)
   - **Option 6:** ASQA Medium (100 examples, ~15-20 min)
   - **Option 7:** Custom command

**Use this when:**
- You want to run benchmark evaluations
- You're not sure of the exact command syntax
- You want preset configurations
- You're comparing LLM providers

**Output Location:**
```
Results/[model-name]/
├── truthfulqa_results.csv      # Per-example results
├── truthfulqa_summary.json     # Aggregated statistics
├── truthfulqa_comparison.txt   # Baseline vs regulated
└── truthfulqa_report.html      # Visual report
```

**Custom Command Example:**
```bash
--benchmark truthfulqa --batch-size 100 --rate-limit 2.0 --batch-start 50
```

---

### 4. GitHub Setup Script (`setup_github.bat` / `setup_github.sh`)

**Purpose:** Initialize git repository and push to GitHub.

**Features:**
- Git initialization check
- .gitignore validation
- API key scan (prevents accidental leaks)
- Interactive GitHub setup
- Support for GitHub CLI (gh) or manual setup

**Usage:**
```bash
# Windows
setup_github.bat

# Linux/Mac
./setup_github.sh
```

**What it does:**
1. Verify `trilogy_config.py` exists (correct directory check)
2. Check if git is installed
3. Initialize git repository (if not already initialized)
4. Verify .gitignore exists
5. Scan files for API keys (sk-ant-, sk-proj-, hf_*)
6. Stage all files (`git add .`)
7. Show git status
8. Prompt for commit message (or use default)
9. Create commit
10. Choose GitHub setup method:
    - **Option 1:** Use GitHub CLI (gh)
    - **Option 2:** Manual setup with instructions

**GitHub CLI Path (Option 1):**
- Checks for `gh` command
- Prompts for repository name (default: IFCS-Implementation)
- Asks if public or private
- Creates repository on GitHub
- Sets origin remote
- Pushes to main branch

**Manual Setup Path (Option 2):**
- Displays step-by-step instructions
- Waits for user to create repository on github.com
- Prompts for repository URL
- Sets origin remote
- Pushes to main branch

**Use this when:**
- You're ready to push to GitHub for the first time
- You want automated API key detection
- You need guided GitHub repository creation

**Security Features:**
- Scans for API keys before commit
- Warns if keys found
- Requires confirmation to continue
- Checks .gitignore exists

---

## 🔧 Troubleshooting

### Script Not Found

**Windows:**
```batch
# Make sure you're in the correct directory
cd "c:\IFCS Implementation"

# Verify script exists
dir run.bat
```

**Linux/Mac:**
```bash
# Make sure you're in the correct directory
cd "/path/to/IFCS Implementation"

# Verify script exists
ls -l run.sh
```

### Permission Denied (Linux/Mac Only)

```bash
# Make all scripts executable
chmod +x *.sh

# Or individually
chmod +x run.sh
chmod +x run_web.sh
chmod +x run_benchmark.sh
chmod +x setup_github.sh
```

### Port 7860 Already in Use

**Scripts automatically handle this!**

- **Windows:** Uses `taskkill` to free port
- **Linux/Mac:** Uses `kill` to free port

If automatic freeing fails:
```bash
# Windows - Find process
netstat -ano | findstr :7860

# Windows - Kill manually (replace PID)
taskkill /F /PID <PID>

# Linux/Mac - Find and kill
lsof -ti:7860 | xargs kill -9
```

### API Key Not Found

Scripts will detect missing API keys and show instructions:

1. Check if .env file exists
2. Verify .env has ONE provider uncommented:
   - LLM_PROVIDER=...
   - LLM_MODEL=...
   - LLM_API_KEY=...
3. Ensure no placeholder keys (your-actual-key-here)

**Fix:**
```bash
# Create .env from template
copy .env.template .env       # Windows
cp .env.template .env         # Linux/Mac

# Edit and uncomment ONE provider
notepad .env                  # Windows
nano .env                     # Linux/Mac
```

### Dependencies Installation Failed

Scripts auto-install dependencies, but if it fails:

```bash
# Manual installation
pip install -r requirements.txt

# Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🆚 Batch Files vs Shell Scripts

### Functional Equivalence

Both .bat and .sh scripts provide **identical functionality**:
- ✅ Same menu options
- ✅ Same validation checks
- ✅ Same error handling
- ✅ Same output

### Platform-Specific Differences

| Feature | Windows (.bat) | Linux/Mac (.sh) |
|---------|----------------|-----------------|
| **Port Check** | `netstat -ano` | `lsof -ti` or `fuser` |
| **Kill Process** | `taskkill /F /PID` | `kill -9` |
| **Clear Screen** | `cls` | `clear` |
| **Pause** | `pause` | `read -p "Press Enter..."` |
| **File Check** | `if exist` | `if [ -f ]` |
| **String Comparison** | `if "%VAR%"=="value"` | `if [ "$VAR" = "value" ]` |
| **Text Editor** | `notepad` | `nano` / `vi` / `vim` |

### Implementation Notes

**Windows (.bat):**
- Uses `choice` command for interactive menus
- `errorlevel` for exit status checks
- `set /p` for user input
- `findstr` for pattern matching

**Linux/Mac (.sh):**
- Uses `read -p` for user input
- `case` statements for menu handling
- `$?` for exit status checks
- `grep` for pattern matching
- Requires execute permissions (`chmod +x`)

---

## 💡 Best Practices

### For Beginners

1. **Start with the main menu:**
   ```bash
   # Windows
   run.bat

   # Linux/Mac
   ./run.sh
   ```

2. **Use launcher scripts instead of Python commands:**
   - Scripts handle dependency installation
   - Scripts validate configuration
   - Scripts provide helpful error messages

3. **Test with small batches first:**
   - Use `--batch-size 5` for quick tests
   - Verify setup before full benchmark runs

### For Advanced Users

1. **Use launcher scripts for automation:**
   ```bash
   # Windows - scheduled task
   schtasks /create /tn "Trilogy Benchmark" /tr "C:\IFCS Implementation\run_benchmark.bat" /sc daily

   # Linux - cron job
   0 2 * * * cd /path/to/IFCS\ Implementation && ./run_benchmark.sh
   ```

2. **Direct Python commands for custom workflows:**
   ```bash
   # If you know exact parameters
   python trilogy_app.py --benchmark truthfulqa --batch-size 200 --rate-limit 1.5
   ```

3. **Modify scripts for custom presets:**
   - Edit .bat or .sh files to add your own menu options
   - Add custom batch sizes
   - Integrate with CI/CD pipelines

---

## 🔗 Related Documentation

- [SETUP.md](SETUP.md) - Complete setup guide
- [QUICK_SETUP.md](QUICK_SETUP.md) - 5-minute quickstart
- [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - Benchmark evaluation workflow
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - GitHub integration guide
- [../README.md](../README.md) - Main documentation

---

## 📝 Summary

**Use launcher scripts when:**
- ✅ You want automatic dependency installation
- ✅ You want automatic API key validation
- ✅ You prefer interactive menus
- ✅ You're not sure of exact Python commands
- ✅ You want port management handled automatically

**Use Python commands directly when:**
- ✅ You know exact parameters
- ✅ You're scripting/automating
- ✅ You want full control
- ✅ You're integrating with other tools

**Both methods are fully supported and produce identical results!**
