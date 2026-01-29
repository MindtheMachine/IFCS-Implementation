# Trilogy System - Documentation Index

Welcome to the ECR-Control Probe-IFCS Trilogy System documentation. This index will help you find the right documentation for your needs.

## 🚀 Getting Started (Start Here!)

### For Windows Users
1. **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Complete Windows setup guide
2. **[WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)** - Quick start for Windows
3. **[SETUP.md](SETUP.md)** - Installation and configuration guide

### For All Users
1. **[README.md](../README.md)** - System overview and main documentation (in root folder)
2. **[QUICK_SETUP.md](QUICK_SETUP.md)** - 5-minute quick start guide
3. **[SETUP.md](SETUP.md)** - Complete setup and installation guide
4. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment options (Replit, local)

## 📚 Core Documentation

### System Understanding
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation details
  - What's implemented
  - Design decisions
  - Validation results
  - Next steps for research

- **[TRANSITION_GUIDE.md](TRANSITION_GUIDE.md)** - Migration and transition information
  - Moving from previous versions
  - Understanding changes

### Integration Guides
- **[CLAUDE_CODE_SETUP.md](CLAUDE_CODE_SETUP.md)** - Using with Claude Code
- **[GITHUB_SETUP.md](GITHUB_SETUP.md)** - GitHub integration and version control

## 🎯 Feature-Specific Documentation

### Launcher Scripts (NEW!)

**📚 Batch Files & Shell Scripts:**
- **[LAUNCHER_SCRIPTS.md](LAUNCHER_SCRIPTS.md)** - Complete launcher scripts guide
  - Windows batch files (.bat) and Linux/Mac shell scripts (.sh)
  - Interactive menus, web interface, benchmark evaluation
  - Automatic dependency installation and API key validation
  - Troubleshooting and best practices

### Configuration (NEW!)

**📚 Complete Configuration Guides:**
- **[CONFIGURATION.md](CONFIGURATION.md)** - Complete configuration guide
  - Domain presets (medical, legal, financial, default)
  - .env configuration (easiest)
  - JSON/YAML configuration (advanced)
  - Configuration recipes and examples
- **[IFCS_CONFIGURATION.md](IFCS_CONFIGURATION.md)** - IFCS threshold tuning deep dive
- **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)** - Using different LLM providers (Claude, GPT-4, Llama, etc.)

**Quick Reference:**
```bash
# Simple: Set ρ in .env
IFCS_DOMAIN=medical  # or legal, financial, default

# Advanced: Fine-tune thresholds
IFCS_RHO=0.35
IFCS_LAMBDA_E=0.45
ECR_TAU_CCI=0.70
CP_TAU=0.38
```

### Benchmark Evaluation (NEW!)

**📚 Complete Documentation:**
- **[SETUP.md](SETUP.md)** - Installation and configuration
- **[BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md)** - Detailed step-by-step process from input to output
- **[OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)** - Examples of what you get (CSV, JSON, HTML, TXT)

**Quick Reference:**
```bash
# Using batch files (Windows)
run.bat                          # Interactive menu
run_benchmark.bat               # Benchmark evaluation menu
run_web.bat                     # Web interface

# Using Python directly
python trilogy_app.py --benchmark truthfulqa --batch-size 5
python trilogy_app.py --benchmark asqa --batch-size 5
```

**What's Included:**
- TruthfulQA benchmark (817 examples)
- ASQA benchmark (5,300+ examples)
- MC1/MC2 metrics for TruthfulQA
- DR score (√(Disambig-F1 × ROUGE-L)) for ASQA
- CSV, JSON, HTML, and TXT reports
- Complete baseline vs regulated comparison
- Mechanism firing details (what fired, when, why)
- Checkpointing and resume capability

## 📖 Documentation by Use Case

### "I just want to try the system"
1. [WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md) or [QUICKSTART.md](QUICKSTART.md)
2. Run `run.bat` (Windows) or `./run.sh` (Linux/Mac)
3. Or: `python trilogy_web.py`

### "I want to run benchmark evaluations"
1. [SETUP.md](SETUP.md) - Setup guide
2. Run `run_benchmark.bat` (Windows) or `./run_benchmark.sh` (Linux/Mac)
3. Or: `python trilogy_app.py --benchmark truthfulqa --batch-size 5`

### "I want to understand how it works"
1. [README.md](../README.md) - System overview (in root folder)
2. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
3. Read the three research papers (in root directory)

### "I want to deploy this system"
1. [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment options
2. [GITHUB_SETUP.md](GITHUB_SETUP.md) - Version control setup
3. [SETUP.md](SETUP.md) - Environment configuration

### "I want to contribute or extend"
1. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Architecture
2. [GITHUB_SETUP.md](GITHUB_SETUP.md) - Git workflow
3. Plan file: `C:\Users\achatt0\.claude\plans\spicy-hatching-hamster.md`

## 🗂️ File Organization

```
c:\IFCS Implementation\
├── README.md                  # Main documentation (ROOT FOLDER)
├── Documentation\             # All other documentation (YOU ARE HERE)
│   ├── INDEX.md              # This file - documentation index
│   ├── SETUP.md              # Complete setup guide
│   ├── QUICK_SETUP.md        # 5-minute quick start
│   ├── CONFIGURATION.md      # Configuration guide
│   ├── IFCS_CONFIGURATION.md # IFCS threshold tuning
│   ├── LLM_PROVIDERS.md      # Multi-LLM support guide
│   ├── BENCHMARK_WORKFLOW.md # Benchmark evaluation workflow
│   ├── OUTPUT_EXAMPLES.md    # Output format examples
│   ├── IMPLEMENTATION_SUMMARY.md  # Implementation details
│   ├── DEPLOYMENT.md         # Deployment guide
│   ├── WINDOWS_SETUP.md      # Windows-specific setup
│   ├── WINDOWS_QUICKSTART.md # Windows quick start
│   ├── GITHUB_SETUP.md       # Git/GitHub guide
│   ├── CLAUDE_CODE_SETUP.md  # Claude Code integration
│   └── TRANSITION_GUIDE.md   # Migration guide
│
├── *.py                      # Python implementation files
├── *.bat                     # Windows batch launchers
├── *.sh                      # Linux/Mac shell scripts
├── requirements.txt          # Dependencies
├── .env.template             # API key template
└── .gitignore                # Git ignore rules
```

## 🔧 Quick Commands Reference

### Windows
```batch
run.bat                  # Main menu
run_web.bat             # Web interface
run_benchmark.bat       # Benchmark evaluation
setup_github.bat        # GitHub setup
```

### Linux/Mac
```bash
./run.sh                 # Main menu
./run_web.sh            # Web interface
./run_benchmark.sh      # Benchmark evaluation
./setup_github.sh       # GitHub setup
```

### Python Direct (All Platforms)
```bash
# Web interface
python trilogy_web.py

# Single query
python trilogy_app.py --prompt "Your question here"

# Test suite
python trilogy_app.py --test-suite

# Benchmark evaluation
python trilogy_app.py --benchmark truthfulqa --batch-size 5
python trilogy_app.py --benchmark asqa --batch-size 5
```

## 📊 Benchmark Datasets

### TruthfulQA
- **Purpose**: Measuring truthfulness in LLM responses
- **Size**: 817 multiple-choice questions across 38 categories
- **Metrics**: MC1 (accuracy), MC2 (calibrated probability)
- **License**: Apache 2.0
- **Source**: https://huggingface.co/datasets/truthfulqa/truthful_qa

### ASQA
- **Purpose**: Ambiguous factoid question answering
- **Size**: 5,300+ questions requiring long-form answers
- **Metrics**: DR score = √(Disambig-F1 × ROUGE-L)
- **License**: Apache 2.0
- **Source**: https://huggingface.co/datasets/din0s/asqa

## 🆘 Getting Help

### Common Issues
See [SETUP.md](SETUP.md) for troubleshooting:
- ImportError issues
- API key problems
- Rate limiting
- Memory issues

### Documentation Questions
- System architecture → [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Quick setup → [QUICKSTART.md](QUICKSTART.md) or [WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)
- Deployment → [DEPLOYMENT.md](DEPLOYMENT.md)
- Git workflow → [GITHUB_SETUP.md](GITHUB_SETUP.md)

### Research Papers
The three foundational papers are in the root directory:
1. Control Probe paper (CP-Type-1 and CP-Type-2)
2. ECR paper (Evaluative Coherence Regulation)
3. IFCS paper (Inference-Time Commitment Shaping)

## 📝 Version Information

- **Current Version**: 1.1.0 (with benchmark evaluation)
- **License**: Apache 2.0
- **Author**: Arijit Chatterjee (ORCID: 0009-0006-5658-4449)
- **Repository**: https://github.com/MindtheMachine/IFCS-Implementation

## 🔄 Recent Updates

### Version 1.1.0 - Benchmark Integration (2026-01-29)
- ✅ Added TruthfulQA benchmark evaluation
- ✅ Added ASQA benchmark evaluation
- ✅ Implemented MC1, MC2, DR score metrics
- ✅ Added CSV, JSON, HTML reporting
- ✅ Checkpointing for long-running evaluations
- ✅ Windows batch file launchers
- ✅ Comprehensive setup documentation

### Version 1.0.0 - Initial Implementation
- ✅ ECR (Evaluative Coherence Regulation)
- ✅ Control Probe Type-1 and Type-2
- ✅ IFCS (Inference-Time Commitment Shaping)
- ✅ 36 test cases from taxonomy
- ✅ Web interface (Gradio)
- ✅ Command-line interface

---

**Need help?** Start with [QUICKSTART.md](QUICKSTART.md) or [WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)!
