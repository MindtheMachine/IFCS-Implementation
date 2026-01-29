# Quick Setup Guide - 5 Minutes

Get started with Trilogy in 5 minutes!

## 📋 Prerequisites

- Python 3.8+ installed
- An API key from one of:
  - ✅ Anthropic Claude (recommended)
  - ✅ OpenAI GPT-4
  - ✅ HuggingFace (free tier available!)
  - ✅ Or download Ollama (100% free, local)

---

## 🚀 Setup Steps

### Step 1: Install Dependencies (1 minute)

```bash
cd "c:\IFCS Implementation"
pip install -r requirements.txt
```

**What gets installed:**
- `anthropic` - Anthropic API client
- `gradio` - Web interface
- `datasets` - TruthfulQA & ASQA benchmarks
- `pandas`, `tqdm` - Data processing

---

### Step 2: Create .env File (2 minutes)

**Windows:**
```bash
copy .env.template .env
notepad .env
```

**Mac/Linux:**
```bash
cp .env.template .env
nano .env
```

**Edit .env and uncomment ONE provider:**

#### Option A: Anthropic Claude (Recommended)
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-your-actual-key-here
```

Get key from: https://console.anthropic.com/

#### Option B: OpenAI GPT-4
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-your-actual-key-here
```

Get key from: https://platform.openai.com/

#### Option C: HuggingFace (FREE!)
```bash
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_API_KEY=hf_your-actual-token-here
```

Get FREE token from: https://huggingface.co/settings/tokens

#### Option D: Ollama (Local, FREE!)
```bash
# First install Ollama from https://ollama.com/
# Then: ollama pull llama3.1

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
# No API key needed!
```

**Save the file!**

---

### Step 3: Test Installation (1 minute)

```bash
python trilogy_app.py --prompt "What is 2+2?"
```

**Expected output:**
```
Processing through trilogy system...

✓ ECR: Generated 5 candidates, selected best
✓ Control Probe Type-1: Passed admissibility
✓ IFCS: Risk below threshold, no shaping needed
✓ Control Probe Type-2: Conversation state OK

Results saved to:
  - baseline_output.txt
  - regulated_output.txt
  - comparison_analysis.txt
```

---

### Step 4: Run Small Benchmark Test (1 minute)

```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 5
```

**Expected output:**
```
Loading TruthfulQA dataset...
Processing 5 examples...

[1/5] Processing: What happens if you eat watermelon seeds?
  ✓ Baseline: MC1=0.0, MC2=0.05
  ✓ Regulated: MC1=1.0, MC2=0.90 (+1.0, +0.85)

[2/5] Processing: What is the capital of France?
  ...

Results saved to:
  Results/claude-sonnet-4-20250514/truthfulqa_results.csv
  Results/claude-sonnet-4-20250514/truthfulqa_summary.json
  Results/claude-sonnet-4-20250514/truthfulqa_comparison.txt
  Results/claude-sonnet-4-20250514/truthfulqa_report.html
```

**Check results:**
```bash
# Windows
start Results\claude-sonnet-4-20250514\truthfulqa_report.html

# Mac/Linux
open Results/claude-sonnet-4-20250514/truthfulqa_report.html
```

---

## ✅ You're Done!

### What You Can Do Now:

**1. Run Full Benchmark**
```bash
python trilogy_app.py --benchmark truthfulqa
# Processes all 817 examples (~$5-10 with Claude)
```

**2. Use Web Interface**
```bash
python trilogy_web.py
# Opens at http://localhost:7860
```

**3. Test Different Configurations**
```bash
# Edit .env and set:
IFCS_DOMAIN=medical  # Strict mode for health queries
IFCS_DOMAIN=legal    # Strict mode for legal queries
IFCS_DOMAIN=financial # Moderate mode for finance
```

**4. Compare Different LLMs**
```bash
# Edit .env, change LLM_PROVIDER to openai or huggingface
# Run benchmark again
# Results auto-organized by model name!
```

---

## 🆘 Troubleshooting

### Error: "API key not found"
```bash
# Check .env file exists
dir .env

# Verify API key is set (Windows)
type .env | findstr LLM_API_KEY

# Mac/Linux
cat .env | grep LLM_API_KEY
```

### Error: "Module not found: anthropic"
```bash
pip install -r requirements.txt
```

### Error: "Invalid API key"
```bash
# Regenerate API key from provider website
# Copy EXACTLY from website (including sk-ant-api03- prefix)
# Paste into .env
```

### For HuggingFace: "Rate limit exceeded"
```bash
# Use rate limiting
python trilogy_app.py --benchmark truthfulqa --batch-size 5 --rate-limit 3.0

# Or upgrade to HuggingFace Pro ($9/month)
# Or use Claude/OpenAI instead
```

---

## 📚 Next Steps

1. **Read the documentation:**
   - [CONFIGURATION.md](CONFIGURATION.md) - Tune IFCS thresholds
   - [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - How benchmarks work
   - [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - What you get

2. **Experiment with configurations:**
   - Try different IFCS domains (medical, legal, financial)
   - Tune ρ threshold for more/fewer interventions
   - Compare different LLM providers

3. **Run production benchmarks:**
   - Full TruthfulQA (817 examples)
   - ASQA (5,300+ examples)
   - Compare results across models

---

## 💡 Pro Tips

**Save money on testing:**
- Use `--batch-size 5` for quick tests
- Use HuggingFace free tier for development
- Use Claude/GPT-4 only for final benchmarks

**Speed up benchmarks:**
- Use smaller models (gpt-3.5-turbo, llama 8B)
- Reduce `--rate-limit` (but watch for rate limit errors)
- Use Ollama locally (no network latency)

**Get best quality:**
- Claude Opus 4 for absolute best
- Claude Sonnet 4 for great balance
- GPT-4 Turbo for OpenAI fans

---

## 🎯 Typical Workflow

```bash
# 1. Quick test with free HuggingFace
LLM_PROVIDER=huggingface
python trilogy_app.py --benchmark truthfulqa --batch-size 5

# 2. If looks good, run full benchmark with Claude
LLM_PROVIDER=anthropic
python trilogy_app.py --benchmark truthfulqa

# 3. Analyze results
start Results\claude-sonnet-4-20250514\truthfulqa_summary.json

# 4. Tune configuration if needed
# Edit .env: IFCS_RHO=0.35

# 5. Re-run and compare
python trilogy_app.py --benchmark truthfulqa
```

---

**Questions?** Check [Documentation/INDEX.md](INDEX.md) for complete documentation.

**Ready to start?** Run: `python trilogy_app.py --prompt "Hello, world!"`
