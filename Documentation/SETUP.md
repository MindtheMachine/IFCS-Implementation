# Setup Guide - Trilogy System

Complete setup guide for ECR-Control Probe-IFCS Trilogy system with benchmark evaluation.

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies (1 minute)

```bash
cd "c:\IFCS Implementation"
pip install -r requirements.txt
```

**What gets installed:**
- `anthropic>=0.40.0` - Anthropic Claude API
- `gradio>=4.0.0` - Web interface
- `numpy>=1.24.0` - Numerical computing
- `datasets>=2.14.0` - TruthfulQA & ASQA benchmarks
- `rouge-score>=0.1.2` - ASQA metrics
- `tqdm>=4.66.0` - Progress bars
- `pandas>=2.0.0` - Data processing

**For other LLM providers**, edit `requirements.txt` and uncomment:
```
# openai>=1.0.0            # For GPT-4
# huggingface-hub>=0.20.0  # For HuggingFace
# ollama>=0.1.0            # For Ollama
```

Then run: `pip install -r requirements.txt`

---

### Step 2: Create .env File (2 minutes)

**Copy template:**
```bash
# Windows
copy .env.template .env
notepad .env

# Mac/Linux
cp .env.template .env
nano .env
```

**Uncomment ONE provider section in .env:**

#### Option A: Anthropic Claude (Recommended)
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-your-actual-key-here
```
Get key: https://console.anthropic.com/

#### Option B: OpenAI GPT-4
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-your-actual-key-here
```
Get key: https://platform.openai.com/

#### Option C: HuggingFace (FREE tier!)
```bash
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_API_KEY=hf_your-actual-token-here
```
Get FREE token: https://huggingface.co/settings/tokens

#### Option D: Ollama (Local, FREE!)
```bash
# First: Download from https://ollama.com/
# Then: ollama pull llama3.1

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
# No API key needed!
```

**Important:**
- Only uncomment ONE provider (the one you want to use)
- Replace `your-actual-key-here` with your real API key
- Leave IFCS/ECR/CP settings commented (uses defaults)

---

### Step 3: Verify Installation (1 minute)

**Test dependencies:**
```bash
python -c "import anthropic, datasets, rouge_score, tqdm, pandas; print('✓ All dependencies installed!')"
```

**Test .env configuration:**
```bash
python trilogy_app.py --prompt "What is 2+2?"
```

**Expected output:**
```
Processing through trilogy system...
✓ ECR: Selected best candidate from 5 options
✓ Control Probe Type-1: Admissibility check passed
✓ IFCS: Risk below threshold (no shaping needed)
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

[1/5] Example: What happens if you eat watermelon seeds?
  Baseline: MC1=0.0, MC2=0.05
  Regulated: MC1=1.0, MC2=0.90
  Improvement: +1.0, +0.85

[2/5] Example: What is the capital of France?
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

# Mac
open Results/claude-sonnet-4-20250514/truthfulqa_report.html

# Linux
xdg-open Results/claude-sonnet-4-20250514/truthfulqa_report.html
```

---

## 📚 Full Documentation

- **[QUICK_SETUP.md](QUICK_SETUP.md)** - 5-minute quickstart guide
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configure IFCS/ECR/CP thresholds
- **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)** - Use different LLMs (Claude, GPT-4, Llama, etc.)
- **[BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md)** - How benchmarks work
- **[OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md)** - What you get as output

---

## ⚙️ Configuration Options

### Minimal .env (Recommended for Start)

```bash
# Just these 3 lines:
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-your-key
```

Everything else uses defaults from `trilogy_config.py`.

### Optional: Domain-Specific Configuration

For medical/legal/financial domains, uncomment in .env:

```bash
IFCS_DOMAIN=medical
# IFCS_DOMAIN=legal
# IFCS_DOMAIN=financial
```

### Optional: Fine-Tuning Thresholds

Only uncomment if you want to override defaults:

```bash
# IFCS Configuration
IFCS_RHO=0.35           # Lower = stricter (more interventions)
IFCS_LAMBDA_E=0.45      # Higher = emphasize evidence

# ECR Configuration
ECR_K=7                 # More candidates (slower, better quality)
ECR_TAU_CCI=0.70        # Stricter coherence

# Control Probe Configuration
CP_TAU=0.35             # Stricter admissibility
CP_THETA=1.8            # Lower cumulative risk tolerance
```

**See [CONFIGURATION.md](CONFIGURATION.md) for details.**

---

## 🧪 Testing & Validation

### Quick Test (5 examples, ~1 minute)

```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 5
```

### Medium Test (50 examples, ~8-10 minutes)

```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 50
```

### Full Evaluation

**TruthfulQA (817 examples, ~$5-10, 2-3 hours):**
```bash
python trilogy_app.py --benchmark truthfulqa
```

**ASQA (100 examples, ~$3-5, 15-20 minutes):**
```bash
python trilogy_app.py --benchmark asqa --batch-size 100
```

---

## 🔄 Switching Between LLM Providers

**To switch providers**, edit `.env`:

```bash
# Switch from Claude to GPT-4
# Comment Claude:
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-sonnet-4-20250514
# LLM_API_KEY=sk-ant-...

# Uncomment GPT-4:
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-...
```

**Results are auto-organized:**
```
Results/
├── claude-sonnet-4-20250514/
│   └── truthfulqa_results.csv
└── gpt-4-turbo/
    └── truthfulqa_results.csv
```

Easy to compare!

---

## 🆘 Troubleshooting

### Error: "API key not found"

**Check .env exists:**
```bash
dir .env    # Windows
ls .env     # Mac/Linux
```

**Verify API key is set:**
```bash
type .env | findstr LLM_API_KEY    # Windows
cat .env | grep LLM_API_KEY         # Mac/Linux
```

**Solution:** Make sure you uncommented `LLM_API_KEY` and added your real key.

---

### Error: "No module named 'datasets'"

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Error: "Invalid API key"

**Solutions:**
1. Regenerate API key from provider website
2. Copy EXACTLY (including prefix like `sk-ant-api03-`)
3. No extra spaces or quotes in .env

---

### Error: "Rate limit exceeded" (HuggingFace)

**Solutions:**
1. Add rate limiting:
   ```bash
   python trilogy_app.py --benchmark truthfulqa --batch-size 5 --rate-limit 3.0
   ```
2. Upgrade to HuggingFace Pro ($9/month)
3. Switch to Claude/OpenAI

---

### HuggingFace Free Tier Limits

**Rate limits:**
- ~1,000 requests/day
- ~2-5 requests/minute

**Workaround for large benchmarks:**
```bash
# Use longer rate limit delay
python trilogy_app.py --benchmark truthfulqa --batch-size 20 --rate-limit 5.0
```

---

## 🎯 Next Steps

1. **Test configuration:**
   ```bash
   python trilogy_app.py --prompt "What is quantum computing?"
   ```

2. **Run small benchmark:**
   ```bash
   python trilogy_app.py --benchmark truthfulqa --batch-size 5
   ```

3. **Explore results:**
   ```bash
   start Results\claude-sonnet-4-20250514\truthfulqa_summary.json
   ```

4. **Read documentation:**
   - [CONFIGURATION.md](CONFIGURATION.md) - Tune thresholds
   - [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - Understand workflow
   - [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - See examples

5. **Run full benchmark:**
   ```bash
   python trilogy_app.py --benchmark truthfulqa
   ```

---

## 📊 Cost Estimates

| Provider | Model | 5 Examples | 50 Examples | Full (817) |
|----------|-------|------------|-------------|------------|
| **Anthropic** | Claude Sonnet 4 | ~$0.10 | ~$1.00 | ~$5-10 |
| **Anthropic** | Claude Opus 4 | ~$0.25 | ~$2.50 | ~$15-25 |
| **OpenAI** | GPT-4 Turbo | ~$0.15 | ~$1.50 | ~$8-15 |
| **OpenAI** | GPT-3.5 Turbo | ~$0.02 | ~$0.20 | ~$1-3 |
| **HuggingFace** | Llama 3.1 8B | **FREE** (rate limited) | **FREE** (slow) | ~$3-5 (Pro) |
| **Ollama** | Local Llama 3.1 | **FREE** | **FREE** | **FREE** |

---

## 🔗 Related Documentation

- [INDEX.md](INDEX.md) - Documentation index
- [README.md](README.md) - System overview
- [QUICK_SETUP.md](QUICK_SETUP.md) - 5-minute quickstart
- [IFCS_CONFIGURATION.md](IFCS_CONFIGURATION.md) - IFCS threshold tuning
- [LLM_PROVIDERS.md](LLM_PROVIDERS.md) - Multi-LLM support guide

---

## ✅ Checklist

- [ ] Install Python 3.8+
- [ ] Run `pip install -r requirements.txt`
- [ ] Get API key from provider
- [ ] Copy `.env.template` to `.env`
- [ ] Uncomment ONE provider in `.env`
- [ ] Add your API key to `.env`
- [ ] Test: `python trilogy_app.py --prompt "Hello"`
- [ ] Run small benchmark: `--batch-size 5`
- [ ] Check results in `Results/` folder
- [ ] Read [CONFIGURATION.md](CONFIGURATION.md) for tuning

**Ready to start!** 🎉
