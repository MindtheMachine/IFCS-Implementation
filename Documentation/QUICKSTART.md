# Quick Start Guide - Trilogy System

Get started with the ECR-Control Probe-IFCS Trilogy System in minutes!

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- Anthropic/OpenAI/HuggingFace API clients
- Gradio web interface
- Benchmark evaluation tools (datasets, rouge-score)
- Data processing libraries (pandas, tqdm)

### Step 2: Configure LLM Provider

**Create .env file:**
```bash
# Windows
copy .env.template .env
notepad .env

# Mac/Linux
cp .env.template .env
nano .env
```

**Uncomment ONE provider in .env:**

**Option A - Anthropic Claude (Recommended)**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-your-actual-key-here
```
Get key from: https://console.anthropic.com/

**Option B - OpenAI GPT-4**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-your-actual-key-here
```
Get key from: https://platform.openai.com/

**Option C - HuggingFace (FREE!)**
```bash
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_API_KEY=hf_your-actual-token-here
```
Get FREE token from: https://huggingface.co/settings/tokens

**Option D - Ollama (Local, FREE!)**
```bash
# First: Download from https://ollama.com/
# Then: ollama pull llama3.1

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
# No API key needed!
```

### Step 3: Launch

**Windows (Recommended) - Use Batch Files:**
```bash
run.bat              # Interactive menu
run_web.bat          # Web interface
run_benchmark.bat    # Benchmark evaluation
```

**Web Interface (All Platforms):**
```bash
python trilogy_web.py
```
Then open: http://localhost:7860

**Command Line:**
```bash
# Single query
python trilogy_app.py --prompt "Your question here"

# Run benchmark test
python trilogy_app.py --benchmark truthfulqa --batch-size 5
```

---

## 📱 Using the Web Interface

### Initialize System
1. System auto-loads configuration from .env
2. Adjust parameters if desired (defaults work well)
3. Click "🔧 Initialize System"
4. Wait for "✅ System initialized successfully!"

### Process a Query
1. Enter your prompt in the text box
2. (Optional) Add context
3. Click "▶️ Process Query"
4. View baseline vs regulated outputs side-by-side
5. Read comparison analysis with mechanism firing details

### Try Test Cases
1. Go to "📋 Test Cases" tab
2. Select a test case from dropdown (36 failure modes from papers)
3. Click "📥 Load Test Case"
4. Click "▶️ Run Test"
5. Compare expected vs actual mechanism firing

### Batch Processing
1. Go to "📤 Batch Processing" tab
2. Create a `.txt` file with prompts (one per line)
3. Upload the file
4. Click "🔄 Process Batch"
5. Download results as JSON

---

## 🎯 Example Queries to Try

### General Knowledge (Low Risk)
```
What is Python programming used for?
```
**Expected:** Minimal intervention (R < ρ), information-rich response

### Medical (High Risk)
```
I have chest pain after exercise. What is it?
```
**Expected:** IFCS fires with strict threshold (ρ=0.30), adds safety disclaimers

### Ambiguous Query
```
What's the best way to implement a tree?
```
**Expected:** IFCS fires, surfaces ambiguity (data structure vs physical tree vs decision tree)

### Temporal Query
```
Should I invest in cryptocurrency right now?
```
**Expected:** IFCS fires, adds temporal boundaries and disclaimers

### Causal Confusion
```
Ice cream sales and shark attacks both peak in July. Does ice cream cause shark attacks?
```
**Expected:** ECR selects coherent response that clarifies correlation ≠ causation

### Overconfident Advice
```
What programming language should I learn first?
```
**Expected:** IFCS reduces universal claims, acknowledges context-dependence

---

## 🧪 Running Benchmarks

### Quick Test (5 examples, ~1 minute)
```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 5
```

### Medium Test (50 examples, ~8-10 minutes)
```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 50
```

### Full Evaluation (817 examples, ~2-3 hours)
```bash
python trilogy_app.py --benchmark truthfulqa
```

### View Results
```bash
# Windows
start Results\claude-sonnet-4-20250514\truthfulqa_report.html

# Mac
open Results/claude-sonnet-4-20250514/truthfulqa_report.html

# Linux
xdg-open Results/claude-sonnet-4-20250514/truthfulqa_report.html
```

**Results include:**
- CSV with per-example metrics
- JSON with summary statistics
- HTML with visualizations
- TXT with baseline vs regulated comparison
- Mechanism firing analysis

---

## 🔧 Understanding Parameters

### Domain Presets (Easiest)

Set in .env for automatic calibration:
```bash
IFCS_DOMAIN=medical     # Strict (ρ=0.30) for health queries
IFCS_DOMAIN=legal       # Strict (ρ=0.30) for legal queries
IFCS_DOMAIN=financial   # Moderate (ρ=0.35) for finance
IFCS_DOMAIN=default     # Balanced (ρ=0.40) general use
```

### Manual Tuning

**ECR Configuration:**
- `ECR_K`: Number of candidates (3-7, default: 5)
  - Higher = more thorough, slower
- `ECR_H`: Horizon steps (2-4, default: 3)
  - Higher = deeper lookahead
- `ECR_TAU_CCI`: Coherence threshold (0.0-1.0, default: 0.65)
  - Higher = stricter coherence requirements

**IFCS Configuration:**
- `IFCS_RHO`: Commitment threshold (0.0-1.0, default: 0.40)
  - Lower = more interventions (stricter)
  - Higher = fewer interventions (more permissive)
- `IFCS_LAMBDA_E`: Evidential weight (0.0-1.0, default: 0.40)
  - Higher = emphasize evidence/grounding
- `IFCS_LAMBDA_S`: Scope weight (0.0-1.0, default: 0.30)
  - Higher = penalize universal claims
- `IFCS_LAMBDA_A`: Authority weight (0.0-1.0, default: 0.30)
  - Higher = penalize directive language

**Control Probe Configuration:**
- `CP_TAU`: Admissibility threshold (0.0-1.0, default: 0.40)
  - Lower = blocks more responses (safer)
- `CP_THETA`: Cumulative risk threshold (0.5-3.0, default: 2.0)
  - Lower = detects drift earlier

See [CONFIGURATION.md](CONFIGURATION.md) for complete tuning guide.

---

## 📊 Reading the Output

### Commitment Risk Scores

| Component | Meaning | High Value Indicates |
|-----------|---------|---------------------|
| `ê` | Evidential insufficiency | Claims exceed grounding |
| `ŝ` | Scope inflation | Too many absolutes |
| `â` | Authority cues | Too many imperatives |
| `R(z)` | Overall risk | Intervention needed |

### Mechanism Decisions

| Symbol | Meaning |
|--------|---------|
| ✅ | Mechanism fired (intervention applied) |
| ⭕ | Mechanism did not fire (no intervention) |
| ⚠️ | Warning/blocking action |

### Example Analysis
```
R(z) = 0.56 = 0.40×0.6 + 0.30×0.7 + 0.30×0.6
       ↑       ↑  ↑      ↑  ↑      ↑  ↑
       Risk    λe ê      λs ŝ      λa â

R(z)=0.56 > ρ=0.40 → IFCS FIRES
After shaping: R(z')=0.22
Reduction: 61%
```

---

## 🔄 Switching LLM Providers

Simply edit `.env` and change the provider:

**Switch from Claude to GPT-4:**
```bash
# Comment Claude
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-sonnet-4-20250514
# LLM_API_KEY=sk-ant-...

# Uncomment GPT-4
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-...
```

**Results are automatically organized:**
```
Results/
├── claude-sonnet-4-20250514/
│   └── truthfulqa_results.csv
└── gpt-4-turbo/
    └── truthfulqa_results.csv
```

See [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for complete provider guide.

---

## ❓ Troubleshooting

### "API key not found"
**Check .env file exists:**
```bash
dir .env    # Windows
ls .env     # Mac/Linux
```

**Verify API key is set:**
```bash
type .env | findstr LLM_API_KEY    # Windows
cat .env | grep LLM_API_KEY         # Mac/Linux
```

**Solution:** Make sure you uncommented the 3 lines (PROVIDER, MODEL, API_KEY) for your chosen provider.

### "No module named 'datasets'"
```bash
pip install -r requirements.txt
```

### "Invalid API key"
1. Regenerate API key from provider website
2. Copy EXACTLY (including prefix like `sk-ant-api03-`)
3. No extra spaces or quotes in .env

### "Rate limit exceeded" (HuggingFace)
```bash
# Use rate limiting
python trilogy_app.py --benchmark truthfulqa --batch-size 5 --rate-limit 3.0

# Or upgrade to HuggingFace Pro ($9/month)
# Or switch to Claude/OpenAI
```

### Slow processing
- Normal! Each query processes K×H candidates
- For TruthfulQA: 817 examples × 10s/example ≈ 2-3 hours
- Reduce K or H if needed (faster but less thorough)
- Use `--batch-size` to test smaller subsets first

### Unexpected mechanism firing
- Check risk scores in analysis output
- Review domain detection (medical/legal/financial keywords)
- Adjust thresholds in .env if needed
- Compare with test cases for similar examples

---

## 💡 Tips

1. **Start with defaults** - they work well for most cases based on paper recommendations
2. **Use test cases** - learn how mechanisms behave on 36 failure modes
3. **Compare outputs** - see exactly what changed and why
4. **Read the papers** - understand the theory behind each mechanism
5. **Experiment with domains** - try medical vs default threshold
6. **Test on small batches** - use `--batch-size 5` before full runs
7. **Check results folder** - organized by model name for easy comparison

---

## 📚 Next Steps

### Level 1: Basic Usage
1. ✅ Try single queries via web interface
2. ✅ Load and run test cases from taxonomy
3. ✅ Compare baseline vs regulated outputs
4. ✅ Run small benchmark test (5 examples)

### Level 2: Configuration
1. 📖 Read [CONFIGURATION.md](CONFIGURATION.md)
2. 🔧 Try different domain presets (medical, legal, financial)
3. 🎚️ Tune individual thresholds
4. 🔄 Compare different LLM providers

### Level 3: Research & Evaluation
1. 📊 Run full benchmark evaluations
2. 📈 Analyze mechanism firing patterns
3. 📄 Read the three research papers
4. 🔬 Experiment with custom thresholds

### Level 4: Advanced
1. 💻 Explore implementation code
2. 🧪 Create custom test cases
3. 📝 Contribute improvements
4. 🚀 Deploy to production

---

## 📖 Documentation Quick Links

- [../README.md](../README.md) - Main documentation (in root folder)
- [SETUP.md](SETUP.md) - Complete setup guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration and tuning
- [LLM_PROVIDERS.md](LLM_PROVIDERS.md) - Multi-LLM support
- [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - Benchmark evaluation
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - Output format examples
- [INDEX.md](INDEX.md) - Documentation index

---

**Happy testing! 🎉**

For detailed documentation, see [INDEX.md](INDEX.md) or the papers in the root directory.
