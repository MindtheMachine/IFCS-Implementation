# Using Different LLM Providers

The Trilogy system supports multiple LLM providers with **zero code changes** - just update your `.env` file!

## 🎯 Quick Start

### 1. Choose Your Provider

Edit your `.env` file and set `LLM_PROVIDER`:

```bash
# Options: anthropic | openai | huggingface | ollama
LLM_PROVIDER=anthropic
```

### 2. Configure Model and API Key

```bash
# Set the model name (provider-specific)
LLM_MODEL=claude-sonnet-4-20250514

# Set your API key
LLM_API_KEY=your-api-key-here
```

### 3. Run!

```bash
python trilogy_app.py --prompt "What is quantum computing?"
```

Results will be organized in `Results/{model-name}/`

---

## ? Provider Capability Matrix

Some features are vendor-specific. The framework will pass options only when supported.

| Provider | temperature | top_p | seed | system | generate_batch |
|----------|-------------|-------|------|--------|----------------|
| Anthropic | Yes | Yes | No | Yes | Default (loop) |
| OpenAI | Yes | Yes | Yes | Yes | Default (loop) |
| HuggingFace | Yes | Yes | No | Yes | Default (loop) |
| Ollama | Yes | Yes | No | Yes | Default (loop) |

`generate_batch` currently uses repeated `generate()` calls; it can be upgraded later to true batch APIs where available.

---

## 📚 Provider-Specific Guides

### Anthropic Claude (Default)

**Models Available:**
- `claude-sonnet-4-20250514` (default, recommended)
- `claude-opus-4-20250514` (most capable)
- `claude-sonnet-3-5-20241022`
- `claude-haiku-3-5-20241022` (fastest, cheapest)

**.env Configuration:**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-...
```

**Get API Key:** https://console.anthropic.com/

**Install:** (already included)
```bash
pip install anthropic>=0.40.0
```

---

### OpenAI GPT-4

**Models Available:**
- `gpt-4-turbo` (recommended)
- `gpt-4o` (multimodal, fast)
- `gpt-4` (original)
- `gpt-3.5-turbo` (cheaper)

**.env Configuration:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-...
```

**Get API Key:** https://platform.openai.com/

**Install:**
```bash
pip install openai>=1.0.0
```

---

### HuggingFace Inference API

**Models Available:** (thousands on HuggingFace Hub)
- `meta-llama/Llama-3.1-70B-Instruct` (recommended)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-72B-Instruct`

**.env Configuration:**
```bash
LLM_PROVIDER=huggingface
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
LLM_API_KEY=hf_...
```

**Get API Key:** https://huggingface.co/settings/tokens

**Install:**
```bash
pip install huggingface-hub>=0.20.0
```

**Note:** Free tier has rate limits. Pro subscription recommended for benchmarks.

---

### Ollama (Local Models)

**Models Available:** (run locally on your machine)
- `llama3.1` (recommended, 8B/70B/405B variants)
- `mistral` (7B)
- `phi3` (3.8B, very fast)
- `qwen2.5` (7B/72B variants)
- `gemma2` (9B/27B)

**.env Configuration:**
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
# No API key needed!
OLLAMA_BASE_URL=http://localhost:11434
```

**Install Ollama:**
1. Download from https://ollama.com/
2. Install and start Ollama
3. Pull model: `ollama pull llama3.1`

**Install Python package:**
```bash
pip install ollama>=0.1.0
```

**Advantages:**
- ✅ Free (no API costs)
- ✅ Private (data stays local)
- ✅ No rate limits
- ✅ Offline capable

**Disadvantages:**
- ❌ Requires GPU for good performance
- ❌ Smaller models may be less capable

---

## 🔍 How Results Are Organized

Results are automatically organized by model name:

```
Results/
├── claude-sonnet-4-20250514/
│   ├── truthfulqa_results.csv
│   └── truthfulqa_summary.json
├── gpt-4-turbo/
│   ├── truthfulqa_results.csv
│   └── truthfulqa_summary.json
└── Llama-3.1-70B-Instruct/
    ├── truthfulqa_results.csv
    └── truthfulqa_summary.json
```

This makes it **easy to compare different models** side-by-side!

---

## 📊 Running Benchmarks with Different Models

### Example: Compare Claude vs GPT-4 on TruthfulQA

**Step 1: Run with Claude**
```bash
# .env: LLM_PROVIDER=anthropic, LLM_MODEL=claude-sonnet-4-20250514
python trilogy_app.py --benchmark truthfulqa --batch-size 50
```
→ Results saved to `Results/claude-sonnet-4-20250514/`

**Step 2: Switch to GPT-4**
```bash
# Edit .env: LLM_PROVIDER=openai, LLM_MODEL=gpt-4-turbo
python trilogy_app.py --benchmark truthfulqa --batch-size 50
```
→ Results saved to `Results/gpt-4-turbo/`

**Step 3: Compare Results**
```python
import pandas as pd

claude = pd.read_csv("Results/claude-sonnet-4-20250514/truthfulqa_results.csv")
gpt4 = pd.read_csv("Results/gpt-4-turbo/truthfulqa_results.csv")

print(f"Claude MC1: {claude['regulated_mc1_accuracy'].mean():.3f}")
print(f"GPT-4 MC1: {gpt4['regulated_mc1_accuracy'].mean():.3f}")
```

---

## ⚙️ Advanced: Model-Specific Tuning

Different models may benefit from different IFCS thresholds. Edit [trilogy_config.py](../trilogy_config.py):

```python
# For GPT-4 (tends to be more confident)
ifcs_rho = 0.35  # Lower threshold (fire more often)

# For smaller models (less confident)
ifcs_rho = 0.50  # Higher threshold (fire less often)
```

---

## 🐛 Troubleshooting

### Error: "Unknown LLM provider"
- Check `LLM_PROVIDER` is one of: `anthropic`, `openai`, `huggingface`, `ollama`
- Check for typos in `.env`

### Error: "API key required"
- Ensure `LLM_API_KEY` is set in `.env`
- For Anthropic, `ANTHROPIC_API_KEY` also works (backward compatibility)

### Error: "Module not found"
- Install the provider's package:
  ```bash
  pip install openai  # for OpenAI
  pip install huggingface-hub  # for HuggingFace
  pip install ollama  # for Ollama
  ```

### Ollama: "Connection refused"
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_BASE_URL` matches your Ollama server
- Verify model is pulled: `ollama list`

### HuggingFace: "Rate limit exceeded"
- Free tier has strict limits
- Consider HuggingFace Pro subscription
- Or use Ollama for unlimited local inference

---

## 💰 Cost Comparison

| Provider | Model | Cost (per 1M tokens) | Speed | Quality |
|----------|-------|---------------------|-------|---------|
| Anthropic | Claude Sonnet 4 | $3/$15 (in/out) | Fast | Excellent |
| Anthropic | Claude Opus 4 | $15/$75 (in/out) | Medium | Best |
| OpenAI | GPT-4 Turbo | $10/$30 (in/out) | Fast | Excellent |
| OpenAI | GPT-3.5 Turbo | $0.50/$1.50 (in/out) | Very Fast | Good |
| HuggingFace | Llama 3.1 70B | ~$0.65/$0.80 (in/out) | Medium | Very Good |
| Ollama | Local Llama 3.1 | **$0 (FREE)** | Fast* | Very Good |

\* Speed depends on your GPU

**For TruthfulQA full benchmark (817 examples):**
- Claude Sonnet 4: ~$5-10
- GPT-4 Turbo: ~$8-15
- Llama 3.1 (Ollama): **$0** (free)

---

## 🎓 Best Practices

1. **Start with Ollama** for testing (free, fast iteration)
2. **Use Claude/GPT-4** for production benchmarks (best quality)
3. **Compare multiple models** to validate trilogy effectiveness
4. **Monitor costs** with cloud providers (set billing alerts)
5. **Keep results organized** by model name for easy comparison

---

## 📖 See Also

- [SETUP.md](SETUP.md) - Installation guide
- [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - How benchmarks work
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - What you get as output
