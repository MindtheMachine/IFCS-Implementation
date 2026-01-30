# ECR-Control Probe-IFCS Trilogy System

Implementation of Arijit Chatterjee's three-paper trilogy on inference-time governance mechanisms for Large Language Models.

## 📚 Papers

1. **Control Probe**: Inference-Time Commitment Control (Type-1 and Type-2)
2. **ECR**: Evaluative Coherence Regulation - An Inference-Time Stability Layer
3. **IFCS**: Inference-Time Commitment Shaping - A Framework for Quiet Failure Mitigation

## 🎯 What This System Does

The trilogy provides a complete inference-time governance architecture that addresses three orthogonal control dimensions:

| Mechanism | Question Answered | Failure Class | Operation |
|-----------|------------------|---------------|-----------|
| **ECR** | Which continuation? | Selection failures | Comparative coherence |
| **Control Probe** | Whether to commit? | Illegitimate commitment | Admissibility gating |
| **IFCS** | How strongly to commit? | Commitment inflation | Modality regulation |

### Pipeline Order (Non-bypassable)
```
User Query
    ↓
ECR: Generate K candidates → Select most coherent
    ↓
Control Probe Type-1: Check σ(z) ≥ τ → PASS or BLOCK
    ↓
IFCS: Compute R(z) → Shape commitment if R(z) > ρ
    ↓
Output to User
    ↓
Control Probe Type-2: Monitor R_cum(H) → Detect drift/sycophancy
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API key from **ONE** of:
  - ✅ [Anthropic Claude](https://console.anthropic.com/) (recommended)
  - ✅ [OpenAI GPT-4](https://platform.openai.com/)
  - ✅ [HuggingFace](https://huggingface.co/settings/tokens) (FREE tier available!)
  - ✅ [Ollama](https://ollama.com/) (100% FREE, runs locally)

### Installation (5 Minutes)

**Windows Users** - Use batch files:
```bash
# Option 1: Interactive menu
run.bat

# Option 2: Direct launch
run_web.bat              # Web interface
run_benchmark.bat        # Benchmark evaluation
```

**Linux/Mac Users** - Use shell scripts:
```bash
# Option 1: Interactive menu
./run.sh

# Option 2: Direct launch
./run_web.sh             # Web interface
./run_benchmark.sh       # Benchmark evaluation
```

**Manual Setup** (all platforms):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
copy .env.template .env    # Windows
cp .env.template .env      # Mac/Linux

# 3. Edit .env and uncomment ONE provider
notepad .env               # Windows
nano .env                  # Mac/Linux
```

**Example .env configuration** (Anthropic Claude):
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-api03-your-actual-key-here
```

See [Documentation/SETUP.md](Documentation/SETUP.md) for detailed setup guide.

### Web Interface (Recommended)

**Using launcher scripts:**
```bash
# Windows
run_web.bat

# Linux/Mac
./run_web.sh
```

**Or directly:**
```bash
python trilogy_web.py
```

Open browser to `http://localhost:7860`

### Command Line

**Using launcher scripts:**
```bash
# Windows - Interactive menu
run.bat

# Linux/Mac - Interactive menu
./run.sh
```

**Or directly:**
```bash
# Single query
python trilogy_app.py --prompt "What is the best programming language?"

# Run benchmark evaluation
python trilogy_app.py --benchmark truthfulqa --batch-size 5

# Run full benchmark
python trilogy_app.py --benchmark truthfulqa
```

## ✨ New Features

### 🔌 Multi-LLM Provider Support

Switch between LLM providers with **zero code changes** - just edit `.env`:

- **Anthropic Claude** - Best quality, recommended
- **OpenAI GPT-4** - Excellent quality
- **HuggingFace** - FREE tier available (Llama, Mistral, Qwen, Gemma)
- **Ollama** - 100% FREE, runs locally on your machine

Example - switch from Claude to GPT-4:
```bash
# In .env file:
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-proj-your-openai-key
```

Results are automatically organized by model:
```
Results/
├── claude-sonnet-4-20250514/
│   └── truthfulqa_results.csv
└── gpt-4-turbo/
    └── truthfulqa_results.csv
```

Provider capabilities vary slightly (e.g., `seed` support), and the framework only passes settings supported by each backend.

See [Documentation/LLM_PROVIDERS.md](Documentation/LLM_PROVIDERS.md) for complete guide.

### 📊 Benchmark Evaluation

Industry-standard benchmark support:

- **TruthfulQA** (817 questions) - Measures truthfulness and calibration
  - Metrics: MC1 (accuracy), MC2 (calibrated probability)
  - Detects common human falsehoods

- **ASQA** (5,300+ questions) - Ambiguous question answering
  - Metrics: DR score = √(Disambig-F1 × ROUGE-L)
  - Tests handling of ambiguous queries

**Quick Test** (5 examples, ~1 minute):
```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 5
```

**Full Evaluation** (817 examples, ~2-3 hours):
```bash
python trilogy_app.py --benchmark truthfulqa
```

**View Results**:
```bash
# Windows
start Results\claude-sonnet-4-20250514\truthfulqa_report.html

# Mac/Linux
open Results/claude-sonnet-4-20250514/truthfulqa_report.html
```

See [Documentation/BENCHMARK_WORKFLOW.md](Documentation/BENCHMARK_WORKFLOW.md) for complete workflow.

### ⚙️ Flexible Configuration System

Configure IFCS thresholds via **multiple sources** with priority:

1. **Environment variables** (.env) - Highest priority
2. **JSON config** (trilogy_config.json)
3. **YAML config** (trilogy_config.yaml)
4. **Python defaults** (trilogy_config.py) - Lowest priority

**Domain Presets** (automatic calibration):
```bash
# In .env file:
IFCS_DOMAIN=medical      # Strict (ρ=0.30) for health queries
IFCS_DOMAIN=legal        # Strict (ρ=0.30) for legal queries
IFCS_DOMAIN=financial    # Moderate (ρ=0.35) for finance
IFCS_DOMAIN=default      # Balanced (ρ=0.40) general use
```

**Manual Tuning**:
```bash
# In .env file:
IFCS_RHO=0.35           # Lower = stricter (more interventions)
IFCS_LAMBDA_E=0.45      # Higher = emphasize evidence
ECR_K=7                 # More candidates (slower, better quality)
CP_TAU=0.35             # Stricter admissibility
```

See [Documentation/CONFIGURATION.md](Documentation/CONFIGURATION.md) for tuning guide.

## 📋 Features

### 1. Single Query Processing
- Input any prompt
- View baseline (unregulated) output
- View trilogy (regulated) output
- See side-by-side comparison
- Mechanism firing analysis

### 2. Test Cases from Taxonomy
- 36 quiet failure modes from the papers
- Organized by category:
  - Selection-dominant (ECR)
  - Commitment-inflation (IFCS)
  - Illegitimate commitment (Control Probe)
- Load and test any case
- Verify expected mechanisms fire

### 3. Batch Processing
- Upload `.txt` file with prompts (one per line)
- Process all queries automatically
- Download results as JSON
- Full comparison analysis

### 4. Configurable Parameters

#### ECR Configuration
- `ECR_K`: Number of candidates (3-7, default: 5)
- `ECR_H`: Horizon steps (2-4, default: 3)
- `ECR_TAU_CCI`: Coherence threshold (0.0-1.0, default: 0.65)

#### Control Probe Configuration
- `CP_TAU`: Admissibility threshold (0.0-1.0, default: 0.40)
- `CP_THETA`: Cumulative risk threshold (0.5-3.0, default: 2.0)

#### IFCS Configuration
- `IFCS_RHO`: Commitment threshold (0.0-1.0, default: 0.40)
- `IFCS_LAMBDA_E`: Evidential weight (0.0-1.0, default: 0.40)
- `IFCS_LAMBDA_S`: Scope weight (0.0-1.0, default: 0.30)
- `IFCS_LAMBDA_A`: Authority weight (0.0-1.0, default: 0.30)
- `IFCS_LAMBDA_T`: Temporal weight (0.0-1.0, default: 0.00)

## 📊 Understanding the Output

### Baseline vs Regulated Comparison

The system shows:

1. **Mechanisms Fired**
   - ✅ ECR: Always fires (selects best candidate)
   - ⚠️ CP Type-1: Fires when output is inadmissible (blocks)
   - ✅ IFCS: Fires when commitment exceeds threshold
   - ⚠️ CP Type-2: Fires when interaction risk accumulates

2. **Commitment Marker Reduction**
   - Universal markers: `always`, `never`, `all`, `every` → reduced
   - Authority markers: `must`, `should`, `need to` → attenuated
   - Certainty markers: `definitely`, `clearly`, `obviously` → weakened

3. **Risk Scores**
   - `ê`: Evidential insufficiency [0-1]
   - `ŝ`: Scope inflation [0-1]
   - `â`: Authority cues [0-1]
   - `R(z)`: Overall commitment risk [0-1]

### Example Output

```
Prompt: "I have chest pain after exercise. What is it?"
Baseline:
"This is likely acid reflux or muscle strain. Take ibuprofen and an antacid.
You'll be fine in a few days."

Risk: R=0.56 (ê=0.6, ŝ=0.7, â=0.6) >> ρ=0.30

Regulated (IFCS shaped):
"⚠ Chest pain requires medical evaluation. I cannot diagnose your symptoms.
Chest pain with exercise can have multiple causes ranging from benign
(musculoskeletal) to serious (cardiac issues, pulmonary embolism).

If pain is severe, sudden, or accompanied by shortness of breath, nausea,
or sweating → Call emergency services immediately.

Otherwise: Schedule urgent appointment with your doctor TODAY."

Risk after shaping: R=0.07
Commitment reduction: 87%
```

## 🧪 Test Cases

The system includes all 36 test cases from the taxonomy:

### Category 1: Selection-Dominant (ECR Primary)
- Point-in-time concept drift
- Compositional drift
- Causal confusion
- Availability bias
- Frequency bias
- Framing bias
- ...and more

### Category 2: Commitment-Inflation (IFCS Primary)
- Early authority gradient
- Ambiguity collapse
- Domain-specific overconfidence (medical, legal)
- Fragile RAG grounding
- Temporal grounding failure
- Confidence miscalibration
- ...and more

### Category 3: Illegitimate Commitment (Control Probe Primary)
- Fabricated facts (Type-1)
- Capability misrepresentation (Type-1)
- Premature closure (Type-1)
- Sycophancy (Type-2)
- Authority laundering (Type-2)
- Semantic drift (Type-2)
- ...and more

### Latest 36-Test Run Summary (gpt-4o-mini, vendor defaults, seed=12345)
- Results saved to `test_results.json`
- Matches: 20/36
- Mismatches: 16/36
- By expected mechanism: ECR 10/10, IFCS 8/13, CP-Type-1 2/4, CP-Type-2 0/7, Lifecycle 0/2
- Mismatched IDs: 2.4, 2.5, 2.8, 2.9, 2.12, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2

## 🔧 Architecture

```
trilogy_app.py              # Main application with benchmark support
trilogy_web.py              # Gradio web interface
trilogy_orchestrator.py     # Pipeline coordination
trilogy_config_loader.py    # Multi-source configuration loader

ecr_engine.py              # ECR implementation
control_probe.py           # Control Probe Type-1 and Type-2
ifcs_engine.py             # IFCS implementation

llm_provider.py            # Multi-LLM abstraction layer

benchmark_loader.py        # Dataset loading (TruthfulQA, ASQA)
benchmark_adapters.py      # Benchmark-specific formatting
benchmark_metrics.py       # MC1, MC2, DR score computation
benchmark_orchestrator.py  # Batch processing with checkpointing
benchmark_reports.py       # CSV, JSON, HTML report generation
benchmark_config.py        # Benchmark configuration

requirements.txt           # Dependencies
.env.template             # Environment configuration template
README.md                 # This file
```

## 📝 Output Files

### Single Query Outputs
When processing queries, the system creates:

1. `baseline_output.txt` - Unregulated LLM output
2. `regulated_output.txt` - Trilogy-regulated output
3. `comparison_analysis.txt` - Side-by-side comparison with metrics

### Benchmark Outputs

Results are organized by model name in `Results/[model-name]/`:

1. **CSV** - Per-example detailed results
   - `truthfulqa_results.csv` or `asqa_results.csv`
   - Columns: prompt, baseline_mc1, regulated_mc1, improvement, mechanisms_fired

2. **JSON** - Summary statistics
   - `truthfulqa_summary.json` or `asqa_summary.json`
   - Overall metrics, per-mechanism performance, firing rates

3. **TXT** - Comparison report
   - `truthfulqa_comparison.txt` or `asqa_comparison.txt`
   - Side-by-side baseline vs regulated analysis

4. **HTML** - Interactive visualizations
   - `truthfulqa_report.html` or `asqa_report.html`
   - Charts, tables, mechanism analysis

See [Documentation/OUTPUT_EXAMPLES.md](Documentation/OUTPUT_EXAMPLES.md) for detailed examples.

## 📚 Documentation

Complete documentation in [Documentation/](Documentation/) folder:

### Getting Started
- [SETUP.md](Documentation/SETUP.md) - Complete setup guide with all providers
- [QUICK_SETUP.md](Documentation/QUICK_SETUP.md) - 5-minute quickstart

### Configuration
- [CONFIGURATION.md](Documentation/CONFIGURATION.md) - Complete configuration guide
- [IFCS_CONFIGURATION.md](Documentation/IFCS_CONFIGURATION.md) - IFCS threshold tuning
- [LLM_PROVIDERS.md](Documentation/LLM_PROVIDERS.md) - Multi-LLM provider guide

### Benchmark Evaluation
- [BENCHMARK_WORKFLOW.md](Documentation/BENCHMARK_WORKFLOW.md) - Complete benchmark workflow
- [OUTPUT_EXAMPLES.md](Documentation/OUTPUT_EXAMPLES.md) - Example outputs explained

### System Documentation
- [INDEX.md](Documentation/INDEX.md) - Documentation index
- [README.md](Documentation/README.md) - System overview

## 💰 Cost Estimates

| Provider | Model | 5 Examples | 50 Examples | Full (817) |
|----------|-------|------------|-------------|------------|
| **Anthropic** | Claude Sonnet 4 | ~$0.10 | ~$1.00 | ~$5-10 |
| **Anthropic** | Claude Opus 4 | ~$0.25 | ~$2.50 | ~$15-25 |
| **OpenAI** | GPT-4 Turbo | ~$0.15 | ~$1.50 | ~$8-15 |
| **OpenAI** | GPT-3.5 Turbo | ~$0.02 | ~$0.20 | ~$1-3 |
| **HuggingFace** | Llama 3.1 8B | **FREE** (rate limited) | **FREE** (slow) | ~$3-5 (Pro) |
| **Ollama** | Local Llama 3.1 | **FREE** | **FREE** | **FREE** |

## 🎓 Academic Context

This implementation is based on:

- **Chatterjee, A. (2026a)**. Control Probe: Inference-time commitment control. *Zenodo preprint*. https://doi.org/10.5281/zenodo.18352963

- **Chatterjee, A. (2026b)**. Evaluative Coherence Regulation (ECR): An inference-time stability layer for reliable LLM deployment. *Zenodo preprint*. https://doi.org/10.5281/zenodo.18353477

- **Chatterjee, A. (2026c)**. Inference-Time Commitment Shaping: A Framework for Quiet Failure Mitigation in LLM Systems. *Zenodo preprint*.

**Author**: Arijit Chatterjee (ORCID: 0009-0006-5658-4449)

### Benchmark Datasets

This implementation supports evaluation on:

**TruthfulQA**
- Paper: Lin, S., Hilton, J., & Evans, O. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods.
- Dataset: 817 questions across 38 categories
- Metrics: MC1 (accuracy), MC2 (calibrated probability)
- License: Apache 2.0
- Source: https://huggingface.co/datasets/truthfulqa/truthful_qa

**ASQA**
- Paper: Stelmakh, I., Luan, Y., Dhingra, B., & Chang, M. W. (2022). ASQA: Factoid Questions Meet Long-Form Answers.
- Dataset: 5,300+ ambiguous factoid questions
- Metrics: DR score = √(Disambig-F1 × ROUGE-L)
- License: Apache 2.0
- Source: https://huggingface.co/datasets/din0s/asqa

## 🔬 Key Findings

From the papers:

- **ECR**: Coherence-based selection reduces incoherence-driven failures
- **Control Probe Type-1**: Blocks inadmissible commitments (σ < τ)
- **Control Probe Type-2**: Detects interaction-level drift (R_cum ≥ Θ)
- **IFCS**: Commitment reduction of 50-87% while preserving information
- **Domain calibration**: Stricter thresholds in medical/legal domains prevent dangerous overconfidence
- **Boundary compliance**: 100% - each mechanism operates only within its jurisdiction

## ⚠️ Limitations

1. **Validation Scope**: Illustrative examples, not statistically generalizable
2. **Computational Cost**: K×H LLM calls per query (latency overhead)
3. **Heuristic Scoring**: R(z) components are operational formulas, not learned
4. **Domain Detection**: Keyword-based (can be enhanced with classifiers)
5. **No Underconfidence Handling**: Current IFCS targets overconfidence only

## 🤝 Contributing

This is a research implementation. For questions or collaboration:
- Contact: Arijit Chatterjee
- ORCID: 0009-0006-5658-4449

## 📜 License

Apache 2.0 - See LICENSE file for details.

All benchmark datasets (TruthfulQA, ASQA) are also licensed under Apache 2.0, ensuring full compatibility.

## 🙏 Acknowledgments

- **Papers & Research**: Arijit Chatterjee
- **Implementation**: Claude (Anthropic)
- **Framework**: Based on inference-time governance research

---

**Version**: 1.1.0
**Last Updated**: January 2026
**Status**: Research prototype with benchmark evaluation support

**Repository**: https://github.com/MindtheMachine/IFCS-Implementation
