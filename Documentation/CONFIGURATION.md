# Trilogy System Configuration Guide

Complete guide to configuring ECR, Control Probe, and IFCS parameters for deployment.

## 🎯 Quick Start

### Option 1: Domain Presets (Easiest)

Edit `.env` and set your domain:

```bash
# Medical domain (strictest, ρ=0.30)
IFCS_DOMAIN=medical

# Legal domain (strict, ρ=0.30)
IFCS_DOMAIN=legal

# Financial domain (moderate, ρ=0.35)
IFCS_DOMAIN=financial

# Default domain (standard, ρ=0.40)
IFCS_DOMAIN=default
```

**Domain presets from IFCS paper Table 1:**

| Domain | ρ (threshold) | λ_e (evidential) | λ_s (scope) | λ_a (authority) | λ_t (temporal) |
|--------|---------------|------------------|-------------|-----------------|----------------|
| Medical | 0.30 | 0.50 | 0.20 | 0.20 | 0.10 |
| Legal | 0.30 | 0.50 | 0.20 | 0.20 | 0.10 |
| Financial | 0.35 | 0.45 | 0.25 | 0.20 | 0.10 |
| Default | 0.40 | 0.40 | 0.30 | 0.30 | 0.00 |

### Option 2: Custom Thresholds

Edit `.env` for fine-grained control:

```bash
# IFCS Configuration
IFCS_RHO=0.35           # Lower threshold = more interventions
IFCS_LAMBDA_E=0.45      # Emphasize evidential grounding
IFCS_LAMBDA_S=0.25      # Scope inflation weight
IFCS_LAMBDA_A=0.30      # Authority cues weight
IFCS_LAMBDA_T=0.00      # Temporal risk (usually 0)

# ECR Configuration
ECR_K=5                 # Number of candidates
ECR_TAU_CCI=0.70        # Higher = stricter coherence

# Control Probe Configuration
CP_TAU=0.35             # Type-1 admissibility threshold
CP_THETA=2.0            # Type-2 cumulative risk threshold
```

**Note:** All λ weights must sum to 1.0

### Option 3: JSON Configuration File

For complex configurations, create `trilogy_config.json`:

```bash
# Copy template
cp trilogy_config.json.template trilogy_config.json

# Edit trilogy_config.json with your preferred settings
```

---

## 📚 Configuration Priority

The system loads configuration in this order (highest priority first):

1. **Environment variables** (`.env`) - Highest priority
2. **JSON config file** (`trilogy_config.json`)
3. **YAML config file** (`trilogy_config.yaml`)
4. **Python defaults** (`trilogy_config.py`) - Lowest priority

**Example:** If `IFCS_RHO` is set in `.env`, it overrides values from JSON/YAML/Python.

---

## 🔧 Configuration Parameters

### IFCS (Inference-Time Commitment Shaping)

#### What is IFCS?

IFCS detects and reduces overconfident language in LLM responses.

**Formula:** R(z) = λ_e·ê + λ_s·ŝ + λ_a·â + λ_t·t̂

**When R(z) > ρ, IFCS fires** and applies commitment shaping transformations.

#### Parameters

**`IFCS_RHO`** (ρ): Commitment risk threshold (0.0-1.0)
- **Default:** 0.40 (standard), 0.30 (medical/legal), 0.35 (financial)
- **Lower values** (e.g., 0.30): IFCS fires more often → more interventions → safer but more cautious
- **Higher values** (e.g., 0.50): IFCS fires less often → fewer interventions → more direct but riskier
-- **Paper guidance:** Medical/legal domains require ρ=0.30 due to severe consequences of mistakes

**`IFCS_LAMBDA_E`** (λ_e): Evidential risk weight (0.0-1.0)
- **Default:** 0.40 (standard), 0.50 (medical/legal)
- Weight for evidential sufficiency component (ê)
- Measures lack of grounding/evidence in responses
- **Higher values** emphasize factual grounding
-- **When to increase:** Domains requiring strong evidence (medical, scientific)

**`IFCS_LAMBDA_S`** (λ_s): Scope inflation weight (0.0-1.0)
- **Default:** 0.30 (standard), 0.20 (medical/legal)
- Weight for scope inflation component (ŝ)
- Measures overgeneralization ("always", "all", "never")
- **Higher values** penalize broad generalizations
- **When to increase:** Contexts where overgeneralization is risky

**`IFCS_LAMBDA_A`** (λ_a): Authority cues weight (0.0-1.0)
- **Default:** 0.30 (standard), 0.20 (medical/legal)
- Weight for authority/directive language (â)
- Measures prescriptive language ("must", "should", "you need to")
- **Higher values** penalize directive language
- **When to increase:** Advisory contexts where prescriptions are inappropriate

**`IFCS_LAMBDA_T`** (λ_t): Temporal risk weight (0.0-1.0)
- **Default:** 0.00 (usually disabled), 0.10 (medical/legal/financial)
- Weight for temporal grounding issues
- Measures claims about time-sensitive information
-- **When to enable:** Time-sensitive domains (news, market data, medical updates)

**Constraint:** λ_e + λ_s + λ_a + λ_t = 1.0

**`IFCS_DOMAIN`**: Auto-apply domain preset (medical|legal|financial|default)
- **Convenience parameter** that sets all IFCS values at once
- Overrides individual parameters unless explicitly set
- Based on IFCS paper Table 1

---

### ECR (Evaluative Coherence Regulation)

#### What is ECR?

ECR generates multiple candidate responses and selects the most coherent one.

**Formula:** CCI = α·(1-EVB) + β·(1-CR) + γ·TS + δ·ES + ε·(1-PD)

#### Parameters

**`ECR_K`**: Number of candidate responses (integer, 3-5 recommended)
- **Default:** 5
- **Paper guidance:** "Choose K ∈ [3, 5]"
- **Higher values** (e.g., 7): More diversity, better selection, higher cost
- **Lower values** (e.g., 3): Faster, cheaper, less diversity
- **When to increase:** Critical applications where quality matters most
- **When to decrease:** Cost-sensitive or latency-sensitive applications

**`ECR_H`**: Inference horizon (integer, 2-4 recommended)
- **Default:** 3
- Number of forward steps for trajectory evaluation
- **Paper guidance:** "Choose H ∈ [2, 4]"
- **Higher values:** Better long-term coherence prediction, slower
- **Lower values:** Faster, less accurate coherence prediction

**`ECR_TAU_CCI`**: Coherence threshold (0.0-1.0)
- **Default:** 0.65
- Minimum Composite Coherence Index for admissibility
- **Paper guidance:** "Tune to admit 70-80% of trajectories"
- **Higher values** (e.g., 0.75): Stricter coherence requirements
- **Lower values** (e.g., 0.55): More permissive

---

### Control Probe

#### What are Control Probes?

**Type-1:** Inference-local admissibility gating (single-turn)
**Type-2:** Interaction monitoring across conversation (multi-turn)

#### Parameters

**`CP_TAU`** (τ): Type-1 admissibility threshold (0.0-1.0)
- **Default:** 0.40
- Minimum evaluative support for commitment
- **Lower values**: More interventions (stricter)
- **Higher values**: Fewer interventions (more permissive)
- **When to lower:** High-risk contexts requiring strong support

**`CP_THETA`** (Θ): Type-2 cumulative risk threshold (0.5-3.0)
- **Default:** 2.0
- Cumulative commitment risk across conversation turns
- **When R_cumulative > Θ:** System halts or resets
- **Lower values** (e.g., 1.5): Detect drift earlier, more conservative
- **Higher values** (e.g., 2.5): Allow more conversational flexibility
- **When to lower:** Contexts where consistency is critical (medical diagnosis)

---

## 🎓 Configuration Examples

### Example 1: Medical Chatbot (Strictest)

**Goal:** Maximum safety, prevent dangerous overconfidence

```bash
# .env
IFCS_DOMAIN=medical
# OR manually:
# IFCS_RHO=0.30
# IFCS_LAMBDA_E=0.50
# IFCS_LAMBDA_S=0.20
# IFCS_LAMBDA_A=0.20
# IFCS_LAMBDA_T=0.10

ECR_K=5
ECR_TAU_CCI=0.70
CP_TAU=0.35
CP_THETA=1.8
```

**Effect:**
- IFCS fires ~60-70% of the time
- Strong emphasis on evidential grounding
- Low tolerance for cumulative risk

### Example 2: General Q&A (Balanced)

**Goal:** Balance safety and naturalness

```bash
# .env
IFCS_DOMAIN=default
# IFCS_RHO=0.40
# IFCS_LAMBDA_E=0.40
# IFCS_LAMBDA_S=0.30
# IFCS_LAMBDA_A=0.30
# IFCS_LAMBDA_T=0.00

ECR_K=5
ECR_TAU_CCI=0.65
CP_TAU=0.40
CP_THETA=2.0
```

**Effect:**
- IFCS fires ~40-50% of the time
- Balanced risk component weights
- Standard thresholds

### Example 3: Creative Writing Assistant (Permissive)

**Goal:** Allow creative freedom, minimal intervention

```bash
# .env
IFCS_RHO=0.55
IFCS_LAMBDA_E=0.30
IFCS_LAMBDA_S=0.35
IFCS_LAMBDA_A=0.35
IFCS_LAMBDA_T=0.00

ECR_K=3
ECR_TAU_CCI=0.60
CP_TAU=0.45
CP_THETA=2.5
```

**Effect:**
- IFCS fires ~20-30% of the time
- More permissive thresholds
- Lower emphasis on evidential grounding

---

## 🧪 Testing Your Configuration

### 1. Single Query Test

```bash
python trilogy_app.py --prompt "What is the best treatment for chest pain?"
```

Check `regulated_output.txt`:
- Did IFCS fire? (`ifcs_fired: True/False`)
- What was R(z)? (commitment risk score)
- Did transformations improve safety?

### 2. Test Suite

```bash
python trilogy_app.py --test-suite
```

Analyze firing rates across 36 taxonomy test cases.

### 3. Benchmark Evaluation

```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 50
```

Analyze results:
```python
import pandas as pd
results = pd.read_csv("Results/{model}/truthfulqa_results.csv")

# IFCS firing rate
print(f"IFCS fires: {results['ifcs_fired'].mean():.1%}")

# Impact on accuracy
ifcs_on = results[results['ifcs_fired'] == True]
ifcs_off = results[results['ifcs_fired'] == False]
print(f"MC1 when IFCS fires: {ifcs_on['regulated_mc1_accuracy'].mean():.3f}")
print(f"MC1 when IFCS off: {ifcs_off['regulated_mc1_accuracy'].mean():.3f}")
```

### 4. Grid Search for Optimal ρ

```python
import subprocess
import json

rho_values = [0.30, 0.35, 0.40, 0.45, 0.50]
results = {}

for rho in rho_values:
    # Set environment variable
    os.environ['IFCS_RHO'] = str(rho)

    # Run benchmark
    subprocess.run([
        "python", "trilogy_app.py",
        "--benchmark", "truthfulqa",
        "--batch-size", "50"
    ])

    # Load results
    summary = json.load(open("Results/{model}/truthfulqa_summary.json"))
    results[rho] = summary['overall_metrics']['regulated']['mc1_accuracy']

# Find optimal
optimal_rho = max(results, key=results.get)
print(f"Optimal ρ: {optimal_rho} (MC1: {results[optimal_rho]:.3f})")
```

---

## 📖 Configuration Recipes

### Recipe 1: "Strict Medical Mode"

```bash
IFCS_RHO=0.28
IFCS_LAMBDA_E=0.55
IFCS_LAMBDA_S=0.20
IFCS_LAMBDA_A=0.15
IFCS_LAMBDA_T=0.10
ECR_TAU_CCI=0.72
CP_TAU=0.32
CP_THETA=1.5
```

Use when: Medical diagnosis, treatment recommendations, health advice

### Recipe 2: "Research Assistant Mode"

```bash
IFCS_RHO=0.38
IFCS_LAMBDA_E=0.50
IFCS_LAMBDA_S=0.25
IFCS_LAMBDA_A=0.25
IFCS_LAMBDA_T=0.00
ECR_TAU_CCI=0.68
CP_TAU=0.38
CP_THETA=2.0
```

Use when: Literature review, citation-heavy content, academic writing

### Recipe 3: "Customer Support Mode"

```bash
IFCS_RHO=0.42
IFCS_LAMBDA_E=0.35
IFCS_LAMBDA_S=0.30
IFCS_LAMBDA_A=0.35
IFCS_LAMBDA_T=0.00
ECR_TAU_CCI=0.65
CP_TAU=0.40
CP_THETA=2.2
```

Use when: General customer service, FAQs, product information

---

## 🔗 Related Documentation

- [IFCS_CONFIGURATION.md](IFCS_CONFIGURATION.md) - Detailed IFCS parameter guide
- [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - How to measure configuration impact
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - See configuration effects in outputs

---

## 📄 Paper References

- **IFCS paper Table 1:** Domain-specific threshold configurations
- **IFCS paper Section 4.3.1:** Domain sensitivity and calibration
- **IFCS paper Appendix D.4:** Deployment-time configuration guidance
- **ECR paper Section 4.3:** CCI threshold tuning
- **Control Probe paper Section 3.2:** Admissibility thresholds
