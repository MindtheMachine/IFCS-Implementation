# IFCS Configuration Guide

Quick reference for configuring IFCS (Inference-Time Commitment Shaping) thresholds and weights.

## 🎯 Quick Answer: Where to Configure IFCS

Edit [trilogy_config.py](../trilogy_config.py) lines 56-96:

```python
@dataclass
class IFCSConfig:
    # THRESHOLD: When to apply commitment shaping
    rho: float = 0.40  # ρ in the paper

    # WEIGHTS: How to combine risk components (must sum to 1.0)
    lambda_e: float = 0.40  # λ_e: Evidential risk weight
    lambda_s: float = 0.30  # λ_s: Scope inflation weight
    lambda_a: float = 0.30  # λ_a: Authority cues weight
    lambda_t: float = 0.00  # λ_t: Temporal risk weight
```

---

## 📊 Understanding the Components

### What are ê, ŝ, â? (Risk Scores)

These are **computed dynamically** from the LLM's response - **not configuration values**:

- **ê (e-hat)**: Evidential risk score (0.0-1.0)
  - Measures lack of grounding/evidence
  - Computed by analyzing:
    - Absence of hedging ("definitely", "always", "never")
    - Lack of qualification ("generally", "typically", "often")
    - Missing caveats

- **ŝ (s-hat)**: Scope inflation score (0.0-1.0)
  - Measures overgeneralization
  - Computed by analyzing:
    - Universal quantifiers ("all", "every", "always")
    - Broad scope claims without qualification

- **â (a-hat)**: Authority gradient score (0.0-1.0)
  - Measures directive/prescriptive language
  - Computed by analyzing:
    - Authority markers ("must", "should", "need to")
    - Prescriptive language ("you should", "the only way")

### What are λ_e, λ_s, λ_a? (Weights)

These **combine the risk scores** into total commitment risk:

**Formula:**
```
R(z) = λ_e·ê + λ_s·ŝ + λ_a·â + λ_t·t̂
```

**Defaults:**
- λ_e = 0.40 (40% weight on evidential risk)
- λ_s = 0.30 (30% weight on scope inflation)
- λ_a = 0.30 (30% weight on authority cues)
- λ_t = 0.00 (temporal risk disabled by default)

**Constraint:** λ_e + λ_s + λ_a + λ_t = 1.0

### What is ρ (rho)? (Threshold)

**When R(z) > ρ, IFCS fires** and applies commitment shaping transformations.

**Default:** ρ = 0.40

**Effect:**
- **Lower ρ** (e.g., 0.30): IFCS fires more often (stricter)
- **Higher ρ** (e.g., 0.50): IFCS fires less often (more permissive)

---

## ⚙️ Common Configuration Scenarios

### Scenario 1: Make IFCS Fire More Often

**Problem:** IFCS isn't catching overconfident responses

**Solution:** Lower the threshold
```python
rho: float = 0.30  # Was 0.40, now fires more often
```

### Scenario 2: Emphasize Evidential Grounding

**Problem:** Model makes ungrounded claims

**Solution:** Increase λ_e, reduce others
```python
lambda_e: float = 0.50  # Was 0.40, now emphasizes evidence
lambda_s: float = 0.25  # Was 0.30, reduced
lambda_a: float = 0.25  # Was 0.30, reduced
```

### Scenario 3: Reduce False Positives

**Problem:** IFCS fires too often on reasonable responses

**Solution:** Raise the threshold
```python
rho: float = 0.50  # Was 0.40, now fires less often
```

### Scenario 4: Domain-Specific Strictness

**Problem:** Need stricter thresholds for medical/legal domains

**Solution:** Use domain-specific configs (lines 82-89)
```python
domain_configs = {
    'medical': DomainConfig(
        rho=0.30,       # Strict threshold
        lambda_e=0.50,  # Emphasize evidence
        lambda_s=0.20,
        lambda_a=0.20,
        lambda_t=0.10
    )
}
```

---

## 🔍 How to Test Configuration Changes

### 1. Test on Single Query
```bash
python trilogy_app.py --prompt "What is the best treatment for chest pain?"
```

Check `regulated_output.txt` for:
- Did IFCS fire? (`ifcs_fired: True/False`)
- What was R(z)? (commitment risk score)
- What transformations were applied?

### 2. Run on Test Suite
```bash
python trilogy_app.py --test-suite
```

Checks firing behavior on 36 taxonomy test cases.

### 3. Run on Benchmark Subset
```bash
python trilogy_app.py --benchmark truthfulqa --batch-size 20
```

Analyze firing rates and impact:
```python
import pandas as pd
results = pd.read_csv("Results/{model}/truthfulqa_results.csv")

# IFCS firing rate
firing_rate = results['ifcs_fired'].mean()
print(f"IFCS firing rate: {firing_rate:.1%}")

# Performance when IFCS fires
ifcs_fired = results[results['ifcs_fired'] == True]
print(f"MC1 when IFCS fires: {ifcs_fired['regulated_mc1_accuracy'].mean():.3f}")
```

---

## 📈 Benchmark-Based Tuning

### Finding Optimal Thresholds

**Approach 1: Grid Search**
```python
# Edit trilogy_config.py to try different values
rho_values = [0.30, 0.35, 0.40, 0.45, 0.50]

for rho in rho_values:
    # Update config
    config = IFCSConfig(rho=rho)

    # Run benchmark
    python trilogy_app.py --benchmark truthfulqa --batch-size 50

    # Compare MC1/MC2 scores
```

**Approach 2: Analyze Existing Results**
```python
import pandas as pd
import json

# Load results
results = pd.read_csv("Results/{model}/truthfulqa_results.csv")
summary = json.load(open("Results/{model}/truthfulqa_summary.json"))

# Check if IFCS improves scores
ifcs_fired = results[results['ifcs_fired'] == True]
ifcs_not_fired = results[results['ifcs_fired'] == False]

print(f"MC1 when IFCS fires: {ifcs_fired['regulated_mc1_accuracy'].mean():.3f}")
print(f"MC1 when IFCS doesn't fire: {ifcs_not_fired['regulated_mc1_accuracy'].mean():.3f}")

# If IFCS helps when it fires, consider lowering ρ to fire more often
# If IFCS hurts, consider raising ρ to fire less often
```

---

## 🎓 Best Practices

1. **Start with defaults** (ρ=0.40) for general use
2. **Use domain configs** for medical/legal/financial queries
3. **Test on small batches** before running full benchmarks
4. **Monitor firing rates**: 40-60% is typical
5. **Validate improvements**: IFCS should improve MC1/MC2 on TruthfulQA
6. **Document changes**: Keep notes on what works for your use case

---

## 📊 Expected Behavior

### Default Configuration (ρ=0.40)

**TruthfulQA:**
- IFCS firing rate: ~40-60%
- MC1 improvement: +5-15 percentage points
- MC2 improvement: +3-10 percentage points

**Typical Firing Triggers:**
- Questions about medical advice
- "What is the best..." questions
- Ambiguous technical questions
- Questions with temporal assumptions

**Typical Non-Firing:**
- Factual questions with clear answers
- "What is the capital of..." questions
- Well-defined technical terms

---

## 🔗 Related Configuration

### ECR Configuration
```python
@dataclass
class ECRConfig:
    tau_CCI: float = 0.65  # Coherence threshold
    K: int = 5  # Number of candidates
```

### Control Probe Configuration
```python
@dataclass
class ControlProbeConfig:
    tau: float = 0.40     # CP-Type-1 threshold
    Theta: float = 2.0    # CP-Type-2 cumulative risk threshold
```

---

## 📖 See Also

- **Paper Reference**: Section 3.3 (IFCS Mechanism Design)
- [BENCHMARK_WORKFLOW.md](BENCHMARK_WORKFLOW.md) - How IFCS works in benchmarks
- [OUTPUT_EXAMPLES.md](OUTPUT_EXAMPLES.md) - Examples of IFCS firing
- [trilogy_config.py](../trilogy_config.py) - Full configuration file
