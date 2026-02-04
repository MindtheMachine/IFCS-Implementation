# Future Improvement Plan: LLM Sensor Architecture  
*(For ECR–Control Probe–IFCS Trilogy)*

---

## Status
**Design Document + Implementation Prompt**

This document is intended to be:
- Placed directly in the repository (e.g. `/docs/llm_sensor_plan.md`)
- Used as a **system / developer prompt** for implementing the mechanism
- Used as a **frontend + backend roadmap**

No part of this document authorizes inference-time self-adaptation.

---

## Design Objective

Introduce **LLM-based perceptual sensors** to enrich semantic signals while preserving:

- Deterministic authority of gates
- Non-generative commitment shaping
- Auditability and replayability
- Dignity-preserving restraint
- Explicit responsibility boundaries

Sensors **observe**.  
Deterministic gates **decide**.  
Learning occurs **only via versioned replacement**.

---

## Core Invariants (Non-Negotiable)

1. Sensors are **read-only** at runtime  
2. Sensors are **stateless**  
3. Sensors **never block, rewrite, or decide**  
4. Deterministic logic always executes fully  
5. Absence of sensor signal ≠ safety  
6. No inference-time learning or calibration  
7. All sensor changes are **versioned and attributable**

> **Learning happens between versions, not between turns.**

---

## High-Level Architecture

```
User Prompt
   ↓
Base LLM (Generator)
   ↓
LLM Sensor(s) [Read-Only, Schema-Only]
   ↓
Deterministic Gates
   ├─ ECR (Selection)
   ├─ Control Probe Type-1 (Admissibility)
   ├─ IFCS (Commitment Shaping)
   └─ Control Probe Type-2 (Interaction Drift)
   ↓
Final Output
```

Sensors may **add signals**, never remove checks.

---

## Sensor Design Per Gate

### 1. ECR Sensor (Selection Assistance)

**Purpose**  
Surface subtle semantic coherence signals between candidates.

**Allowed Outputs**
```json
{
  "internal_consistency": 0.0,
  "semantic_focus": 0.0,
  "temporal_alignment": 0.0,
  "implicit_assumptions": 0.0
}
```

**Forbidden**
- Ranking candidates
- Selecting outputs
- Generating text

**Usage**
- Combined with deterministic ECR metrics
- Selection logic unchanged

---

### 2. Control Probe Type-1 Sensor (Admissibility Signals)

**Purpose**  
Increase recall for illegitimate commitment cues.

**Allowed Outputs**
```json
{
  "fabrication_cues": 0.0,
  "procedural_confidence": 0.0,
  "unsupported_specificity": 0.0,
  "capability_misrepresentation": 0.0
}
```

**Usage**
- May trigger escalation
- Cannot override σ < τ blocking

---

### 3. IFCS Sensor (Commitment Pressure Detection)

**Purpose**  
Detect *pressure*, not perform shaping.

**Allowed Outputs**
```json
{
  "authority_pressure": 0.0,
  "certainty_markers": 0.0,
  "scope_inflation": 0.0,
  "premature_closure": 0.0
}
```

**Usage**
- Feeds ê / ŝ / â estimation only
- IFCS shaping remains deterministic and templated

**Forbidden**
- Rewriting text
- Hedging insertion
- Suggesting refusal

---

### 4. Control Probe Type-2 Sensor (Interaction Drift)

**Purpose**  
Detect longitudinal interaction drift.

**Allowed Outputs**
```json
{
  "sycophancy_trend": 0.0,
  "authority_accumulation": 0.0,
  "confidence_creep": 0.0,
  "normative_alignment_drift": 0.0
}
```

**Usage**
- Deterministic aggregation
- Alerts, warnings, resets only

---

## Runtime Safety Guarantees

- Deterministic gates always run  
- Sensor signals can be ignored or discarded  
- Sensor failure cannot force unsafe behavior  
- System remains safe under sensor disablement  

> **Worst case with sensor = baseline deterministic behavior**

---

## Sensor Versioning Model

### Runtime
```
Sensor_vN (Frozen)
```

### Offline Improvement
```
Logs → Analysis → Candidate Sensor → Regression Tests → Deploy v(N+1)
```

No feedback loop enters runtime.

---

## Automation Pipeline (Safe)

1. **Log Disagreements**
   - Sensor-only flags
   - Deterministic-only triggers
   - Near-threshold cases

2. **Offline Weak Labeling**
   - Rule-based, inspectable
   - No outcome-based reinforcement

3. **Candidate Sensor Generation**
   - Prompt refinement
   - Schema evolution
   - Model swap (rare)

4. **Regression Testing**
   - 36 taxonomy cases
   - Referent ambiguity test (X.1)
   - Calibration slice

5. **Promotion Rule**
   - Recall ↑ or precision ↑
   - Deterministic behavior invariant

6. **Explicit Deployment**
   - Version bump
   - Logged activation

---

## Frontend Changes (Required)

### New UI Sections

#### 1. Sensor Panel
- Active sensor version
- Sensor signals (read-only)
- Clear label: *Advisory*

#### 2. Gate Decision Panel
- Final authority
- Explicit statement:
  > “Deterministic gate decided this outcome.”

#### 3. Disagreement View
- Sensor flagged / gate ignored
- Gate triggered / sensor silent

This makes quiet failures **visible**.

---

## Frontend Anti-Patterns (Forbidden)

- “Sensor approved / rejected”
- Hiding deterministic decisions
- Treating sensor silence as safety

---

## Reset & Fail-Safe Behavior

Allowed:
- Disable sensor
- Ignore sensor outputs
- Fallback to deterministic-only mode

Forbidden:
- Resetting internal sensor state
- Dynamic recalibration during interaction

---

## Compliance Summary

- Deterministic authority preserved
- No silent drift
- No inference-time learning
- Full auditability and replayability
- Clear separation of perception and judgment

---

## Canonical Rule (Lock This)

> **Sensors may be wrong; judges may not.**  
> **Perception may evolve; judgment must remain attributable.**

---

## Intended Use

- Repository documentation
- Implementation prompt
- Architecture review artifact
- Reviewer / auditor explanation
- Future roadmap

---

## End of Document
