"""
ECR Empirical Validation Experiments

This package contains experiments for validating the Evaluative Coherence Regulation (ECR)
framework against baselines for journal publication.

Experiments:
- baseline_comparison: ECR vs Self-Consistency vs Vanilla on TruthfulQA
- ablation_study: Test individual metric contributions
- domain_calibration: Domain-specific weight optimization
"""

from . import baselines
from . import runners
from . import analysis
from . import configs
from . import utils

__all__ = ['baselines', 'runners', 'analysis', 'configs', 'utils']
