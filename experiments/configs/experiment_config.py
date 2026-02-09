"""
Experiment Configuration

Dataclasses and presets for ECR empirical validation experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    # Experiment parameters
    n_questions: int = 100
    k_candidates: int = 5
    seed: int = 42

    # Model configuration (loaded from .env by default)
    model_name: Optional[str] = None

    # ECR parameters
    ecr_h: int = 2  # Inference horizon
    ecr_tau_cci: float = 0.65  # CCI threshold

    # Self-consistency parameters
    sc_similarity_threshold: float = 0.85
    sc_embedding_model: str = "all-MiniLM-L6-v2"

    # Generation parameters
    temperature: float = 0.7

    # Output configuration
    output_dir: str = "Results/experiments"
    save_intermediate: bool = True

    # Checkpoint/resume
    checkpoint_every: int = 10
    resume_from: Optional[str] = None


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str

    # ECR weights (must sum to 1.0)
    alpha: float  # EVB weight (1 - EVB in CCI)
    beta: float   # CR weight (1 - CR in CCI)
    gamma: float  # TS weight
    delta: float  # ES weight
    epsilon: float  # PD weight (1 - PD in CCI)

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Ablation weights must sum to 1.0, got {total:.3f} "
                f"(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
                f"delta={self.delta}, epsilon={self.epsilon})"
            )

    def to_dict(self) -> Dict[str, float]:
        """Return weights as dictionary for ECRConfig."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'epsilon': self.epsilon
        }


@dataclass
class DomainConfig:
    """Configuration for domain-specific calibration."""

    name: str
    description: str

    # Questions for this domain (category filter or custom)
    category_filter: Optional[str] = None
    custom_questions: Optional[List[str]] = None

    # Domain-specific ECR weights
    alpha: float = 0.20
    beta: float = 0.20
    gamma: float = 0.20
    delta: float = 0.20
    epsilon: float = 0.20

    # Number of questions to use
    n_questions: int = 10


# Predefined ablation configurations
ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    # Full model (baseline)
    'full': AblationConfig(
        name='full',
        description='Full ECR with all 5 metrics (equal weights)',
        alpha=0.20, beta=0.20, gamma=0.20, delta=0.20, epsilon=0.20
    ),

    # Remove individual metrics (redistribute weight equally)
    'no_evb': AblationConfig(
        name='no_evb',
        description='ECR without EVB (Evaluative Variance Bound)',
        alpha=0.00, beta=0.25, gamma=0.25, delta=0.25, epsilon=0.25
    ),
    'no_cr': AblationConfig(
        name='no_cr',
        description='ECR without CR (Contradiction Rate)',
        alpha=0.25, beta=0.00, gamma=0.25, delta=0.25, epsilon=0.25
    ),
    'no_ts': AblationConfig(
        name='no_ts',
        description='ECR without TS (Trajectory Smoothness)',
        alpha=0.25, beta=0.25, gamma=0.00, delta=0.25, epsilon=0.25
    ),
    'no_es': AblationConfig(
        name='no_es',
        description='ECR without ES (Expectation Stability)',
        alpha=0.25, beta=0.25, gamma=0.25, delta=0.00, epsilon=0.25
    ),
    'no_pd': AblationConfig(
        name='no_pd',
        description='ECR without PD (Policy Divergence)',
        alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, epsilon=0.00
    ),

    # Single metric only
    'evb_only': AblationConfig(
        name='evb_only',
        description='Only EVB metric',
        alpha=1.00, beta=0.00, gamma=0.00, delta=0.00, epsilon=0.00
    ),
    'cr_only': AblationConfig(
        name='cr_only',
        description='Only CR metric',
        alpha=0.00, beta=1.00, gamma=0.00, delta=0.00, epsilon=0.00
    ),
    'ts_only': AblationConfig(
        name='ts_only',
        description='Only TS metric',
        alpha=0.00, beta=0.00, gamma=1.00, delta=0.00, epsilon=0.00
    ),
    'es_only': AblationConfig(
        name='es_only',
        description='Only ES metric',
        alpha=0.00, beta=0.00, gamma=0.00, delta=1.00, epsilon=0.00
    ),
    'pd_only': AblationConfig(
        name='pd_only',
        description='Only PD metric',
        alpha=0.00, beta=0.00, gamma=0.00, delta=0.00, epsilon=1.00
    ),

    # Core metrics only (EVB + CR + TS)
    'core': AblationConfig(
        name='core',
        description='Core metrics only (EVB, CR, TS)',
        alpha=0.33, beta=0.33, gamma=0.34, delta=0.00, epsilon=0.00
    ),
}


# Predefined domain configurations
DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    'medical': DomainConfig(
        name='medical',
        description='Medical/health domain with emphasis on evidential support',
        category_filter='Health',
        alpha=0.15,  # EVB - lower (prioritize stability)
        beta=0.30,   # CR - higher (avoid contradictions in medical advice)
        gamma=0.10,  # TS - lower
        delta=0.20,  # ES - moderate
        epsilon=0.25,  # PD - moderate-high
        n_questions=10
    ),
    'legal': DomainConfig(
        name='legal',
        description='Legal domain with emphasis on consistency',
        category_filter='Law',
        alpha=0.15,
        beta=0.30,   # CR - high (legal consistency critical)
        gamma=0.10,
        delta=0.20,
        epsilon=0.25,
        n_questions=10
    ),
    'technical': DomainConfig(
        name='technical',
        description='Technical/science domain with emphasis on trajectory smoothness',
        category_filter='Science',
        alpha=0.20,
        beta=0.15,
        gamma=0.25,  # TS - higher (smooth technical explanations)
        delta=0.25,  # ES - higher (consistent expectations)
        epsilon=0.15,
        n_questions=10
    ),
    'default': DomainConfig(
        name='default',
        description='Default configuration (equal weights)',
        category_filter=None,  # All categories
        alpha=0.20,
        beta=0.20,
        gamma=0.20,
        delta=0.20,
        epsilon=0.20,
        n_questions=10
    ),
}
