"""
Configuration Loader for Trilogy System
Supports .env, JSON, and YAML configuration with fallback to trilogy_config.py defaults
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from pathlib import Path


@dataclass
class IFCSRuntimeConfig:
    """Runtime-configurable IFCS parameters (can be set via .env or config files)"""

    # Commitment risk threshold
    rho: float = 0.40

    # Risk component weights (must sum to 1.0)
    lambda_e: float = 0.40  # Evidential sufficiency
    lambda_s: float = 0.30  # Scope inflation
    lambda_a: float = 0.30  # Authority cues
    lambda_t: float = 0.00  # Temporal risk (optional)

    # Domain-specific settings (if domain is detected)
    domain: str = "default"  # Options: default, medical, legal, financial

    def validate(self):
        """Validate configuration"""
        weight_sum = self.lambda_e + self.lambda_s + self.lambda_a + self.lambda_t
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"IFCS weights must sum to 1.0, got {weight_sum}")

        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [0.0, 1.0], got {self.rho}")


@dataclass
class ECRRuntimeConfig:
    """Runtime-configurable ECR parameters"""

    # Number of candidate responses to generate
    K: int = 5

    # Inference horizon
    H: int = 3

    # Composite Coherence Index threshold
    tau_CCI: float = 0.65

    # Weights for CCI computation (must sum to 1.0)
    alpha: float = 0.20   # EVB weight
    beta: float = 0.20    # CR weight
    gamma: float = 0.20   # TS weight
    delta: float = 0.20   # ES weight
    epsilon: float = 0.20  # PD weight

    def validate(self):
        """Validate configuration"""
        weight_sum = self.alpha + self.beta + self.gamma + self.delta + self.epsilon
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"ECR weights must sum to 1.0, got {weight_sum}")


@dataclass
class ControlProbeRuntimeConfig:
    """Runtime-configurable Control Probe parameters"""

    # Type-1: Inference-local admissibility threshold
    tau: float = 0.40

    # Type-2: Cumulative risk threshold
    Theta: float = 2.0

    # Minimum support for commitment
    min_support: float = 0.40


class TrilogyConfigLoader:
    """Loads trilogy configuration from multiple sources with priority:

    Priority (highest to lowest):
    1. Environment variables (.env)
    2. JSON config file (trilogy_config.json)
    3. YAML config file (trilogy_config.yaml)
    4. Python defaults (trilogy_config.py)

    Environment Variable Format:
        IFCS_RHO=0.30
        IFCS_LAMBDA_E=0.50
        IFCS_DOMAIN=medical
        ECR_TAU_CCI=0.70
        CP_TAU=0.35
        CP_THETA=1.8
    """

    # Domain-specific presets (from IFCS paper Table 1)
    DOMAIN_PRESETS = {
        'medical': {
            'rho': 0.30,
            'lambda_e': 0.50,
            'lambda_s': 0.20,
            'lambda_a': 0.20,
            'lambda_t': 0.10
        },
        'legal': {
            'rho': 0.30,
            'lambda_e': 0.50,
            'lambda_s': 0.20,
            'lambda_a': 0.20,
            'lambda_t': 0.10
        },
        'financial': {
            'rho': 0.35,
            'lambda_e': 0.45,
            'lambda_s': 0.25,
            'lambda_a': 0.20,
            'lambda_t': 0.10
        },
        'default': {
            'rho': 0.40,
            'lambda_e': 0.40,
            'lambda_s': 0.30,
            'lambda_a': 0.30,
            'lambda_t': 0.00
        }
    }

    @staticmethod
    def load_from_env() -> Dict:
        """Load configuration from environment variables"""
        config = {}

        # IFCS parameters
        if os.getenv('IFCS_RHO'):
            config['ifcs_rho'] = float(os.getenv('IFCS_RHO'))
        if os.getenv('IFCS_LAMBDA_E'):
            config['ifcs_lambda_e'] = float(os.getenv('IFCS_LAMBDA_E'))
        if os.getenv('IFCS_LAMBDA_S'):
            config['ifcs_lambda_s'] = float(os.getenv('IFCS_LAMBDA_S'))
        if os.getenv('IFCS_LAMBDA_A'):
            config['ifcs_lambda_a'] = float(os.getenv('IFCS_LAMBDA_A'))
        if os.getenv('IFCS_LAMBDA_T'):
            config['ifcs_lambda_t'] = float(os.getenv('IFCS_LAMBDA_T'))
        if os.getenv('IFCS_DOMAIN'):
            config['ifcs_domain'] = os.getenv('IFCS_DOMAIN').lower()

        # ECR parameters
        if os.getenv('ECR_K'):
            config['ecr_K'] = int(os.getenv('ECR_K'))
        if os.getenv('ECR_H'):
            config['ecr_H'] = int(os.getenv('ECR_H'))
        if os.getenv('ECR_TAU_CCI'):
            config['ecr_tau_CCI'] = float(os.getenv('ECR_TAU_CCI'))

        # Control Probe parameters
        if os.getenv('CP_TAU'):
            config['cp_tau'] = float(os.getenv('CP_TAU'))
        if os.getenv('CP_THETA'):
            config['cp_Theta'] = float(os.getenv('CP_THETA'))

        return config

    @staticmethod
    def load_from_json(filepath: str = "trilogy_config.json") -> Optional[Dict]:
        """Load configuration from JSON file"""
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_from_yaml(filepath: str = "trilogy_config.yaml") -> Optional[Dict]:
        """Load configuration from YAML file"""
        if not os.path.exists(filepath):
            return None

        try:
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not installed. Install with: pip install pyyaml")
            return None

    @classmethod
    def load_ifcs_config(cls) -> IFCSRuntimeConfig:
        """Load IFCS configuration with priority: env > json > yaml > defaults"""

        # Start with defaults
        config = IFCSRuntimeConfig()

        # Try YAML
        yaml_config = cls.load_from_yaml()
        if yaml_config and 'ifcs' in yaml_config:
            for key, value in yaml_config['ifcs'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try JSON (overrides YAML)
        json_config = cls.load_from_json()
        if json_config and 'ifcs' in json_config:
            for key, value in json_config['ifcs'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try environment variables (highest priority)
        env_config = cls.load_from_env()
        if 'ifcs_rho' in env_config:
            config.rho = env_config['ifcs_rho']
        if 'ifcs_lambda_e' in env_config:
            config.lambda_e = env_config['ifcs_lambda_e']
        if 'ifcs_lambda_s' in env_config:
            config.lambda_s = env_config['ifcs_lambda_s']
        if 'ifcs_lambda_a' in env_config:
            config.lambda_a = env_config['ifcs_lambda_a']
        if 'ifcs_lambda_t' in env_config:
            config.lambda_t = env_config['ifcs_lambda_t']
        if 'ifcs_domain' in env_config:
            config.domain = env_config['ifcs_domain']

        # Apply domain preset if domain is specified
        if config.domain in cls.DOMAIN_PRESETS:
            preset = cls.DOMAIN_PRESETS[config.domain]
            # Only apply preset if individual values weren't explicitly set
            if 'ifcs_rho' not in env_config and (not json_config or 'rho' not in json_config.get('ifcs', {})):
                config.rho = preset['rho']
            if 'ifcs_lambda_e' not in env_config:
                config.lambda_e = preset['lambda_e']
            if 'ifcs_lambda_s' not in env_config:
                config.lambda_s = preset['lambda_s']
            if 'ifcs_lambda_a' not in env_config:
                config.lambda_a = preset['lambda_a']
            if 'ifcs_lambda_t' not in env_config:
                config.lambda_t = preset['lambda_t']
        # Validate
        config.validate()

        return config

    @classmethod
    def load_ecr_config(cls) -> ECRRuntimeConfig:
        """Load ECR configuration with priority: env > json > yaml > defaults"""

        config = ECRRuntimeConfig()

        # Try YAML
        yaml_config = cls.load_from_yaml()
        if yaml_config and 'ecr' in yaml_config:
            for key, value in yaml_config['ecr'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try JSON (overrides YAML)
        json_config = cls.load_from_json()
        if json_config and 'ecr' in json_config:
            for key, value in json_config['ecr'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try environment variables (highest priority)
        env_config = cls.load_from_env()
        if 'ecr_K' in env_config:
            config.K = env_config['ecr_K']
        if 'ecr_H' in env_config:
            config.H = env_config['ecr_H']
        if 'ecr_tau_CCI' in env_config:
            config.tau_CCI = env_config['ecr_tau_CCI']

        # Validate
        config.validate()

        return config

    @classmethod
    def load_control_probe_config(cls) -> ControlProbeRuntimeConfig:
        """Load Control Probe configuration with priority: env > json > yaml > defaults"""

        config = ControlProbeRuntimeConfig()

        # Try YAML
        yaml_config = cls.load_from_yaml()
        if yaml_config and 'control_probe' in yaml_config:
            for key, value in yaml_config['control_probe'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try JSON (overrides YAML)
        json_config = cls.load_from_json()
        if json_config and 'control_probe' in json_config:
            for key, value in json_config['control_probe'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Try environment variables (highest priority)
        env_config = cls.load_from_env()
        if 'cp_tau' in env_config:
            config.tau = env_config['cp_tau']
        if 'cp_Theta' in env_config:
            config.Theta = env_config['cp_Theta']

        return config

    @classmethod
    def create_template_json(cls, filepath: str = "trilogy_config.json.template"):
        """Create a template JSON config file with documentation"""
        template = {
            "_comment": "Trilogy System Configuration Template",
            "_docs": {
                "priority": "env > json > yaml > defaults",
                "domains": "medical, legal, financial, default",
                "paper_reference": "IFCS paper Table 1, Section 4.3.1"
            },
            "ifcs": {
                "_comment": "Inference-Time Commitment Shaping (IFCS)",
                "domain": "default",
                "rho": 0.40,
                "lambda_e": 0.40,
                "lambda_s": 0.30,
                "lambda_a": 0.30,
                "lambda_t": 0.00,
                "_domains": {
                    "medical": {"rho": 0.30, "lambda_e": 0.50, "lambda_s": 0.20, "lambda_a": 0.20, "lambda_t": 0.10},
                    "legal": {"rho": 0.30, "lambda_e": 0.50, "lambda_s": 0.20, "lambda_a": 0.20, "lambda_t": 0.10},
                    "financial": {"rho": 0.35, "lambda_e": 0.45, "lambda_s": 0.25, "lambda_a": 0.20, "lambda_t": 0.10}
                }
            },
            "ecr": {
                "_comment": "Evaluative Coherence Regulation (ECR)",
                "K": 5,
                "H": 3,
                "tau_CCI": 0.65,
                "alpha": 0.20,
                "beta": 0.20,
                "gamma": 0.20,
                "delta": 0.20,
                "epsilon": 0.20
            },
            "control_probe": {
                "_comment": "Control Probe (Type-1 and Type-2)",
                "tau": 0.40,
                "Theta": 2.0,
                "min_support": 0.40
            }
        }

        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"Created template config file: {filepath}")
        print("Copy to trilogy_config.json and customize as needed")


# Convenience function for external use
def load_trilogy_config() -> Dict:
    """Load complete trilogy configuration from all sources"""
    return {
        'ifcs': TrilogyConfigLoader.load_ifcs_config(),
        'ecr': TrilogyConfigLoader.load_ecr_config(),
        'control_probe': TrilogyConfigLoader.load_control_probe_config()
    }
