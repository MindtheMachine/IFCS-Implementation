"""
Baseline methods for ECR comparison experiments.
"""

from .self_consistency import SelfConsistencyBaseline
from .vanilla import VanillaBaseline

__all__ = ['SelfConsistencyBaseline', 'VanillaBaseline']
