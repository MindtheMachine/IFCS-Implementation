"""
Statistical analysis and table generation for ECR experiments.
"""

from .statistical_tests import StatisticalAnalyzer
from .latex_tables import LaTeXTableGenerator

__all__ = ['StatisticalAnalyzer', 'LaTeXTableGenerator']
