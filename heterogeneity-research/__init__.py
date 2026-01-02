"""
Heterogeneity research tools for spatial voting simulations.

Provides:
- Research framework and questions
- Systematic test suite
- Visualization tools
- Analysis and interpretation tools
"""

from .visualization_tools import HeterogeneityVisualizer
from .systematic_tests import HeterogeneityTestSuite
from .analysis_tools import HeterogeneityAnalyzer

__all__ = [
    'HeterogeneityVisualizer',
    'HeterogeneityTestSuite',
    'HeterogeneityAnalyzer',
]





