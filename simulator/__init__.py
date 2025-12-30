"""
Spatial Voting Simulator

A modular voting simulation system with spatial preferences,
manipulation support, and comprehensive metrics analysis.
"""

__version__ = "1.0.0"
__author__ = "Social Choice Simulator"

from .config import (
    SimulationConfig, GeometryConfig, ManipulationConfig,
    HeterogeneousDistanceConfig, UtilityConfig
)
from .main import generate_preferences, run_simulation, run_experiment
from .comparison import (
    ExperimentComparator, ExperimentSummary, ComparisonResult,
    compare_experiments, get_comparison_modes
)

__all__ = [
    'SimulationConfig',
    'GeometryConfig', 
    'ManipulationConfig',
    'HeterogeneousDistanceConfig',
    'UtilityConfig',
    'generate_preferences',
    'run_simulation',
    'run_experiment',
    # Comparison
    'ExperimentComparator',
    'ExperimentSummary',
    'ComparisonResult',
    'compare_experiments',
    'get_comparison_modes',
]

# GUI entry point
def run_gui():
    """Launch the Textual GUI application."""
    from .gui import main
    main()

