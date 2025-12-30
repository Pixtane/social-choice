"""
Configuration classes for the voting simulator.

Contains dataclasses for simulation parameters, geometry settings,
and manipulation configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import uuid
from datetime import datetime


@dataclass
class GeometryConfig:
    """Configuration for spatial geometry generation."""
    
    # Generation method
    method: Literal[
        'uniform', 'clustered', 'single_peaked', 
        'polarized', '1d', '2d', 'custom'
    ] = 'uniform'
    
    # Spatial dimensions
    n_dim: int = 2
    
    # Clustering parameters
    cluster_variance: float = 0.15  # For clustered generation
    phi: float = 0.5  # Dispersion parameter (0=tight, 1=loose)
    
    # Candidate placement
    candidate_method: Literal[
        'random', 'fixed_triangle', 'fixed_line', 'clustered'
    ] = 'random'
    
    # Bounds for positions
    position_min: float = 0.0
    position_max: float = 1.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_dim < 1:
            raise ValueError("n_dim must be >= 1")
        if self.phi < 0 or self.phi > 1:
            raise ValueError("phi must be in [0, 1]")
        if self.position_min >= self.position_max:
            raise ValueError("position_min must be < position_max")


@dataclass
class ManipulationConfig:
    """Configuration for strategic voting manipulation."""
    
    # Whether manipulation is enabled
    enabled: bool = False
    
    # Fraction of voters who manipulate (0.0 to 1.0)
    manipulator_fraction: float = 0.2
    
    # How manipulators are selected
    selection_method: Literal[
        'random', 'extremists', 'centrists', 'informed'
    ] = 'random'
    
    # Manipulation strategy
    strategy: Literal[
        'bullet', 'compromise', 'burial', 'pushover', 'optimal'
    ] = 'compromise'
    
    # Information available to manipulators
    information_level: Literal[
        'none', 'polls', 'full'  # none=blind, polls=approximate, full=exact
    ] = 'polls'
    
    # For poll-based information: noise in polls
    poll_noise: float = 0.1
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.manipulator_fraction <= 1:
            raise ValueError("manipulator_fraction must be in [0, 1]")
        if self.poll_noise < 0:
            raise ValueError("poll_noise must be >= 0")


@dataclass
class VotingRuleConfig:
    """Configuration for voting rule parameters."""
    
    # Approval voting parameters
    approval_policy: Literal['top_k', 'threshold', 'mean', 'above_average'] = 'top_k'
    approval_k: int = 2  # For top_k: approve top k candidates
    approval_threshold: float = 0.5  # For threshold: approve if utility > threshold
    
    # Score voting parameters
    score_max: int = 5  # Maximum score (0 to score_max)
    score_granularity: Literal['continuous', 'integer'] = 'integer'
    
    # STAR voting uses score_max
    
    # Tie-breaking
    tiebreak_method: Literal['random', 'lexicographic', 'none'] = 'random'


@dataclass
class HeterogeneousDistanceConfig:
    """Configuration for heterogeneous distance metrics.
    
    Allows different voters to use different distance metrics based on
    their position in space. Two strategies are available:
    
    1. center_extreme: Center voters use one metric (default L2),
       extreme voters use cosine. Controlled by a threshold parameter.
       
    2. radial_steps: Different metrics at different radial distances.
       L1 in center, L2 further out, Chebyshev at extremes.
       Boundaries defined by a scaling function.
    """
    
    # Whether heterogeneous distance is enabled
    enabled: bool = False
    
    # Strategy type
    strategy: Literal['center_extreme', 'radial_steps'] = 'center_extreme'
    
    # =========================================================================
    # Center-Extreme Strategy Parameters
    # =========================================================================
    
    # Metric used by center voters
    center_metric: Literal['l2', 'l1', 'cosine', 'chebyshev'] = 'l2'
    
    # Metric used by extreme voters (always cosine by default)
    extreme_metric: Literal['l2', 'l1', 'cosine', 'chebyshev'] = 'cosine'
    
    # Threshold for "extreme" classification (distance from center)
    # Expressed as fraction of max possible distance (0.0-1.0)
    # Voters beyond this threshold use extreme_metric
    extreme_threshold: float = 0.5
    
    # =========================================================================
    # Radial Steps Strategy Parameters
    # =========================================================================
    
    # Ordered list of metrics from center outward
    # Default: L1 (center) -> L2 (middle) -> Chebyshev (far)
    radial_metrics: List[str] = field(
        default_factory=lambda: ['l1', 'l2', 'chebyshev']
    )
    
    # How boundaries between metrics are spaced
    # 'linear': Equal spacing
    # 'logarithmic': Boundaries at log scale (more near center)
    # 'exponential': Boundaries at exp scale (more near edge)
    radial_scaling: Literal['linear', 'logarithmic', 'exponential'] = 'linear'
    
    # Base parameter for non-linear scaling
    # For logarithmic: log_base controls compression
    # For exponential: exp_rate controls expansion
    scaling_parameter: float = 2.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.extreme_threshold < 0 or self.extreme_threshold > 1:
            raise ValueError("extreme_threshold must be in [0, 1]")
        if len(self.radial_metrics) < 2:
            raise ValueError("radial_metrics must have at least 2 metrics")
        valid_metrics = {'l1', 'l2', 'cosine', 'chebyshev'}
        for m in self.radial_metrics:
            if m not in valid_metrics:
                raise ValueError(f"Invalid metric '{m}' in radial_metrics")
        if self.scaling_parameter <= 0:
            raise ValueError("scaling_parameter must be > 0")


@dataclass 
class UtilityConfig:
    """Configuration for utility computation."""
    
    # Utility function type
    function: Literal['gaussian', 'quadratic', 'linear', 'exponential'] = 'gaussian'
    
    # Distance metric (used when heterogeneous is disabled)
    distance_metric: Literal['l2', 'l1', 'cosine', 'chebyshev'] = 'l2'
    
    # Heterogeneous distance configuration
    heterogeneous_distance: HeterogeneousDistanceConfig = field(
        default_factory=HeterogeneousDistanceConfig
    )
    
    # Gaussian utility parameters
    sigma_factor: float = 0.5  # sigma = sigma_factor * sqrt(n_dim)
    
    # Exponential utility parameters
    decay_rate: float = 2.0


@dataclass
class SimulationConfig:
    """Master configuration for simulation runs."""
    
    # Unique experiment identifier
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Core simulation parameters
    n_profiles: int = 1000
    n_voters: int = 25
    n_candidates: int = 3
    
    # Voting rules to evaluate (list of rule names)
    voting_rules: List[str] = field(default_factory=lambda: ['plurality', 'borda', 'irv'])
    
    # Sub-configurations
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    manipulation: ManipulationConfig = field(default_factory=ManipulationConfig)
    voting_rule_config: VotingRuleConfig = field(default_factory=VotingRuleConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    
    # Reproducibility
    rng_seed: Optional[int] = None
    
    # Epsilon for floating point comparisons
    epsilon: float = 1e-9
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        if self.n_profiles < 1:
            raise ValueError("n_profiles must be >= 1")
        if self.n_voters < 1:
            raise ValueError("n_voters must be >= 1")
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be >= 2")
        
        self.geometry.validate()
        self.manipulation.validate()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for storage."""
        result = {
            'experiment_id': self.experiment_id,
            'created_at': self.created_at,
            'n_profiles': self.n_profiles,
            'n_voters': self.n_voters,
            'n_candidates': self.n_candidates,
            'voting_rules': self.voting_rules,
            'geometry_method': self.geometry.method,
            'geometry_n_dim': self.geometry.n_dim,
            'geometry_phi': self.geometry.phi,
            'manipulation_enabled': self.manipulation.enabled,
            'manipulation_fraction': self.manipulation.manipulator_fraction,
            'manipulation_strategy': self.manipulation.strategy,
            'utility_function': self.utility.function,
            'utility_distance_metric': self.utility.distance_metric,
            'utility_sigma_factor': self.utility.sigma_factor,
            'rng_seed': self.rng_seed,
            'epsilon': self.epsilon,
        }
        
        # Add heterogeneous distance config if enabled
        het_config = self.utility.heterogeneous_distance
        result['heterogeneous_distance_enabled'] = het_config.enabled
        if het_config.enabled:
            result['heterogeneous_strategy'] = het_config.strategy
            result['heterogeneous_center_metric'] = het_config.center_metric
            result['heterogeneous_extreme_metric'] = het_config.extreme_metric
            result['heterogeneous_extreme_threshold'] = het_config.extreme_threshold
            result['heterogeneous_radial_metrics'] = het_config.radial_metrics
            result['heterogeneous_radial_scaling'] = het_config.radial_scaling
            result['heterogeneous_scaling_parameter'] = het_config.scaling_parameter
        
        return result


# Registry of available voting rules
AVAILABLE_VOTING_RULES = {
    # Ordinal rules
    'plurality': {'type': 'ordinal', 'description': 'First-past-the-post voting'},
    'borda': {'type': 'ordinal', 'description': 'Borda count'},
    'irv': {'type': 'ordinal', 'description': 'Instant Runoff Voting'},
    'condorcet': {'type': 'ordinal', 'description': 'Condorcet method (returns None if no winner)'},
    'minimax': {'type': 'ordinal', 'description': 'Minimax Condorcet method'},
    'copeland': {'type': 'ordinal', 'description': 'Copeland method'},
    'schulze': {'type': 'ordinal', 'description': 'Schulze method'},
    'kemeny_young': {'type': 'ordinal', 'description': 'Kemeny-Young method'},
    'ranked_pairs': {'type': 'ordinal', 'description': 'Ranked pairs (Tideman)'},
    'anti_plurality': {'type': 'ordinal', 'description': 'Vote against least favorite'},
    'veto': {'type': 'ordinal', 'description': 'Veto (anti-plurality alias)'},
    'coombs': {'type': 'ordinal', 'description': 'Coombs elimination'},
    'bucklin': {'type': 'ordinal', 'description': 'Bucklin voting'},
    'nanson': {'type': 'ordinal', 'description': 'Nanson method'},
    'baldwin': {'type': 'ordinal', 'description': 'Baldwin method'},
    
    # Cardinal rules
    'approval': {'type': 'cardinal', 'description': 'Approval voting'},
    'score': {'type': 'cardinal', 'description': 'Score/range voting'},
    'star': {'type': 'cardinal', 'description': 'STAR voting (Score Then Automatic Runoff)'},
    'utilitarian': {'type': 'cardinal', 'description': 'Maximize sum of utilities'},
    'median': {'type': 'cardinal', 'description': 'Median voter utility'},
    'quadratic': {'type': 'cardinal', 'description': 'Quadratic voting'},
}

# Available geometry methods
AVAILABLE_GEOMETRY_METHODS = {
    'uniform': 'Uniform random distribution in hypercube',
    'clustered': 'Clustered around center with configurable variance',
    'single_peaked': '1D left-right spectrum',
    'polarized': 'Two opposing clusters with small center',
    '1d': '1D uniform distribution',
    '2d': '2D with triangle candidate placement',
    'custom': 'Custom geometry (provide positions directly)',
}

# Available manipulation strategies
AVAILABLE_MANIPULATION_STRATEGIES = {
    'bullet': 'Only vote for top choice (for approval/score)',
    'compromise': 'Rank viable alternative higher to block worse candidate',
    'burial': 'Rank viable competitor lower than sincere preference',
    'pushover': 'Support weak opponent to face in runoff',
    'optimal': 'Compute optimal strategic vote (expensive)',
}

# Available heterogeneous distance strategies
AVAILABLE_HETEROGENEOUS_STRATEGIES = {
    'center_extreme': 'Center voters use one metric, extremists use another (e.g., cosine)',
    'radial_steps': 'Different metrics at different radial distances (L1→L2→L∞)',
}

# Available radial scaling functions
AVAILABLE_RADIAL_SCALING = {
    'linear': 'Equal spacing between metric boundaries',
    'logarithmic': 'More boundaries near center (log scale)',
    'exponential': 'More boundaries near edge (exp scale)',
}

# Available distance metrics
AVAILABLE_DISTANCE_METRICS = {
    'l2': 'Euclidean (straight-line) distance',
    'l1': 'Manhattan (city-block) distance',
    'cosine': 'Cosine distance (directional similarity)',
    'chebyshev': 'Chebyshev (max coordinate difference)',
}


