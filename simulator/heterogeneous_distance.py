"""
Heterogeneous distance metrics for voting simulation.

Allows different voters to use different distance metrics based on
their position in space. This models cognitive heterogeneity where
voters may perceive political "distance" differently.

Two main strategies are provided:
1. CenterExtremeStrategy: Center voters use one metric, extremists use another
2. RadialStepsStrategy: Different metrics at different radial distances
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass

from .config import HeterogeneousDistanceConfig


# =============================================================================
# Distance Computation Functions
# =============================================================================

def compute_l2_distance(
    voters: np.ndarray,  # (n_voters, n_dim)
    candidates: np.ndarray  # (n_candidates, n_dim)
) -> np.ndarray:
    """Euclidean (L2) distance."""
    diff = voters[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def compute_l1_distance(
    voters: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray:
    """Manhattan (L1) distance."""
    diff = voters[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    return np.sum(np.abs(diff), axis=-1)


def compute_cosine_distance(
    voters: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity."""
    voters_norm = voters / (np.linalg.norm(voters, axis=-1, keepdims=True) + 1e-10)
    cands_norm = candidates / (np.linalg.norm(candidates, axis=-1, keepdims=True) + 1e-10)
    similarity = np.sum(
        voters_norm[:, np.newaxis, :] * cands_norm[np.newaxis, :, :],
        axis=-1
    )
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity


def compute_chebyshev_distance(
    voters: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray:
    """Chebyshev (L∞) distance - maximum coordinate difference."""
    diff = voters[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    return np.max(np.abs(diff), axis=-1)


# Registry of distance functions
DISTANCE_FUNCTIONS = {
    'l1': compute_l1_distance,
    'l2': compute_l2_distance,
    'cosine': compute_cosine_distance,
    'chebyshev': compute_chebyshev_distance,
}


def compute_distance_single_metric(
    voters: np.ndarray,
    candidates: np.ndarray,
    metric: str
) -> np.ndarray:
    """Compute distances using a single metric."""
    if metric not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric}")
    return DISTANCE_FUNCTIONS[metric](voters, candidates)


# =============================================================================
# Voter Classification Functions
# =============================================================================

def compute_voter_centrality(
    voter_positions: np.ndarray,  # (n_voters, n_dim)
    center: Optional[np.ndarray] = None,  # (n_dim,) or None for auto
    position_min: float = 0.0,
    position_max: float = 1.0,
) -> np.ndarray:
    """
    Compute how "central" each voter is within a hypercube.

    Returns normalized distance from center (0 = center, 1 = farthest corner).

    Args:
        voter_positions: Voter position array (n_voters, n_dim)
        center: Center point (default: geometric center of the hypercube)
        position_min: Minimum coordinate value (hypercube lower bound)
        position_max: Maximum coordinate value (hypercube upper bound)

    Returns:
        Array of shape (n_voters,) with values in [0, 1]
    """
    if center is None:
        # Use geometric center of the configured hypercube.
        n_dim = voter_positions.shape[-1]
        center = np.full(n_dim, (position_min + position_max) / 2.0)

    # Compute distances from center
    distances = np.linalg.norm(voter_positions - center, axis=-1)

    # Normalize to [0, 1] based on max possible distance from center.
    n_dim = voter_positions.shape[-1]
    max_distance = np.sqrt(n_dim) * (position_max - position_min) / 2.0  # Half diagonal

    normalized = distances / max_distance
    return np.clip(normalized, 0, 1)


# =============================================================================
# Heterogeneous Distance Strategy Interface
# =============================================================================

class HeterogeneousDistanceStrategy(ABC):
    """
    Abstract base class for heterogeneous distance strategies.

    Each strategy assigns different distance metrics to voters based
    on their position in space, then computes distances accordingly.
    """

    @abstractmethod
    def compute_distances(
        self,
        voter_positions: np.ndarray,  # (n_voters, n_dim)
        candidate_positions: np.ndarray,  # (n_candidates, n_dim)
    ) -> np.ndarray:
        """
        Compute heterogeneous distances.

        Args:
            voter_positions: Voter positions (n_voters, n_dim)
            candidate_positions: Candidate positions (n_candidates, n_dim)

        Returns:
            Distance matrix of shape (n_voters, n_candidates)
        """
        pass

    @abstractmethod
    def get_voter_metrics(
        self,
        voter_positions: np.ndarray
    ) -> np.ndarray:
        """
        Get the metric assigned to each voter.

        Args:
            voter_positions: Voter positions (n_voters, n_dim)

        Returns:
            Array of metric names for each voter
        """
        pass

    def get_metric_distribution(
        self,
        voter_positions: np.ndarray
    ) -> Dict[str, float]:
        """
        Get the distribution of metrics across voters.

        Returns:
            Dictionary mapping metric names to fractions
        """
        metrics = self.get_voter_metrics(voter_positions)
        unique, counts = np.unique(metrics, return_counts=True)
        n_voters = len(voter_positions)
        return {m: c / n_voters for m, c in zip(unique, counts)}


# =============================================================================
# Strategy 1: Center-Extreme
# =============================================================================

class CenterExtremeStrategy(HeterogeneousDistanceStrategy):
    """
    Strategy where center voters use one metric and extremists use another.

    Voters are classified by *percentile* of centrality (normalized L2 distance
    to the hypercube center). Let t be in [0, 1]. The closest floor(t * N)
    voters (smallest centrality) use center_metric, and the remaining voters
    use extreme_metric.

    This models the idea that moderate voters care about overall policy
    similarity (L2), while extreme voters care about ideological direction
    (cosine).
    """

    def __init__(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        threshold: float = 0.5,
        threshold_mode: Literal['percentile', 'radius'] = 'percentile',
        center: Optional[np.ndarray] = None,
        position_min: float = 0.0,
        position_max: float = 1.0,
    ):
        """
        Initialize the strategy.

        Args:
            center_metric: Metric for central voters
            extreme_metric: Metric for extreme voters
            threshold: Center percentile t (0-1) for classification
            threshold_mode: 'percentile' (rank cutoff) or 'radius' (cent(x) cutoff)
            center: Optional fixed center point
        """
        self.center_metric = center_metric
        self.extreme_metric = extreme_metric
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.center = center
        self.position_min = position_min
        self.position_max = position_max

        # Validate metrics
        for m in [center_metric, extreme_metric]:
            if m not in DISTANCE_FUNCTIONS:
                raise ValueError(f"Unknown metric: {m}")
        if threshold_mode not in ('percentile', 'radius'):
            raise ValueError("threshold_mode must be 'percentile' or 'radius'")

    def _percentile_cutoff(
        self,
        centrality: np.ndarray,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute the percentile cutoff for center assignment.

        Returns:
            order: indices that sort voters by increasing centrality
            k: number of voters assigned to center metric (floor(t * N))
            effective_radius: centrality value at the cutoff voter
        """
        n_voters = int(centrality.shape[0])
        k = int(np.floor(float(self.threshold) * n_voters))

        # Stable sort for deterministic tie-handling across equal centralities.
        order = np.argsort(centrality, kind="mergesort")

        if k <= 0:
            effective_radius = 0.0
        else:
            effective_radius = float(centrality[order[k - 1]])

        return order, k, effective_radius

    def get_effective_radius(
        self,
        voter_positions: np.ndarray
    ) -> float:
        """Return the effective_radius for this profile under the current t."""
        centrality = compute_voter_centrality(
            voter_positions,
            center=self.center,
            position_min=self.position_min,
            position_max=self.position_max,
        )
        if self.threshold_mode == 'radius':
            return float(self.threshold)
        _, _, effective_radius = self._percentile_cutoff(centrality)
        return effective_radius

    def get_voter_metrics(
        self,
        voter_positions: np.ndarray
    ) -> np.ndarray:
        """Classify each voter as using center or extreme metric."""
        centrality = compute_voter_centrality(
            voter_positions,
            center=self.center,
            position_min=self.position_min,
            position_max=self.position_max,
        )

        if self.threshold_mode == 'radius':
            return np.where(
                centrality <= self.threshold,
                self.center_metric,
                self.extreme_metric
            )

        order, k, _ = self._percentile_cutoff(centrality)

        metrics = np.full(len(voter_positions), self.extreme_metric, dtype=object)
        if k > 0:
            metrics[order[:k]] = self.center_metric
        return metrics

    def compute_distances(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """Compute distances with heterogeneous metrics."""
        n_voters = voter_positions.shape[0]
        n_candidates = candidate_positions.shape[0]

        # Classify voters
        metrics = self.get_voter_metrics(voter_positions)

        # Initialize result
        distances = np.zeros((n_voters, n_candidates))

        # Compute distances for each metric group
        for metric in [self.center_metric, self.extreme_metric]:
            mask = metrics == metric
            if np.any(mask):
                group_positions = voter_positions[mask]
                group_distances = compute_distance_single_metric(
                    group_positions, candidate_positions, metric
                )
                distances[mask] = group_distances

        return distances


# =============================================================================
# Strategy 2: Radial Steps
# =============================================================================

class RadialStepsStrategy(HeterogeneousDistanceStrategy):
    """
    Strategy with different metrics at different radial distances.

    The space is divided into concentric regions, with each region
    using a different distance metric. By default:
    - Inner region: L1 (Manhattan)
    - Middle region: L2 (Euclidean)
    - Outer region: Chebyshev (L∞)

    The boundaries between regions are controlled by a scaling function:
    - Linear: Equal spacing
    - Logarithmic: More regions near center
    - Exponential: More regions near edge
    """

    def __init__(
        self,
        metrics: List[str] = None,
        scaling: str = 'linear',
        scaling_parameter: float = 2.0,
        center: Optional[np.ndarray] = None,
        position_min: float = 0.0,
        position_max: float = 1.0,
    ):
        """
        Initialize the strategy.

        Args:
            metrics: Ordered list of metrics from center outward
            scaling: Boundary scaling function ('linear', 'logarithmic', 'exponential')
            scaling_parameter: Parameter for non-linear scaling
            center: Optional fixed center point
        """
        self.metrics = metrics if metrics else ['l1', 'l2', 'chebyshev']
        self.scaling = scaling
        self.scaling_parameter = scaling_parameter
        self.center = center
        self.position_min = position_min
        self.position_max = position_max

        # Validate metrics
        for m in self.metrics:
            if m not in DISTANCE_FUNCTIONS:
                raise ValueError(f"Unknown metric: {m}")

        if len(self.metrics) < 2:
            raise ValueError("Need at least 2 metrics for radial steps")

        # Compute boundaries
        self._boundaries = self._compute_boundaries()

    def _compute_boundaries(self) -> np.ndarray:
        """
        Compute the boundaries between metric regions.

        Returns:
            Array of boundary values in [0, 1], length = len(metrics) - 1
        """
        n_boundaries = len(self.metrics) - 1

        if self.scaling == 'linear':
            # Equal spacing
            boundaries = np.linspace(0, 1, len(self.metrics) + 1)[1:-1]

        elif self.scaling == 'logarithmic':
            # More boundaries near center (log scale)
            # Transform [1, base^n] -> [0, 1]
            base = self.scaling_parameter
            log_points = np.logspace(0, 1, len(self.metrics) + 1, base=base)
            log_points = (log_points - 1) / (base - 1)  # Normalize to [0, 1]
            boundaries = log_points[1:-1]

        elif self.scaling == 'exponential':
            # More boundaries near edge (exp scale)
            # Use inverse of logarithmic
            base = self.scaling_parameter
            exp_points = np.linspace(0, 1, len(self.metrics) + 1)
            exp_points = (np.power(base, exp_points) - 1) / (base - 1)
            boundaries = exp_points[1:-1]

        else:
            raise ValueError(f"Unknown scaling: {self.scaling}")

        return boundaries

    def get_voter_metrics(
        self,
        voter_positions: np.ndarray
    ) -> np.ndarray:
        """Assign metrics based on radial distance."""
        centrality = compute_voter_centrality(
            voter_positions,
            center=self.center,
            position_min=self.position_min,
            position_max=self.position_max,
        )

        # Determine which region each voter belongs to
        # Using searchsorted to find the appropriate bucket
        region_indices = np.searchsorted(self._boundaries, centrality)

        # Map region indices to metrics
        metrics = np.array([self.metrics[min(i, len(self.metrics) - 1)]
                          for i in region_indices])

        return metrics

    def compute_distances(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """Compute distances with heterogeneous metrics."""
        n_voters = voter_positions.shape[0]
        n_candidates = candidate_positions.shape[0]

        # Classify voters
        metrics = self.get_voter_metrics(voter_positions)

        # Initialize result
        distances = np.zeros((n_voters, n_candidates))

        # Compute distances for each metric group
        unique_metrics = np.unique(metrics)
        for metric in unique_metrics:
            mask = metrics == metric
            if np.any(mask):
                group_positions = voter_positions[mask]
                group_distances = compute_distance_single_metric(
                    group_positions, candidate_positions, metric
                )
                distances[mask] = group_distances

        return distances

    def get_boundaries(self) -> List[float]:
        """Get the computed boundaries between regions."""
        return self._boundaries.tolist()


# =============================================================================
# Factory Function
# =============================================================================

def create_strategy(
    config: HeterogeneousDistanceConfig,
    position_min: float = 0.0,
    position_max: float = 1.0,
) -> HeterogeneousDistanceStrategy:
    """
    Create a heterogeneous distance strategy from configuration.

    Args:
        config: HeterogeneousDistanceConfig

    Returns:
        Appropriate strategy instance
    """
    if config.strategy == 'center_extreme':
        return CenterExtremeStrategy(
            center_metric=config.center_metric,
            extreme_metric=config.extreme_metric,
            threshold=config.extreme_threshold,
            threshold_mode=config.center_extreme_threshold_mode,
            position_min=position_min,
            position_max=position_max,
        )

    elif config.strategy == 'radial_steps':
        return RadialStepsStrategy(
            metrics=config.radial_metrics,
            scaling=config.radial_scaling,
            scaling_parameter=config.scaling_parameter,
            position_min=position_min,
            position_max=position_max,
        )

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


# =============================================================================
# Convenience Function for Heterogeneous Distance Computation
# =============================================================================

def compute_heterogeneous_distances(
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    config: HeterogeneousDistanceConfig,
    position_min: float = 0.0,
    position_max: float = 1.0,
) -> np.ndarray:
    """
    Compute distances using heterogeneous metrics.

    Convenience function that handles single or batch inputs.

    Args:
        voter_positions: Shape (n_voters, n_dim) or (n_profiles, n_voters, n_dim)
        candidate_positions: Shape (n_candidates, n_dim) or (n_profiles, n_candidates, n_dim)
        config: Heterogeneous distance configuration

    Returns:
        Distance matrix of shape (..., n_voters, n_candidates)
    """
    strategy = create_strategy(config, position_min=position_min, position_max=position_max)

    if voter_positions.ndim == 2:
        # Single profile
        return strategy.compute_distances(voter_positions, candidate_positions)

    else:
        # Batch of profiles
        n_profiles = voter_positions.shape[0]
        n_voters = voter_positions.shape[1]
        n_candidates = candidate_positions.shape[1]

        distances = np.zeros((n_profiles, n_voters, n_candidates))

        for i in range(n_profiles):
            distances[i] = strategy.compute_distances(
                voter_positions[i], candidate_positions[i]
            )

        return distances


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_metric_distribution(
    voter_positions: np.ndarray,
    config: HeterogeneousDistanceConfig
) -> Dict[str, any]:
    """
    Analyze the distribution of metrics across voters.

    Args:
        voter_positions: Voter positions (can be single or batch)
        config: Heterogeneous distance configuration

    Returns:
        Dictionary with distribution statistics
    """
    strategy = create_strategy(config)

    if voter_positions.ndim == 2:
        # Single profile
        metrics = strategy.get_voter_metrics(voter_positions)
        distribution = strategy.get_metric_distribution(voter_positions)

        return {
            'per_voter_metrics': metrics,
            'distribution': distribution,
            'n_voters': len(voter_positions),
        }

    else:
        # Batch - analyze aggregate
        all_distributions = []
        for i in range(voter_positions.shape[0]):
            dist = strategy.get_metric_distribution(voter_positions[i])
            all_distributions.append(dist)

        # Aggregate
        all_metrics = set()
        for d in all_distributions:
            all_metrics.update(d.keys())

        avg_distribution = {}
        for m in all_metrics:
            values = [d.get(m, 0) for d in all_distributions]
            avg_distribution[m] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

        return {
            'aggregate_distribution': avg_distribution,
            'n_profiles': voter_positions.shape[0],
            'n_voters_per_profile': voter_positions.shape[1],
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Strategies
    'HeterogeneousDistanceStrategy',
    'CenterExtremeStrategy',
    'RadialStepsStrategy',

    # Functions
    'create_strategy',
    'compute_heterogeneous_distances',
    'compute_voter_centrality',
    'analyze_metric_distribution',

    # Distance functions
    'DISTANCE_FUNCTIONS',
    'compute_distance_single_metric',
]
