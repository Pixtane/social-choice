"""
Utility computation module.

Handles distance and utility computations for spatial voting models.
"""

import numpy as np
from typing import Optional
from .config import UtilityConfig
from .heterogeneous_distance import compute_heterogeneous_distances
from .config import GeometryConfig
from .heterogeneous_distance import create_strategy


def utilities_to_rankings(utilities: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Convert utilities to preference rankings.

    Args:
        utilities: (n_voters, n_candidates) utility matrix
        epsilon: Tolerance for considering utilities as tied

    Returns:
        (n_voters, n_candidates) ranking matrix where ranking[v] is a permutation
        of [0, 1, ..., n_candidates-1] in preference order (best first)

    Notes:
        - Epsilon-ties are broken arbitrarily (by index order via argsort stability)
    """
    # argsort in descending order of utility
    rankings = np.argsort(-utilities, axis=1)
    return rankings.astype(np.int8)


def compute_distances(
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    """
    Compute distances between voters and candidates.

    Args:
        voter_positions: (n_voters, n_dim) array of voter positions
        candidate_positions: (n_candidates, n_dim) array of candidate positions
        metric: Distance metric to use ('l2', 'l1', 'linf', 'cosine')

    Returns:
        (n_voters, n_candidates) distance matrix
    """
    n_voters = voter_positions.shape[0]
    n_candidates = candidate_positions.shape[0]
    distances = np.zeros((n_voters, n_candidates))

    for v in range(n_voters):
        for c in range(n_candidates):
            diff = voter_positions[v] - candidate_positions[c]

            if metric == 'l2':
                distances[v, c] = np.linalg.norm(diff, ord=2)
            elif metric == 'l1':
                distances[v, c] = np.linalg.norm(diff, ord=1)
            elif metric == 'linf' or metric == 'chebyshev':
                distances[v, c] = np.linalg.norm(diff, ord=np.inf)
            elif metric == 'cosine':
                # Cosine distance = 1 - cosine similarity
                vv = voter_positions[v]
                cc = candidate_positions[c]
                denom = (np.linalg.norm(vv) + 1e-10) * (np.linalg.norm(cc) + 1e-10)
                cos_sim = float(np.dot(vv, cc) / denom)
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                distances[v, c] = 1.0 - cos_sim
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return distances


def _compute_opposite_corner(
    voter_positions: np.ndarray,
    position_min: float,
    position_max: float,
) -> np.ndarray:
    """
    Compute the "opposite" hypercube corner for each voter.

    The opposite corner is defined coordinate-wise as the bound (min/max)
    that is farthest from the voter's coordinate along that axis.
    """
    # Compare distance to each bound per coordinate
    dist_to_min = np.abs(voter_positions - position_min)
    dist_to_max = np.abs(voter_positions - position_max)
    use_min = dist_to_min >= dist_to_max
    return np.where(use_min, position_min, position_max)


def _pointwise_distance(
    a: np.ndarray,  # (n, n_dim)
    b: np.ndarray,  # (n, n_dim)
    metric: str,
) -> np.ndarray:
    """Distance between corresponding rows of a and b."""
    diff = a - b
    if metric == 'l2':
        return np.linalg.norm(diff, ord=2, axis=-1)
    if metric == 'l1':
        return np.linalg.norm(diff, ord=1, axis=-1)
    if metric == 'chebyshev' or metric == 'linf':
        return np.linalg.norm(diff, ord=np.inf, axis=-1)
    if metric == 'cosine':
        # 1 - cosine similarity, with epsilon to avoid division by zero
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
        sim = np.sum(a_norm * b_norm, axis=-1)
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim
    raise ValueError(f"Unknown metric: {metric}")


class UtilityComputer:
    """Computes utilities from spatial distances."""

    def __init__(self, config: UtilityConfig):
        """
        Initialize utility computer.

        Args:
            config: Utility configuration
        """
        self.config = config

    def compute_distances(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray,
        geometry: Optional[GeometryConfig] = None,
    ) -> np.ndarray:
        """
        Compute distances between voters and candidates.

        Args:
            voter_positions: (n_voters, n_dim) array
            candidate_positions: (n_candidates, n_dim) array

        Returns:
            (n_voters, n_candidates) distance matrix
        """
        het = self.config.heterogeneous_distance
        if het is not None and getattr(het, "enabled", False):
            # Use configured geometry bounds when available; default matches legacy [0, 1].
            pmin = geometry.position_min if geometry is not None else 0.0
            pmax = geometry.position_max if geometry is not None else 1.0
            return compute_heterogeneous_distances(
                voter_positions,
                candidate_positions,
                het,
                position_min=pmin,
                position_max=pmax,
            )

        return compute_distances(
            voter_positions,
            candidate_positions,
            self.config.distance_metric,
        )

    def compute_utilities(
        self,
        distances: np.ndarray,
        n_dim: int,
        *,
        voter_positions: Optional[np.ndarray] = None,
        geometry: Optional[GeometryConfig] = None,
    ) -> np.ndarray:
        """
        Convert distances to utilities.

        Args:
            distances: (n_voters, n_candidates) distance matrix
            n_dim: Number of spatial dimensions

        Returns:
            (n_voters, n_candidates) utility matrix
        """
        utility_func = self.config.function

        if utility_func == 'gaussian':
            # Gaussian: u = exp(-d² / (2 * sigma²))
            sigma = self.config.sigma_factor * np.sqrt(n_dim)
            utilities = np.exp(-distances**2 / (2 * sigma**2))

        elif utility_func == 'quadratic':
            # Quadratic: u = max(0, 1 - (d / d_max)²)
            d_max = self.config.d_max
            if d_max is None:
                # Default assumes unit hypercube [0,1]^n_dim. If you change the geometry
                # bounds (e.g. to [-1,1]^n_dim), set UtilityConfig.d_max explicitly.
                d_max = np.sqrt(n_dim)
            utilities = np.maximum(0.0, 1.0 - (distances / d_max)**2)

        elif utility_func == 'linear':
            # Linear: utility directly reflects distance (inverted so closer = higher utility).
            # Utilities are computed as negative distances, then normalized per-voter
            # in main.py to ensure best candidate = 1, worst candidate = 0.
            # No per-voter normalization by opposite corner - utilities directly reflect
            # the raw distances computed with each voter's metric.
            utilities = -distances  # Closer candidates have higher (less negative) utilities

        else:
            raise ValueError(f"Unknown utility function: {utility_func}")

        return utilities
