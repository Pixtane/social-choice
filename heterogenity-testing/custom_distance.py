"""
Custom distance assignment utilities for heterogeneity tests.
"""

import numpy as np
from typing import List, Optional, Dict
from simulator.heterogeneous_distance import (
    compute_distance_single_metric, DISTANCE_FUNCTIONS
)


def assign_metrics_by_fraction(
    n_voters: int,
    fraction_metric1: float,
    metric1: str,
    metric2: str,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Assign metrics to voters based on fraction.
    
    Args:
        n_voters: Number of voters
        fraction_metric1: Fraction using metric1 (0-1)
        metric1: First metric name
        metric2: Second metric name
        rng: Random generator
        
    Returns:
        Array of metric names per voter
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_metric1 = int(fraction_metric1 * n_voters)
    metrics = np.full(n_voters, metric2, dtype=object)
    metrics[:n_metric1] = metric1
    
    # Shuffle to randomize assignment
    rng.shuffle(metrics)
    return metrics


def assign_metrics_by_extremity(
    voter_positions: np.ndarray,
    n_extreme: int,
    extreme_metric: str,
    default_metric: str,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Assign extreme metric to n_extreme most extreme voters.
    
    Args:
        voter_positions: Voter positions (n_voters, n_dim)
        n_extreme: Number of extreme voters
        extreme_metric: Metric for extreme voters
        default_metric: Metric for others
        center: Center point (default: 0.5 in all dims)
        
    Returns:
        Array of metric names per voter
    """
    if center is None:
        n_dim = voter_positions.shape[1]
        center = np.full(n_dim, 0.5)
    
    # Compute distances from center
    distances = np.linalg.norm(voter_positions - center, axis=1)
    
    # Find most extreme voters
    extreme_indices = np.argsort(-distances)[:n_extreme]
    
    metrics = np.full(len(voter_positions), default_metric, dtype=object)
    metrics[extreme_indices] = extreme_metric
    
    return metrics


def assign_metrics_by_radius(
    voter_positions: np.ndarray,
    boundaries: List[float],
    metrics: List[str],
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Assign metrics based on radial distance from center.
    
    Args:
        voter_positions: Voter positions (n_voters, n_dim)
        boundaries: List of boundary values (e.g., [0.3, 0.6])
        metrics: List of metrics for each region (e.g., ['l2', 'l1', 'chebyshev'])
        center: Center point
        
    Returns:
        Array of metric names per voter
    """
    if center is None:
        n_dim = voter_positions.shape[1]
        center = np.full(n_dim, 0.5)
    
    # Compute distances from center
    distances = np.linalg.norm(voter_positions - center, axis=1)
    
    # Normalize to [0, 1]
    n_dim = voter_positions.shape[1]
    max_distance = np.sqrt(n_dim) * 0.5
    normalized = distances / max_distance
    
    # Assign metrics based on boundaries
    result = np.full(len(voter_positions), metrics[-1], dtype=object)
    
    for i, boundary in enumerate(boundaries):
        mask = normalized < boundary
        if i < len(metrics):
            result[mask] = metrics[i]
    
    return result


def assign_metrics_randomly(
    n_voters: int,
    metrics: List[str],
    probabilities: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Randomly assign metrics to voters.
    
    Args:
        n_voters: Number of voters
        metrics: List of possible metrics
        probabilities: Probabilities for each metric (default: uniform)
        rng: Random generator
        
    Returns:
        Array of metric names per voter
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if probabilities is None:
        probabilities = [1.0 / len(metrics)] * len(metrics)
    
    return rng.choice(metrics, size=n_voters, p=probabilities)


def compute_distances_with_assignment(
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    metric_assignment: np.ndarray
) -> np.ndarray:
    """
    Compute distances using per-voter metric assignment.
    
    Args:
        voter_positions: Voter positions (n_voters, n_dim)
        candidate_positions: Candidate positions (n_candidates, n_dim)
        metric_assignment: Array of metric names per voter
        
    Returns:
        Distance matrix (n_voters, n_candidates)
    """
    n_voters = voter_positions.shape[0]
    n_candidates = candidate_positions.shape[0]
    distances = np.zeros((n_voters, n_candidates))
    
    # Compute distances for each unique metric
    unique_metrics = np.unique(metric_assignment)
    
    for metric in unique_metrics:
        mask = metric_assignment == metric
        if np.any(mask):
            group_positions = voter_positions[mask]
            group_distances = compute_distance_single_metric(
                group_positions, candidate_positions, metric
            )
            distances[mask] = group_distances
    
    return distances





