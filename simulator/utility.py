"""
Utility computation module.

Handles distance and utility computations for spatial voting models.
"""

import numpy as np
from typing import Optional
from .config import UtilityConfig


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
                norm_v = np.linalg.norm(voter_positions[v])
                norm_c = np.linalg.norm(candidate_positions[c])
                if norm_v > 0 and norm_c > 0:
                    cos_sim = np.dot(voter_positions[v], candidate_positions[c]) / (norm_v * norm_c)
                    distances[v, c] = 1.0 - cos_sim
                else:
                    distances[v, c] = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    return distances


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
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between voters and candidates.
        
        Args:
            voter_positions: (n_voters, n_dim) array
            candidate_positions: (n_candidates, n_dim) array
        
        Returns:
            (n_voters, n_candidates) distance matrix
        """
        return compute_distances(
            voter_positions,
            candidate_positions,
            self.config.distance_metric
        )
    
    def compute_utilities(
        self,
        distances: np.ndarray,
        n_dim: int
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
            # Use fixed sigma based on 2D baseline for consistent scaling
            sigma = self.config.sigma_factor * np.sqrt(2.0)
            utilities = np.exp(-distances**2 / (2 * sigma**2))
        
        elif utility_func == 'quadratic':
            # Quadratic: u = max(0, 1 - (d / d_max)²)
            d_max = self.config.d_max
            if d_max is None:
                # Use mean distance scaled by dimension factor for consistent utility interpretation
                # This accounts for how distances scale with dimensionality while maintaining
                # comparable utility ranges (Dawson model expectation: VSE should decrease with dim)
                mean_dist = np.mean(distances)
                # Scale by sqrt(2) / sqrt(n_dim) to normalize relative to 2D baseline
                d_max = mean_dist * (np.sqrt(2.0) / np.sqrt(max(n_dim, 1)))
                if d_max < 1e-9:  # Fallback if all distances are near zero
                    d_max = np.sqrt(2.0)
            utilities = np.maximum(0.0, 1.0 - (distances / d_max)**2)
        
        elif utility_func == 'linear':
            # Linear: u = max(0, 1 - d / d_max)
            d_max = self.config.d_max
            if d_max is None:
                # Use mean distance scaled by dimension factor for consistent utility interpretation
                # This accounts for how distances scale with dimensionality while maintaining
                # comparable utility ranges (Dawson model expectation: VSE should decrease with dim)
                mean_dist = np.mean(distances)
                # Scale by sqrt(2) / sqrt(n_dim) to normalize relative to 2D baseline
                d_max = mean_dist * (np.sqrt(2.0) / np.sqrt(max(n_dim, 1)))
                if d_max < 1e-9:  # Fallback if all distances are near zero
                    d_max = np.sqrt(2.0)
            utilities = np.maximum(0.0, 1.0 - distances / d_max)
        
        else:
            raise ValueError(f"Unknown utility function: {utility_func}")
        
        return utilities
