"""
Utility computation functions for voting simulation.

Computes voter utilities for candidates based on spatial distances
using various distance metrics and utility functions.

Supports both homogeneous (all voters use same metric) and
heterogeneous (voters use different metrics based on position)
distance computation.
"""

import numpy as np
from typing import Literal, Optional

from .config import UtilityConfig


class UtilityComputer:
    """
    Computes utilities from spatial positions.
    
    Utility = f(distance) where f is a decay function.
    Higher utility = more preferred candidate.
    
    Supports heterogeneous distance metrics where different voters
    may use different distance functions based on their position.
    """
    
    def __init__(self, config: UtilityConfig):
        """
        Initialize utility computer.
        
        Args:
            config: Utility configuration
        """
        self.config = config
        
        # Initialize heterogeneous distance strategy if enabled
        self._het_strategy = None
        if config.heterogeneous_distance.enabled:
            from .heterogeneous_distance import create_strategy
            self._het_strategy = create_strategy(config.heterogeneous_distance)
    
    def compute_distances(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances from voters to candidates.
        
        If heterogeneous distance is enabled, different voters may use
        different distance metrics based on their position in space.
        
        Args:
            voter_positions: Shape (n_voters, n_dim) or (n_profiles, n_voters, n_dim)
            candidate_positions: Shape (n_candidates, n_dim) or (n_profiles, n_candidates, n_dim)
            
        Returns:
            Distance matrix of shape (..., n_voters, n_candidates)
        """
        # Check if heterogeneous distance is enabled
        if self._het_strategy is not None:
            return self._compute_distances_heterogeneous(
                voter_positions, candidate_positions
            )
        
        # Standard homogeneous distance computation
        metric = self.config.distance_metric
        
        # Handle both 2D and 3D inputs
        if voter_positions.ndim == 2:
            # Single profile: (n_voters, n_dim), (n_candidates, n_dim)
            return self._compute_distances_single(
                voter_positions, candidate_positions, metric
            )
        else:
            # Multiple profiles: (n_profiles, n_voters, n_dim)
            return self._compute_distances_batch(
                voter_positions, candidate_positions, metric
            )
    
    def _compute_distances_heterogeneous(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """Compute distances using heterogeneous metrics."""
        if voter_positions.ndim == 2:
            # Single profile
            return self._het_strategy.compute_distances(
                voter_positions, candidate_positions
            )
        else:
            # Batch of profiles
            n_profiles = voter_positions.shape[0]
            n_voters = voter_positions.shape[1]
            n_candidates = candidate_positions.shape[1]
            
            distances = np.zeros((n_profiles, n_voters, n_candidates))
            
            for i in range(n_profiles):
                distances[i] = self._het_strategy.compute_distances(
                    voter_positions[i], candidate_positions[i]
                )
            
            return distances
    
    def get_voter_metrics(
        self,
        voter_positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get the distance metric assigned to each voter.
        
        Only available when heterogeneous distance is enabled.
        
        Args:
            voter_positions: Voter positions (n_voters, n_dim)
            
        Returns:
            Array of metric names per voter, or None if homogeneous
        """
        if self._het_strategy is not None:
            return self._het_strategy.get_voter_metrics(voter_positions)
        return None
    
    def get_metric_distribution(
        self,
        voter_positions: np.ndarray
    ) -> dict:
        """
        Get the distribution of metrics across voters.
        
        Args:
            voter_positions: Voter positions
            
        Returns:
            Dictionary mapping metric names to fractions
        """
        if self._het_strategy is not None:
            return self._het_strategy.get_metric_distribution(voter_positions)
        
        # Homogeneous - all use the same metric
        return {self.config.distance_metric: 1.0}
    
    def _compute_distances_single(
        self,
        voters: np.ndarray,  # (n_voters, n_dim)
        candidates: np.ndarray,  # (n_candidates, n_dim)
        metric: str
    ) -> np.ndarray:
        """Compute distances for single profile."""
        # Expand for broadcasting: (n_voters, 1, n_dim) - (n_candidates, n_dim)
        diff = voters[:, np.newaxis, :] - candidates[np.newaxis, :, :]
        
        if metric == 'l2':
            # Euclidean distance
            return np.sqrt(np.sum(diff ** 2, axis=-1))
        
        elif metric == 'l1':
            # Manhattan distance
            return np.sum(np.abs(diff), axis=-1)
        
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            voters_norm = voters / (np.linalg.norm(voters, axis=-1, keepdims=True) + 1e-10)
            cands_norm = candidates / (np.linalg.norm(candidates, axis=-1, keepdims=True) + 1e-10)
            similarity = np.sum(
                voters_norm[:, np.newaxis, :] * cands_norm[np.newaxis, :, :],
                axis=-1
            )
            return 1.0 - similarity
        
        elif metric == 'chebyshev':
            # Maximum coordinate difference
            return np.max(np.abs(diff), axis=-1)
        
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def _compute_distances_batch(
        self,
        voters: np.ndarray,  # (n_profiles, n_voters, n_dim)
        candidates: np.ndarray,  # (n_profiles, n_candidates, n_dim)
        metric: str
    ) -> np.ndarray:
        """Compute distances for multiple profiles (vectorized)."""
        # Shape: (n_profiles, n_voters, 1, n_dim) - (n_profiles, 1, n_candidates, n_dim)
        diff = voters[:, :, np.newaxis, :] - candidates[:, np.newaxis, :, :]
        
        if metric == 'l2':
            return np.sqrt(np.sum(diff ** 2, axis=-1))
        
        elif metric == 'l1':
            return np.sum(np.abs(diff), axis=-1)
        
        elif metric == 'cosine':
            voters_norm = voters / (np.linalg.norm(voters, axis=-1, keepdims=True) + 1e-10)
            cands_norm = candidates / (np.linalg.norm(candidates, axis=-1, keepdims=True) + 1e-10)
            similarity = np.sum(
                voters_norm[:, :, np.newaxis, :] * cands_norm[:, np.newaxis, :, :],
                axis=-1
            )
            return 1.0 - similarity
        
        elif metric == 'chebyshev':
            return np.max(np.abs(diff), axis=-1)
        
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def compute_utilities(
        self,
        distances: np.ndarray,
        n_dim: int
    ) -> np.ndarray:
        """
        Convert distances to utilities using configured function.
        
        Args:
            distances: Distance matrix from compute_distances
            n_dim: Dimensionality of space (affects sigma scaling)
            
        Returns:
            Utility matrix of same shape as distances
        """
        func = self.config.function
        
        if func == 'gaussian':
            return self._utility_gaussian(distances, n_dim)
        elif func == 'quadratic':
            return self._utility_quadratic(distances, n_dim)
        elif func == 'linear':
            return self._utility_linear(distances, n_dim)
        elif func == 'exponential':
            return self._utility_exponential(distances, n_dim)
        else:
            raise ValueError(f"Unknown utility function: {func}")
    
    def _utility_gaussian(self, distances: np.ndarray, n_dim: int) -> np.ndarray:
        """
        Gaussian utility: exp(-d^2 / (2*sigma^2))
        
        Sigma scales with sqrt(n_dim) for consistent behavior.
        """
        sigma = self.config.sigma_factor * np.sqrt(n_dim)
        return np.exp(-distances ** 2 / (2 * sigma ** 2))
    
    def _utility_quadratic(self, distances: np.ndarray, n_dim: int) -> np.ndarray:
        """
        Quadratic utility: max(0, 1 - d^2 / d_max^2)
        
        d_max is the maximum possible distance in the space.
        """
        d_max = self._get_d_max(n_dim)
        return np.maximum(0, 1 - distances ** 2 / d_max ** 2)
    
    def _utility_linear(self, distances: np.ndarray, n_dim: int) -> np.ndarray:
        """
        Linear utility: max(0, 1 - d / d_max)
        
        Simple linear decay from 1 at d=0 to 0 at d=d_max.
        """
        d_max = self._get_d_max(n_dim)
        return np.maximum(0, 1 - distances / d_max)
    
    def _utility_exponential(self, distances: np.ndarray, n_dim: int) -> np.ndarray:
        """
        Exponential utility: exp(-decay_rate * d)
        """
        return np.exp(-self.config.decay_rate * distances)
    
    def _get_d_max(self, n_dim: int) -> float:
        """Get maximum distance for the space [0,1]^n_dim."""
        metric = self.config.distance_metric
        
        if metric == 'l2':
            return np.sqrt(n_dim)  # Diagonal of unit hypercube
        elif metric == 'l1':
            return n_dim  # Sum of all coordinate differences
        elif metric == 'cosine':
            return 2.0  # Maximum cosine distance
        elif metric == 'chebyshev':
            return 1.0  # Maximum single coordinate difference
        else:
            return np.sqrt(n_dim)  # Default


def compute_distances(
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    """
    Convenience function to compute distances.
    
    Args:
        voter_positions: Voter position array
        candidate_positions: Candidate position array
        metric: Distance metric ('l2', 'l1', 'cosine', 'chebyshev')
        
    Returns:
        Distance matrix
    """
    config = UtilityConfig(distance_metric=metric)
    computer = UtilityComputer(config)
    return computer.compute_distances(voter_positions, candidate_positions)


def compute_utilities(
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    utility_func: str = 'gaussian',
    distance_metric: str = 'l2',
    sigma_factor: float = 0.5
) -> np.ndarray:
    """
    Convenience function to compute utilities from positions.
    
    Args:
        voter_positions: Voter position array
        candidate_positions: Candidate position array
        utility_func: Utility function type
        distance_metric: Distance metric
        sigma_factor: Sigma factor for Gaussian utility
        
    Returns:
        Utility matrix
    """
    config = UtilityConfig(
        function=utility_func,
        distance_metric=distance_metric,
        sigma_factor=sigma_factor
    )
    computer = UtilityComputer(config)
    
    distances = computer.compute_distances(voter_positions, candidate_positions)
    n_dim = voter_positions.shape[-1]
    
    return computer.compute_utilities(distances, n_dim)


def utilities_to_rankings(
    utilities: np.ndarray,
    epsilon: float = 1e-9
) -> np.ndarray:
    """
    Convert utility matrix to ordinal rankings.
    
    Args:
        utilities: Utility matrix (..., n_voters, n_candidates)
        epsilon: Small noise for tie-breaking
        
    Returns:
        Rankings matrix where rankings[..., v, i] is the candidate
        ranked at position i by voter v (0=best)
    """
    # Add tiny noise for tie-breaking
    noisy_utilities = utilities + np.random.uniform(
        0, epsilon, utilities.shape
    )
    
    # argsort gives indices that would sort (ascending)
    # negate to get descending order (highest utility first)
    rankings = np.argsort(-noisy_utilities, axis=-1)
    
    return rankings


def rankings_to_rank_positions(rankings: np.ndarray, n_candidates: int) -> np.ndarray:
    """
    Convert rankings to rank positions.
    
    Args:
        rankings: Rankings where rankings[..., i] is candidate at rank i
        n_candidates: Number of candidates
        
    Returns:
        Rank positions where positions[..., c] is the rank of candidate c
    """
    # Create output array
    positions = np.empty_like(rankings)
    
    # For each rank position, store the rank value
    for rank in range(n_candidates):
        # At each position, store which rank this candidate has
        np.put_along_axis(
            positions,
            rankings,
            np.arange(n_candidates),
            axis=-1
        )
    
    return positions


