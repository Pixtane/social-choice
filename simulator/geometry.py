"""
Spatial geometry generation for voting simulation.

Generates voter and candidate positions in n-dimensional space
using various distribution methods.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import GeometryConfig


@dataclass
class SpatialProfile:
    """Container for generated spatial positions."""
    voter_positions: np.ndarray  # Shape: (n_profiles, n_voters, n_dim)
    candidate_positions: np.ndarray  # Shape: (n_profiles, n_candidates, n_dim)
    n_profiles: int
    n_voters: int
    n_candidates: int
    n_dim: int
    method: str


class GeometryGenerator:
    """
    Generates spatial voter and candidate positions.
    
    All methods return positions in [0, 1]^n_dim space.
    """
    
    def __init__(self, config: GeometryConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize generator.
        
        Args:
            config: Geometry configuration
            rng: NumPy random generator for reproducibility
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def generate(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int = 3
    ) -> SpatialProfile:
        """
        Generate spatial positions using configured method.
        
        Args:
            n_profiles: Number of election profiles
            n_voters: Number of voters per profile
            n_candidates: Number of candidates per profile
            
        Returns:
            SpatialProfile with voter and candidate positions
        """
        method = self.config.method
        
        if method == 'uniform':
            return self._gen_uniform(n_profiles, n_voters, n_candidates)
        elif method == 'clustered':
            return self._gen_clustered(n_profiles, n_voters, n_candidates)
        elif method == 'single_peaked':
            return self._gen_single_peaked(n_profiles, n_voters, n_candidates)
        elif method == 'polarized':
            return self._gen_polarized(n_profiles, n_voters, n_candidates)
        elif method == '1d':
            return self._gen_1d(n_profiles, n_voters, n_candidates)
        elif method == '2d':
            return self._gen_2d(n_profiles, n_voters, n_candidates)
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def _gen_uniform(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        Uniform random distribution in [0,1]^n_dim.
        
        Most unbiased spatial distribution - voters and candidates
        uniformly distributed throughout the space.
        """
        n_dim = self.config.n_dim
        pmin, pmax = self.config.position_min, self.config.position_max
        
        voter_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_voters, n_dim)
        )
        candidate_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_candidates, n_dim)
        )
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='uniform'
        )
    
    def _gen_clustered(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        Clustered distribution around random center.
        
        Voters are Gaussian-distributed around a random center.
        phi parameter controls variance (0=tight, 1=loose).
        """
        n_dim = self.config.n_dim
        phi = self.config.phi
        pmin, pmax = self.config.position_min, self.config.position_max
        
        # Variance scales with dimension and phi
        voter_variance = phi * np.sqrt(n_dim) * self.config.cluster_variance
        candidate_variance = voter_variance * 0.5
        
        voter_positions = np.empty((n_profiles, n_voters, n_dim))
        candidate_positions = np.empty((n_profiles, n_candidates, n_dim))
        
        for i in range(n_profiles):
            # Random center, avoiding edges
            margin = (pmax - pmin) * 0.3
            center = self.rng.uniform(pmin + margin, pmax - margin, n_dim)
            
            # Voters: Gaussian around center, clipped to bounds
            voter_positions[i] = np.clip(
                self.rng.normal(center, voter_variance, (n_voters, n_dim)),
                pmin, pmax
            )
            
            # Candidates: Also from cluster
            candidate_positions[i] = np.clip(
                self.rng.normal(center, candidate_variance, (n_candidates, n_dim)),
                pmin, pmax
            )
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='clustered'
        )
    
    def _gen_single_peaked(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        Single-peaked preferences on 1D spectrum.
        
        Forces 1D regardless of config. Candidates at fixed positions.
        Naturally produces single-peaked preferences (no Condorcet cycles).
        """
        n_dim = 1  # Force 1D
        pmin, pmax = self.config.position_min, self.config.position_max
        span = pmax - pmin
        
        voter_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_voters, n_dim)
        )
        
        # Fixed candidate positions along spectrum
        candidate_positions = np.zeros((n_profiles, n_candidates, n_dim))
        positions = np.linspace(pmin + 0.2 * span, pmax - 0.2 * span, n_candidates)
        for i in range(n_candidates):
            candidate_positions[:, i, 0] = positions[i]
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='single_peaked'
        )
    
    def _gen_polarized(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        Polarized electorate with two opposing clusters.
        
        Simulates political polarization with left/right factions
        and a small center group.
        """
        n_dim = self.config.n_dim
        pmin, pmax = self.config.position_min, self.config.position_max
        span = pmax - pmin
        
        # Define cluster centers
        left_center = np.full(n_dim, pmin + 0.15 * span)
        right_center = np.full(n_dim, pmax - 0.15 * span)
        middle_center = np.full(n_dim, (pmin + pmax) / 2)
        
        cluster_std = 0.08 * span * np.sqrt(n_dim)
        
        voter_positions = np.empty((n_profiles, n_voters, n_dim))
        candidate_positions = np.empty((n_profiles, n_candidates, n_dim))
        
        for i in range(n_profiles):
            # Random faction sizes
            faction_left = int(n_voters * self.rng.uniform(0.35, 0.50))
            faction_right = int(n_voters * self.rng.uniform(0.35, 0.50))
            faction_middle = n_voters - faction_left - faction_right
            
            idx = 0
            
            # Left faction
            voter_positions[i, idx:idx+faction_left] = np.clip(
                self.rng.normal(left_center, cluster_std, (faction_left, n_dim)),
                pmin, pmax
            )
            idx += faction_left
            
            # Right faction
            voter_positions[i, idx:idx+faction_right] = np.clip(
                self.rng.normal(right_center, cluster_std, (faction_right, n_dim)),
                pmin, pmax
            )
            idx += faction_right
            
            # Middle faction
            if faction_middle > 0:
                voter_positions[i, idx:idx+faction_middle] = np.clip(
                    self.rng.normal(middle_center, cluster_std, (faction_middle, n_dim)),
                    pmin, pmax
                )
            
            # Shuffle voters
            self.rng.shuffle(voter_positions[i])
            
            # Candidates: one near each major cluster
            if n_candidates >= 3:
                candidate_positions[i, 0] = np.clip(
                    self.rng.normal(left_center, cluster_std/2, n_dim), pmin, pmax
                )
                candidate_positions[i, 1] = np.clip(
                    self.rng.normal(middle_center, cluster_std/2, n_dim), pmin, pmax
                )
                candidate_positions[i, 2] = np.clip(
                    self.rng.normal(right_center, cluster_std/2, n_dim), pmin, pmax
                )
                # Any additional candidates random
                for j in range(3, n_candidates):
                    candidate_positions[i, j] = self.rng.uniform(pmin, pmax, n_dim)
            else:
                for j in range(n_candidates):
                    candidate_positions[i, j] = self.rng.uniform(pmin, pmax, n_dim)
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='polarized'
        )
    
    def _gen_1d(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        1D spatial model with uniform voters and random candidates.
        """
        n_dim = 1  # Force 1D
        pmin, pmax = self.config.position_min, self.config.position_max
        
        voter_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_voters, n_dim)
        )
        candidate_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_candidates, n_dim)
        )
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='1d'
        )
    
    def _gen_2d(
        self,
        n_profiles: int,
        n_voters: int,
        n_candidates: int
    ) -> SpatialProfile:
        """
        2D spatial model with triangle candidate placement.
        
        Voters uniform in square, candidates form equilateral triangle.
        """
        n_dim = 2  # Force 2D
        pmin, pmax = self.config.position_min, self.config.position_max
        
        voter_positions = self.rng.uniform(
            pmin, pmax, (n_profiles, n_voters, n_dim)
        )
        
        # Fixed triangle positions for first 3 candidates
        span = pmax - pmin
        candidate_positions = np.zeros((n_profiles, n_candidates, n_dim))
        
        if n_candidates >= 3:
            # Equilateral triangle
            candidate_positions[:, 0] = [pmin + 0.2 * span, pmin + 0.3 * span]  # Bottom left
            candidate_positions[:, 1] = [pmax - 0.2 * span, pmin + 0.3 * span]  # Bottom right
            candidate_positions[:, 2] = [(pmin + pmax) / 2, pmax - 0.2 * span]  # Top center
            
            # Additional candidates random
            for j in range(3, n_candidates):
                candidate_positions[:, j] = self.rng.uniform(
                    pmin, pmax, (n_profiles, n_dim)
                )
        else:
            for j in range(n_candidates):
                candidate_positions[:, j] = self.rng.uniform(
                    pmin, pmax, (n_profiles, n_dim)
                )
        
        return SpatialProfile(
            voter_positions=voter_positions,
            candidate_positions=candidate_positions,
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            n_dim=n_dim,
            method='2d'
        )


def generate_voter_positions(
    n_profiles: int,
    n_voters: int,
    config: GeometryConfig,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Convenience function to generate only voter positions.
    
    Args:
        n_profiles: Number of profiles
        n_voters: Number of voters
        config: Geometry configuration
        rng: Random generator
        
    Returns:
        Voter positions array (n_profiles, n_voters, n_dim)
    """
    generator = GeometryGenerator(config, rng)
    profile = generator.generate(n_profiles, n_voters, 3)
    return profile.voter_positions


def generate_candidate_positions(
    n_profiles: int,
    n_candidates: int,
    config: GeometryConfig,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Convenience function to generate only candidate positions.
    
    Args:
        n_profiles: Number of profiles
        n_candidates: Number of candidates
        config: Geometry configuration
        rng: Random generator
        
    Returns:
        Candidate positions array (n_profiles, n_candidates, n_dim)
    """
    generator = GeometryGenerator(config, rng)
    profile = generator.generate(n_profiles, 10, n_candidates)  # Dummy voter count
    return profile.candidate_positions




