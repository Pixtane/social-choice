"""
Metrics computation for voting simulation.

Computes various quality metrics including:
- Voter Satisfaction Efficiency (VSE)
- Condorcet winner detection
- Condorcet cycle detection and classification
- Social utility measures
- Regret measures
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ProfileMetrics:
    """Metrics computed for a single election profile."""
    
    # Winner information
    winner_index: int
    winner_rank: int  # 0=best, 1=second, etc.
    
    # VSE metrics
    vse: float
    raw_social_utility: float
    optimal_utility: float
    worst_utility: float
    
    # Regret
    regret: float  # optimal - actual
    regret_ratio: float  # regret / (optimal - worst)
    
    # Condorcet analysis
    condorcet_winner_exists: bool
    condorcet_winner_index: int  # -1 if none
    is_condorcet_winner: bool  # Did the rule select Condorcet winner?
    
    # Cycle analysis
    has_cycle: bool
    cycle_type: int  # 0=none, 1=type1 (A>B>C>A), 2=type2 (B>A>C>B)


@dataclass 
class AggregateMetrics:
    """Aggregated metrics across multiple profiles."""
    
    n_profiles: int
    
    # VSE statistics
    vse_mean: float
    vse_std: float
    vse_min: float
    vse_max: float
    vse_median: float
    
    # Utility statistics
    utility_mean: float
    utility_std: float
    
    # Regret statistics  
    regret_mean: float
    regret_max: float
    
    # Winner rank distribution
    winner_rank_1st_pct: float
    winner_rank_2nd_pct: float
    winner_rank_3rd_pct: float
    
    # Condorcet statistics
    condorcet_winner_pct: float  # % of profiles with CW
    condorcet_efficiency: float  # % where CW was selected (when exists)
    
    # Cycle statistics
    cycle_percentage: float
    type1_percentage: float
    type2_percentage: float


class MetricsComputer:
    """
    Computes comprehensive voting metrics.
    """
    
    def __init__(self, epsilon: float = 1e-9):
        """
        Initialize metrics computer.
        
        Args:
            epsilon: Small value for numerical comparisons
        """
        self.epsilon = epsilon
    
    def compute_profile_metrics(
        self,
        utilities: np.ndarray,
        rankings: np.ndarray,
        winner_index: int
    ) -> ProfileMetrics:
        """
        Compute metrics for a single election profile.
        
        Args:
            utilities: Utility matrix (n_voters, n_candidates)
            rankings: Rankings matrix (n_voters, n_candidates)
            winner_index: Index of elected winner
            
        Returns:
            ProfileMetrics for this election
        """
        n_voters, n_candidates = utilities.shape
        
        # Social utility computations
        social_utilities = np.mean(utilities, axis=0)
        winner_utility = social_utilities[winner_index]
        optimal_utility = np.max(social_utilities)
        worst_utility = np.min(social_utilities)
        
        # VSE: (actual - worst) / (optimal - worst)
        utility_range = optimal_utility - worst_utility
        if utility_range > self.epsilon:
            vse = (winner_utility - worst_utility) / utility_range
        else:
            vse = 1.0  # All candidates equal
        
        # Regret
        regret = optimal_utility - winner_utility
        if utility_range > self.epsilon:
            regret_ratio = regret / utility_range
        else:
            regret_ratio = 0.0
        
        # Winner rank (position in utility ordering)
        utility_ranking = np.argsort(-social_utilities)
        winner_rank = np.where(utility_ranking == winner_index)[0][0]
        
        # Condorcet analysis
        margins = self._compute_pairwise_margins(rankings)
        condorcet_winner = self._find_condorcet_winner(margins)
        
        condorcet_winner_exists = condorcet_winner >= 0
        is_condorcet_winner = (winner_index == condorcet_winner) if condorcet_winner_exists else True
        
        # Cycle analysis
        has_cycle = self._detect_cycle(margins)
        cycle_type = self._classify_cycle(margins) if has_cycle else 0
        
        return ProfileMetrics(
            winner_index=winner_index,
            winner_rank=winner_rank,
            vse=vse,
            raw_social_utility=winner_utility,
            optimal_utility=optimal_utility,
            worst_utility=worst_utility,
            regret=regret,
            regret_ratio=regret_ratio,
            condorcet_winner_exists=condorcet_winner_exists,
            condorcet_winner_index=condorcet_winner,
            is_condorcet_winner=is_condorcet_winner,
            has_cycle=has_cycle,
            cycle_type=cycle_type,
        )
    
    def aggregate_metrics(
        self,
        profile_metrics: List[ProfileMetrics],
        n_candidates: int = 3
    ) -> AggregateMetrics:
        """
        Aggregate metrics across multiple profiles.
        
        Args:
            profile_metrics: List of ProfileMetrics
            n_candidates: Number of candidates
            
        Returns:
            AggregateMetrics with summary statistics
        """
        n_profiles = len(profile_metrics)
        
        if n_profiles == 0:
            return self._empty_aggregate_metrics()
        
        # Extract arrays
        vse_values = np.array([m.vse for m in profile_metrics])
        utility_values = np.array([m.raw_social_utility for m in profile_metrics])
        regret_values = np.array([m.regret for m in profile_metrics])
        winner_ranks = np.array([m.winner_rank for m in profile_metrics])
        
        # Winner rank distribution
        rank_counts = np.zeros(n_candidates)
        for rank in winner_ranks:
            if rank < n_candidates:
                rank_counts[rank] += 1
        rank_pcts = 100 * rank_counts / n_profiles
        
        # Condorcet statistics
        cw_exists = sum(1 for m in profile_metrics if m.condorcet_winner_exists)
        cw_selected = sum(1 for m in profile_metrics 
                         if m.condorcet_winner_exists and m.is_condorcet_winner)
        
        condorcet_winner_pct = 100 * cw_exists / n_profiles
        condorcet_efficiency = 100 * cw_selected / cw_exists if cw_exists > 0 else 100.0
        
        # Cycle statistics
        n_cycles = sum(1 for m in profile_metrics if m.has_cycle)
        n_type1 = sum(1 for m in profile_metrics if m.cycle_type == 1)
        n_type2 = sum(1 for m in profile_metrics if m.cycle_type == 2)
        
        return AggregateMetrics(
            n_profiles=n_profiles,
            vse_mean=float(np.mean(vse_values)),
            vse_std=float(np.std(vse_values)),
            vse_min=float(np.min(vse_values)),
            vse_max=float(np.max(vse_values)),
            vse_median=float(np.median(vse_values)),
            utility_mean=float(np.mean(utility_values)),
            utility_std=float(np.std(utility_values)),
            regret_mean=float(np.mean(regret_values)),
            regret_max=float(np.max(regret_values)),
            winner_rank_1st_pct=float(rank_pcts[0]) if len(rank_pcts) > 0 else 0.0,
            winner_rank_2nd_pct=float(rank_pcts[1]) if len(rank_pcts) > 1 else 0.0,
            winner_rank_3rd_pct=float(rank_pcts[2]) if len(rank_pcts) > 2 else 0.0,
            condorcet_winner_pct=float(condorcet_winner_pct),
            condorcet_efficiency=float(condorcet_efficiency),
            cycle_percentage=100 * n_cycles / n_profiles,
            type1_percentage=100 * n_type1 / n_profiles,
            type2_percentage=100 * n_type2 / n_profiles,
        )
    
    def _empty_aggregate_metrics(self) -> AggregateMetrics:
        """Return empty aggregate metrics."""
        return AggregateMetrics(
            n_profiles=0,
            vse_mean=0.0, vse_std=0.0, vse_min=0.0, vse_max=0.0, vse_median=0.0,
            utility_mean=0.0, utility_std=0.0,
            regret_mean=0.0, regret_max=0.0,
            winner_rank_1st_pct=0.0, winner_rank_2nd_pct=0.0, winner_rank_3rd_pct=0.0,
            condorcet_winner_pct=0.0, condorcet_efficiency=0.0,
            cycle_percentage=0.0, type1_percentage=0.0, type2_percentage=0.0,
        )
    
    def _compute_pairwise_margins(self, rankings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise margin matrix.
        
        margins[i, j] = voters preferring i to j minus voters preferring j to i
        """
        n_voters, n_candidates = rankings.shape
        margins = np.zeros((n_candidates, n_candidates))
        
        for v in range(n_voters):
            for i in range(n_candidates):
                for j in range(i + 1, n_candidates):
                    c1, c2 = rankings[v, i], rankings[v, j]
                    margins[c1, c2] += 1
                    margins[c2, c1] -= 1
        
        return margins
    
    def _find_condorcet_winner(self, margins: np.ndarray) -> int:
        """
        Find Condorcet winner from margin matrix.
        
        Returns:
            Index of Condorcet winner, or -1 if none
        """
        n_candidates = margins.shape[0]
        
        for c in range(n_candidates):
            # Check if c beats all others
            beats_all = True
            for other in range(n_candidates):
                if other != c and margins[c, other] <= 0:
                    beats_all = False
                    break
            if beats_all:
                return c
        
        return -1
    
    def _detect_cycle(self, margins: np.ndarray) -> bool:
        """
        Detect if there's a Condorcet cycle.
        
        Returns:
            True if cycle exists
        """
        # For 3 candidates, cycle exists iff no Condorcet winner
        if margins.shape[0] == 3:
            return self._find_condorcet_winner(margins) < 0
        
        # General case: check if there's a cycle using DFS
        n = margins.shape[0]
        
        # Build graph of strict preferences
        graph = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and margins[i, j] > 0:
                    graph[i].add(j)
        
        # DFS to find cycle
        def has_cycle_from(start: int) -> bool:
            visited = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    return True
                visited.add(node)
                
                for neighbor in graph[node]:
                    if neighbor == start:
                        return True
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            return False
        
        return any(has_cycle_from(i) for i in range(n))
    
    def _classify_cycle(self, margins: np.ndarray) -> int:
        """
        Classify cycle type for 3 candidates.
        
        Type 1: A > B > C > A (clockwise)
        Type 2: B > A > C > B (counterclockwise)
        
        Returns:
            Cycle type (1 or 2), or 0 if no cycle or more than 3 candidates
        """
        if margins.shape[0] != 3:
            return 0
        
        # A=0, B=1, C=2
        # Type 1: A>B, B>C, C>A
        if margins[0, 1] > 0 and margins[1, 2] > 0 and margins[2, 0] > 0:
            return 1
        
        # Type 2: B>A, A>C, C>B  (equivalent to A>C, C>B, B>A)
        if margins[1, 0] > 0 and margins[0, 2] > 0 and margins[2, 1] > 0:
            return 2
        
        return 0


# Convenience functions

def compute_vse(
    utilities: np.ndarray,
    winner_index: int,
    epsilon: float = 1e-9
) -> float:
    """
    Compute Voter Satisfaction Efficiency for a single election.
    
    VSE = (actual_utility - worst_utility) / (best_utility - worst_utility)
    
    Args:
        utilities: Utility matrix (n_voters, n_candidates)
        winner_index: Index of elected winner
        epsilon: Epsilon for numerical stability
        
    Returns:
        VSE value in [0, 1]
    """
    social_utilities = np.mean(utilities, axis=0)
    winner_utility = social_utilities[winner_index]
    best_utility = np.max(social_utilities)
    worst_utility = np.min(social_utilities)
    
    utility_range = best_utility - worst_utility
    if utility_range < epsilon:
        return 1.0
    
    return (winner_utility - worst_utility) / utility_range


def compute_regret(
    utilities: np.ndarray,
    winner_index: int
) -> float:
    """
    Compute regret (difference from optimal).
    
    Args:
        utilities: Utility matrix (n_voters, n_candidates)
        winner_index: Index of elected winner
        
    Returns:
        Regret value >= 0
    """
    social_utilities = np.mean(utilities, axis=0)
    return np.max(social_utilities) - social_utilities[winner_index]


def find_condorcet_winner(rankings: np.ndarray) -> int:
    """
    Find Condorcet winner from rankings.
    
    Args:
        rankings: Rankings matrix (n_voters, n_candidates)
        
    Returns:
        Index of Condorcet winner, or -1 if none
    """
    computer = MetricsComputer()
    margins = computer._compute_pairwise_margins(rankings)
    return computer._find_condorcet_winner(margins)


def has_condorcet_cycle(rankings: np.ndarray) -> bool:
    """
    Check if election has a Condorcet cycle.
    
    Args:
        rankings: Rankings matrix (n_voters, n_candidates)
        
    Returns:
        True if cycle exists
    """
    computer = MetricsComputer()
    margins = computer._compute_pairwise_margins(rankings)
    return computer._detect_cycle(margins)


def get_pairwise_matrix(rankings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise comparison matrix.
    
    Args:
        rankings: Rankings matrix (n_voters, n_candidates)
        
    Returns:
        Margin matrix where [i,j] is net voters preferring i to j
    """
    computer = MetricsComputer()
    return computer._compute_pairwise_margins(rankings)


def compute_social_utility(
    utilities: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Compute social utility for each candidate.
    
    Args:
        utilities: Utility matrix (n_voters, n_candidates)
        method: 'mean', 'sum', 'median', 'min', 'max'
        
    Returns:
        Social utility array (n_candidates,)
    """
    if method == 'mean':
        return np.mean(utilities, axis=0)
    elif method == 'sum':
        return np.sum(utilities, axis=0)
    elif method == 'median':
        return np.median(utilities, axis=0)
    elif method == 'min':
        return np.min(utilities, axis=0)
    elif method == 'max':
        return np.max(utilities, axis=0)
    else:
        raise ValueError(f"Unknown social utility method: {method}")


