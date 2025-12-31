"""
Test comparing Plurality voting to Random winner selection.

Tests across different dimensions (1, 2, 3, 5, 10) and candidate counts (2, 3, 5, 10)
with 100 Monte Carlo runs each, using configurable distance metric (L1 or L2).

Metrics tracked:
- Average distance to ideal
- Winner extremism
- Worst-off distance
- Rule disagreement frequency (Plurality ≠ Random)
- Condorcet consistency
- Variance in voter satisfaction
- VSE (Voter Satisfaction Efficiency)
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import sys
import os
import time
from dataclasses import dataclass, asdict

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig, ManipulationConfig
)
from simulator.geometry import GeometryGenerator
from simulator.voting_rules import VotingRuleEngine
from simulator.config import VotingRuleConfig
from simulator.metrics import (
    compute_average_distance_to_ideal, compute_winner_extremism,
    compute_worst_off_distance, compute_vse
)

# Import utility functions from expanded_rules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expanded_rules import utilities_to_rankings


@dataclass
class GridMetrics:
    """Metrics for a single grid cell (dimension x candidate count)."""
    avg_distance_plurality: float
    avg_distance_random: float
    extremism_plurality: float
    extremism_random: float
    worst_off_plurality: float
    worst_off_random: float
    disagreement_frequency: float
    condorcet_consistency_plurality: float
    condorcet_consistency_random: float
    variance_plurality: float
    variance_random: float
    vse_plurality: float
    vse_random: float


class PluralityVsRandomTest:
    """Test comparing Plurality vs Random winner."""
    
    def __init__(self, n_runs: int = 100, distance_metric: str = 'l1'):
        """
        Initialize test.
        
        Args:
            n_runs: Number of Monte Carlo runs per configuration
            distance_metric: Distance metric to use ('l1' or 'l2')
        """
        self.n_runs = n_runs
        self.n_voters = 100
        self.dimensions = [1, 2, 3, 5, 10]
        self.candidate_counts = [2, 3, 5, 10]
        self.distance_metric = distance_metric.lower()
        if self.distance_metric not in ['l1', 'l2']:
            raise ValueError(f"distance_metric must be 'l1' or 'l2', got '{distance_metric}'")
        self.results = {}
    
    def run_single_configuration(
        self, 
        n_dim: int, 
        n_candidates: int
    ) -> GridMetrics:
        """
        Run test for a single dimension/candidate configuration.
        
        Args:
            n_dim: Number of dimensions
            n_candidates: Number of candidates
            
        Returns:
            GridMetrics containing all metrics for this configuration
        """
        # Create geometry configuration
        geometry_config = GeometryConfig(
            method='uniform',
            n_dim=n_dim,
            position_min=0.0,
            position_max=1.0
        )
        
        # Generate spatial positions
        rng = np.random.default_rng()
        geometry_gen = GeometryGenerator(geometry_config, rng)
        spatial_profile = geometry_gen.generate(
            self.n_runs,
            self.n_voters,
            n_candidates
        )
        
        # Initialize voting rule engine
        plurality_engine = VotingRuleEngine(
            VotingRuleConfig(), 
            rng=rng
        )
        
        # Storage for metrics across all runs
        plurality_winners = []
        random_winners = []
        
        avg_dist_plur = []
        avg_dist_rand = []
        extremism_plur = []
        extremism_rand = []
        worst_off_plur = []
        worst_off_rand = []
        condorcet_plur = []
        condorcet_rand = []
        variance_plur = []
        variance_rand = []
        vse_plur = []
        vse_rand = []
        disagreements = 0
        
        center = np.full(n_dim, 0.5)  # Center of [0,1]^d space
        
        for i in range(self.n_runs):
            voter_pos = spatial_profile.voter_positions[i]
            candidate_pos = spatial_profile.candidate_positions[i]
            
            # Compute distances using the configured metric
            distances = self._compute_distances(voter_pos, candidate_pos)
            
            # Compute utilities (linear: u = -d, normalized)
            # Max distance depends on metric: L1 = n_dim, L2 = sqrt(n_dim)
            max_dist = n_dim if self.distance_metric == 'l1' else np.sqrt(n_dim)
            utilities = np.maximum(0.0, 1.0 - distances / max_dist)
            
            # Get rankings
            rankings = utilities_to_rankings(utilities)
            
            # Run plurality voting
            plur_result = plurality_engine.plurality(rankings)
            plur_winner = plur_result.winner
            plurality_winners.append(plur_winner)
            
            # Random winner selection
            rand_winner = rng.integers(0, n_candidates)
            random_winners.append(rand_winner)
            
            # Check disagreement
            if plur_winner != rand_winner:
                disagreements += 1
            
            # Compute Condorcet winner for this profile
            condorcet_winner = self._find_condorcet_winner(rankings)
            
            # --- Metrics for Plurality ---
            # Average distance to ideal
            avg_dist_plur.append(
                compute_average_distance_to_ideal(
                    voter_pos, candidate_pos, plur_winner, self.distance_metric
                )
            )
            
            # Winner extremism
            extremism_plur.append(
                compute_winner_extremism(
                    candidate_pos, plur_winner, center, self.distance_metric
                )
            )
            
            # Worst-off distance
            worst_off_plur.append(
                compute_worst_off_distance(
                    voter_pos, candidate_pos, plur_winner, self.distance_metric
                )
            )
            
            # Condorcet consistency
            if condorcet_winner is not None:
                condorcet_plur.append(1.0 if plur_winner == condorcet_winner else 0.0)
            
            # Variance in voter satisfaction (variance of distances)
            distances_plur = self._compute_distances_single(
                voter_pos, candidate_pos[plur_winner:plur_winner+1]
            )
            variance_plur.append(float(np.var(distances_plur)))
            
            # --- Metrics for Random ---
            avg_dist_rand.append(
                compute_average_distance_to_ideal(
                    voter_pos, candidate_pos, rand_winner, self.distance_metric
                )
            )
            
            extremism_rand.append(
                compute_winner_extremism(
                    candidate_pos, rand_winner, center, self.distance_metric
                )
            )
            
            worst_off_rand.append(
                compute_worst_off_distance(
                    voter_pos, candidate_pos, rand_winner, self.distance_metric
                )
            )
            
            if condorcet_winner is not None:
                condorcet_rand.append(1.0 if rand_winner == condorcet_winner else 0.0)
            
            distances_rand = self._compute_distances_single(
                voter_pos, candidate_pos[rand_winner:rand_winner+1]
            )
            variance_rand.append(float(np.var(distances_rand)))
            
            # --- VSE (Voter Satisfaction Efficiency) ---
            # VSE = (Actual - Worst) / (Best - Worst)
            # Using the standard compute_vse function for consistency
            vse_plur.append(compute_vse(utilities, plur_winner))
            vse_rand.append(compute_vse(utilities, rand_winner))
        
        # Aggregate metrics
        return GridMetrics(
            avg_distance_plurality=float(np.mean(avg_dist_plur)),
            avg_distance_random=float(np.mean(avg_dist_rand)),
            extremism_plurality=float(np.mean(extremism_plur)),
            extremism_random=float(np.mean(extremism_rand)),
            worst_off_plurality=float(np.mean(worst_off_plur)),
            worst_off_random=float(np.mean(worst_off_rand)),
            disagreement_frequency=float(disagreements / self.n_runs),
            condorcet_consistency_plurality=float(np.mean(condorcet_plur)) if condorcet_plur else 0.0,
            condorcet_consistency_random=float(np.mean(condorcet_rand)) if condorcet_rand else 0.0,
            variance_plurality=float(np.mean(variance_plur)),
            variance_random=float(np.mean(variance_rand)),
            vse_plurality=float(np.mean(vse_plur)) if vse_plur else 0.0,
            vse_random=float(np.mean(vse_rand)) if vse_rand else 0.0
        )
    
    def _find_condorcet_winner(self, rankings: np.ndarray) -> int | None:
        """
        Find Condorcet winner if one exists.
        
        Args:
            rankings: Voter rankings (n_voters x n_candidates)
            
        Returns:
            Condorcet winner index or None
        """
        n_voters, n_candidates = rankings.shape
        
        # Build pairwise comparison matrix
        for cand_a in range(n_candidates):
            is_condorcet = True
            for cand_b in range(n_candidates):
                if cand_a == cand_b:
                    continue
                
                # Count how many voters prefer cand_a to cand_b
                votes_for_a = 0
                for voter in range(n_voters):
                    rank_a = np.where(rankings[voter] == cand_a)[0][0]
                    rank_b = np.where(rankings[voter] == cand_b)[0][0]
                    if rank_a < rank_b:  # Lower rank = more preferred
                        votes_for_a += 1
                
                # If cand_a doesn't beat cand_b, not a Condorcet winner
                if votes_for_a <= n_voters / 2:
                    is_condorcet = False
                    break
            
            if is_condorcet:
                return cand_a
        
        return None
    
    def _compute_distances(
        self, 
        voter_positions: np.ndarray, 
        candidate_positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between voters and candidates using configured metric.
        
        Args:
            voter_positions: (n_voters, n_dim) array
            candidate_positions: (n_candidates, n_dim) array
            
        Returns:
            (n_voters, n_candidates) distance matrix
        """
        n_voters = voter_positions.shape[0]
        n_candidates = candidate_positions.shape[0]
        distances = np.zeros((n_voters, n_candidates))
        
        if self.distance_metric == 'l1':
            for v in range(n_voters):
                for c in range(n_candidates):
                    distances[v, c] = np.sum(np.abs(voter_positions[v] - candidate_positions[c]))
        else:  # l2
            for v in range(n_voters):
                for c in range(n_candidates):
                    distances[v, c] = np.linalg.norm(voter_positions[v] - candidate_positions[c])
        
        return distances
    
    def _compute_distances_single(
        self,
        voter_positions: np.ndarray,
        candidate_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances from voters to a single candidate.
        
        Args:
            voter_positions: (n_voters, n_dim) array
            candidate_position: (1, n_dim) array
            
        Returns:
            (n_voters,) distance array
        """
        n_voters = voter_positions.shape[0]
        distances = np.zeros(n_voters)
        
        if self.distance_metric == 'l1':
            for v in range(n_voters):
                distances[v] = np.sum(np.abs(voter_positions[v] - candidate_position[0]))
        else:  # l2
            for v in range(n_voters):
                distances[v] = np.linalg.norm(voter_positions[v] - candidate_position[0])
        
        return distances
    
    def run_all_configurations(self) -> Dict:
        """
        Run tests for all dimension/candidate combinations.
        
        Returns:
            Dictionary with results for all configurations
        """
        print("=" * 80)
        print("PLURALITY VS RANDOM WINNER TEST")
        print("=" * 80)
        print(f"Monte Carlo runs: {self.n_runs}")
        print(f"Voters per run: {self.n_voters}")
        metric_name = "L1 (Manhattan)" if self.distance_metric == 'l1' else "L2 (Euclidean)"
        print(f"Distance metric: {metric_name}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Candidate counts: {self.candidate_counts}")
        print("=" * 80)
        print()
        
        # Run all configurations
        for n_dim in self.dimensions:
            self.results[n_dim] = {}
            for n_candidates in self.candidate_counts:
                print(f"Running: D={n_dim}, C={n_candidates}...", end=" ", flush=True)
                start_time = time.perf_counter()
                
                metrics = self.run_single_configuration(n_dim, n_candidates)
                self.results[n_dim][n_candidates] = asdict(metrics)
                
                elapsed = time.perf_counter() - start_time
                print(f"Done ({elapsed:.2f}s)")
        
        print()
        return self.results
    
    def display_grid(self, metric_name: str, rule: str):
        """
        Display a grid for a specific metric and rule.
        
        Args:
            metric_name: Name of the metric to display
            rule: 'plurality' or 'random'
        """
        print(f"\n{metric_name.upper()} - {rule.upper()}")
        print("=" * 80)
        
        # Header
        print(f"{'Dim':<6}", end="")
        for n_cand in self.candidate_counts:
            print(f"{n_cand:>12}", end="")
        print()
        print("-" * 80)
        
        # Rows
        for n_dim in self.dimensions:
            print(f"{n_dim:<6}", end="")
            for n_cand in self.candidate_counts:
                if n_dim in self.results and n_cand in self.results[n_dim]:
                    value = self.results[n_dim][n_cand][f"{metric_name}_{rule}"]
                    print(f"{value:>12.4f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()
        print()
    
    def display_all_grids(self):
        """Display all metric grids."""
        metrics = [
            'avg_distance',
            'extremism',
            'worst_off',
            'variance',
            'vse'
        ]
        
        for metric in metrics:
            self.display_grid(metric, 'plurality')
            self.display_grid(metric, 'random')
        
        # Display disagreement frequency (not rule-specific)
        print("\nRULE DISAGREEMENT FREQUENCY (Plurality != Random)")
        print("=" * 80)
        print(f"{'Dim':<6}", end="")
        for n_cand in self.candidate_counts:
            print(f"{n_cand:>12}", end="")
        print()
        print("-" * 80)
        
        for n_dim in self.dimensions:
            print(f"{n_dim:<6}", end="")
            for n_cand in self.candidate_counts:
                if n_dim in self.results and n_cand in self.results[n_dim]:
                    value = self.results[n_dim][n_cand]['disagreement_frequency']
                    print(f"{value:>12.4f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()
        print()
        
        # Display Condorcet consistency
        print("\nCONDORCET CONSISTENCY (Fraction picking Condorcet winner)")
        print("=" * 80)
        for rule in ['plurality', 'random']:
            print(f"\n{rule.upper()}")
            print("-" * 80)
            print(f"{'Dim':<6}", end="")
            for n_cand in self.candidate_counts:
                print(f"{n_cand:>12}", end="")
            print()
            print("-" * 80)
            
            for n_dim in self.dimensions:
                print(f"{n_dim:<6}", end="")
                for n_cand in self.candidate_counts:
                    if n_dim in self.results and n_cand in self.results[n_dim]:
                        value = self.results[n_dim][n_cand][f'condorcet_consistency_{rule}']
                        print(f"{value:>12.4f}", end="")
                    else:
                        print(f"{'N/A':>12}", end="")
                print()
        print()
    
    def save_results(self, filename: str = "results_plur-rand.json"):
        """
        Save results to JSON file.
        
        Args:
            filename: Name of output file
        """
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Convert to serializable format
        output = {
            'test_name': 'Plurality vs Random Winner',
            'n_runs': self.n_runs,
            'n_voters': self.n_voters,
            'dimensions': self.dimensions,
            'candidate_counts': self.candidate_counts,
            'distance_metric': self.distance_metric,
            'results': {
                str(dim): {
                    str(cand): metrics
                    for cand, metrics in cand_dict.items()
                }
                for dim, cand_dict in self.results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {filepath}")


def main():
    """Main entry point."""
    test = PluralityVsRandomTest(n_runs=100, distance_metric='l1')
    
    # Run all configurations
    test.run_all_configurations()
    
    # Display results
    test.display_all_grids()
    
    # Save to file
    test.save_results()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

