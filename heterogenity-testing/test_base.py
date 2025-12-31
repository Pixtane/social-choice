"""
Base test class for heterogeneity experiments.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig, 
    HeterogeneousDistanceConfig, ManipulationConfig
)
from simulator.main import run_experiment
from simulator.metrics import (
    compute_average_distance_to_ideal, compute_winner_extremism,
    compute_worst_off_distance, compute_rule_disagreement
)


@dataclass
class TestResult:
    """Result of a single test run."""
    test_name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    winners: Dict[str, List[int]]  # rule -> list of winners per profile
    compute_time: float


class BaseHeterogeneityTest:
    """Base class for heterogeneity tests."""
    
    def __init__(self, test_name: str, n_runs: int = 100):
        """
        Initialize test.
        
        Args:
            test_name: Name of the test
            n_runs: Number of Monte Carlo runs
        """
        self.test_name = test_name
        self.n_runs = n_runs
        
        # Baseline settings
        self.baseline_n_voters = 100
        self.baseline_n_candidates = 5
        self.baseline_n_dim = 2
        self.baseline_voting_rules = ['plurality', 'borda', 'ranked_pairs']
    
    def get_baseline_config(self) -> SimulationConfig:
        """Get baseline configuration."""
        return SimulationConfig(
            n_profiles=self.n_runs,
            n_voters=self.baseline_n_voters,
            n_candidates=self.baseline_n_candidates,
            voting_rules=self.baseline_voting_rules,
            geometry=GeometryConfig(
                method='uniform',
                n_dim=self.baseline_n_dim,
                position_min=0.0,
                position_max=1.0
            ),
            utility=UtilityConfig(
                function='linear',  # u = -d
                distance_metric='l2'
            ),
            manipulation=ManipulationConfig(enabled=False),
            rng_seed=None
        )
    
    def compute_metrics(
        self,
        experiment_result,
        config: SimulationConfig
    ) -> Dict[str, Any]:
        """
        Compute all metrics for the experiment.
        
        Args:
            experiment_result: Result from run_experiment
            config: Configuration used
            
        Returns:
            Dictionary of metrics
        """
        preferences = experiment_result.preferences
        rule_results = experiment_result.rule_results
        
        n_profiles = preferences.n_profiles
        n_voters = preferences.n_voters
        n_candidates = preferences.n_candidates
        
        # Collect winners for each rule
        winners_by_rule = {}
        for rule_name, result in rule_results.items():
            winners_by_rule[rule_name] = result.winners.tolist()
        
        # Compute per-profile metrics
        avg_distances = {rule: [] for rule in rule_results.keys()}
        winner_extremisms = {rule: [] for rule in rule_results.keys()}
        worst_off_distances = {rule: [] for rule in rule_results.keys()}
        condorcet_consistencies = {rule: [] for rule in rule_results.keys()}
        
        # Rule disagreement (per profile)
        rule_disagreements = []
        
        center = np.full(self.baseline_n_dim, 0.5)
        
        for i in range(n_profiles):
            voter_pos = preferences.voter_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            
            # Check rule disagreement
            profile_winners = {rule: winners_by_rule[rule][i] for rule in winners_by_rule}
            disagrees = compute_rule_disagreement(profile_winners)
            rule_disagreements.append(disagrees)
            
            # Compute metrics for each rule
            for rule_name, result in rule_results.items():
                winner_idx = result.winners[i]
                
                # Average distance to ideal
                avg_dist = compute_average_distance_to_ideal(
                    voter_pos, candidate_pos, winner_idx, 'l2'
                )
                avg_distances[rule_name].append(avg_dist)
                
                # Winner extremism
                extremism = compute_winner_extremism(
                    candidate_pos, winner_idx, center, 'l2'
                )
                winner_extremisms[rule_name].append(extremism)
                
                # Worst-off distance
                worst_off = compute_worst_off_distance(
                    voter_pos, candidate_pos, winner_idx, 'l2'
                )
                worst_off_distances[rule_name].append(worst_off)
                
                # Condorcet consistency
                profile_metrics = result.profile_metrics[i]
                condorcet_consistencies[rule_name].append(
                    1.0 if profile_metrics.is_condorcet_winner else 0.0
                )
        
        # Aggregate metrics
        metrics = {
            'average_distance_to_ideal': {
                rule: {
                    'mean': float(np.mean(avg_distances[rule])),
                    'std': float(np.std(avg_distances[rule]))
                }
                for rule in rule_results.keys()
            },
            'winner_extremism': {
                rule: {
                    'mean': float(np.mean(winner_extremisms[rule])),
                    'std': float(np.std(winner_extremisms[rule]))
                }
                for rule in rule_results.keys()
            },
            'worst_off_distance': {
                rule: {
                    'mean': float(np.mean(worst_off_distances[rule])),
                    'std': float(np.std(worst_off_distances[rule]))
                }
                for rule in rule_results.keys()
            },
            'condorcet_consistency': {
                rule: float(np.mean(condorcet_consistencies[rule]))
                for rule in rule_results.keys()
            },
            'rule_disagreement': float(np.mean(rule_disagreements)),
            'rule_disagreement_pct': 100.0 * float(np.mean(rule_disagreements))
        }
        
        return metrics
    
    def run_test(self, config: Optional[SimulationConfig] = None) -> TestResult:
        """
        Run the test.
        
        Args:
            config: Optional custom configuration
            
        Returns:
            TestResult
        """
        if config is None:
            config = self.get_baseline_config()
        
        import time
        start_time = time.perf_counter()
        
        # Run experiment
        experiment_result = run_experiment(
            config, save_results=False, verbose=False
        )
        
        compute_time = time.perf_counter() - start_time
        
        # Compute metrics
        metrics = self.compute_metrics(experiment_result, config)
        
        # Collect winners
        winners = {}
        for rule_name, result in experiment_result.rule_results.items():
            winners[rule_name] = result.winners.tolist()
        
        return TestResult(
            test_name=self.test_name,
            config=config.to_dict(),
            metrics=metrics,
            winners=winners,
            compute_time=compute_time
        )

