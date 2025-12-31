"""
Complete test runner for all 15 heterogeneity tests.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
import sys
import os
import time
from dataclasses import dataclass, asdict

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig, ManipulationConfig
)
from simulator.main import generate_preferences, run_simulation
from simulator.utility import UtilityComputer, utilities_to_rankings
from simulator.manipulation import ManipulationEngine
from simulator.metrics import MetricsComputer
# Import custom distance functions
import importlib.util
custom_dist_path = os.path.join(os.path.dirname(__file__), 'custom_distance.py')
spec = importlib.util.spec_from_file_location("custom_distance", custom_dist_path)
custom_distance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_distance)

assign_metrics_by_fraction = custom_distance.assign_metrics_by_fraction
assign_metrics_by_extremity = custom_distance.assign_metrics_by_extremity
assign_metrics_by_radius = custom_distance.assign_metrics_by_radius
assign_metrics_randomly = custom_distance.assign_metrics_randomly
compute_distances_with_assignment = custom_distance.compute_distances_with_assignment
from simulator.metrics import (
    compute_average_distance_to_ideal, compute_winner_extremism,
    compute_worst_off_distance, compute_rule_disagreement
)


@dataclass
class TestResult:
    """Result of a test."""
    test_name: str
    metrics: Dict
    winners: Dict[str, List[int]]
    compute_time: float
    config_summary: Dict


class HeterogeneityTestRunner:
    """Runner for heterogeneity tests."""
    
    def __init__(self, n_runs: int = 100):
        self.n_runs = n_runs
        self.results = {}
        self.baseline_n_voters = 100
        self.baseline_n_candidates = 5
        self.baseline_n_dim = 2
        self.baseline_rules = ['plurality', 'borda', 'ranked_pairs']
    
    def get_baseline_config(self) -> SimulationConfig:
        """Get baseline configuration."""
        return SimulationConfig(
            n_profiles=self.n_runs,
            n_voters=self.baseline_n_voters,
            n_candidates=self.baseline_n_candidates,
            voting_rules=self.baseline_rules,
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
    
    def run_with_custom_distances(
        self,
        config: SimulationConfig,
        distance_assignments: List[np.ndarray]  # One per profile
    ) -> TestResult:
        """Run experiment with custom distance assignments per profile."""
        start_time = time.perf_counter()
        
        # Generate spatial positions
        preferences = generate_preferences(config)
        
        # Override utilities and rankings with custom distances
        n_profiles = preferences.n_profiles
        all_utilities = []
        all_rankings = []
        
        utility_computer = UtilityComputer(config.utility)
        
        for i in range(n_profiles):
            voter_pos = preferences.voter_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            metric_assignment = distance_assignments[i]
            
            # Compute distances with custom assignment
            distances = compute_distances_with_assignment(
                voter_pos, candidate_pos, metric_assignment
            )
            
            # Compute utilities
            utilities = utility_computer.compute_utilities(
                distances, config.geometry.n_dim
            )
            all_utilities.append(utilities)
            
            # Compute rankings
            rankings = utilities_to_rankings(utilities, config.epsilon)
            all_rankings.append(rankings)
        
        preferences.utilities = np.array(all_utilities)
        preferences.rankings = np.array(all_rankings)
        
        # Run simulations for each rule
        rule_results = {}
        for rule_name in config.voting_rules:
            result = run_simulation(preferences, rule_name, config)
            rule_results[rule_name] = result
        
        # Compute metrics
        metrics = self._compute_all_metrics(preferences, rule_results, config)
        
        # Collect winners
        winners = {rule: result.winners.tolist() 
                  for rule, result in rule_results.items()}
        
        compute_time = time.perf_counter() - start_time
        
        return TestResult(
            test_name="",
            metrics=metrics,
            winners=winners,
            compute_time=compute_time,
            config_summary=config.to_dict()
        )
    
    def _compute_all_metrics(
        self,
        preferences,
        rule_results: Dict,
        config: SimulationConfig
    ) -> Dict:
        """Compute all metrics."""
        n_profiles = preferences.n_profiles
        center = np.full(config.geometry.n_dim, 0.5)
        
        # Per-rule metrics
        avg_distances = {rule: [] for rule in rule_results.keys()}
        extremisms = {rule: [] for rule in rule_results.keys()}
        worst_offs = {rule: [] for rule in rule_results.keys()}
        condorcet_consistencies = {rule: [] for rule in rule_results.keys()}
        rule_disagreements = []
        
        for i in range(n_profiles):
            voter_pos = preferences.voter_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            
            # Check rule disagreement
            profile_winners = {rule: rule_results[rule].winners[i] 
                            for rule in rule_results.keys()}
            disagrees = compute_rule_disagreement(profile_winners)
            rule_disagreements.append(disagrees)
            
            # Per-rule metrics
            for rule_name, result in rule_results.items():
                winner_idx = result.winners[i]
                
                avg_dist = compute_average_distance_to_ideal(
                    voter_pos, candidate_pos, winner_idx, 'l2'
                )
                avg_distances[rule_name].append(avg_dist)
                
                extremism = compute_winner_extremism(
                    candidate_pos, winner_idx, center, 'l2'
                )
                extremisms[rule_name].append(extremism)
                
                worst_off = compute_worst_off_distance(
                    voter_pos, candidate_pos, winner_idx, 'l2'
                )
                worst_offs[rule_name].append(worst_off)
                
                # Condorcet consistency
                profile_metrics = result.profile_metrics[i]
                condorcet_consistencies[rule_name].append(
                    1.0 if profile_metrics.is_condorcet_winner else 0.0
                )
        
        return {
            'average_distance_to_ideal': {
                rule: {'mean': float(np.mean(avg_distances[rule])),
                      'std': float(np.std(avg_distances[rule]))}
                for rule in rule_results.keys()
            },
            'winner_extremism': {
                rule: {'mean': float(np.mean(extremisms[rule])),
                      'std': float(np.std(extremisms[rule]))}
                for rule in rule_results.keys()
            },
            'worst_off_distance': {
                rule: {'mean': float(np.mean(worst_offs[rule])),
                      'std': float(np.std(worst_offs[rule]))}
                for rule in rule_results.keys()
            },
            'condorcet_consistency': {
                rule: float(np.mean(condorcet_consistencies[rule]))
                for rule in rule_results.keys()
            },
            'rule_disagreement': float(np.mean(rule_disagreements)),
            'rule_disagreement_pct': 100.0 * float(np.mean(rule_disagreements))
        }
    
    # =========================================================================
    # Test Implementations
    # =========================================================================
    
    def test_1_l2_cosine_fraction(self) -> TestResult:
        """Test 1: Vary L2/Cosine fraction."""
        print("Test 1: Varying L2/Cosine fraction...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        # Sample fractions: 0%, 25%, 50%, 75%, 100%
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        all_metrics = []
        
        for frac in fractions:
            # Assign metrics per profile
            distance_assignments = []
            for _ in range(self.n_runs):
                assignment = assign_metrics_by_fraction(
                    self.baseline_n_voters, frac, 'l2', 'cosine', rng
                )
                distance_assignments.append(assignment)
            
            result = self.run_with_custom_distances(config, distance_assignments)
            all_metrics.append((frac, result.metrics))
        
        # Use 50/50 result as representative
        _, result = all_metrics[len(all_metrics) // 2]
        return TestResult(
            test_name="test_1_l2_cosine_fraction",
            metrics=result,
            winners={},
            compute_time=0.0,
            config_summary={'l2_fraction': 0.5}
        )
    
    def test_2_extreme_voters_linf(self) -> TestResult:
        """Test 2: Extreme voters use L∞."""
        print("Test 2: Extreme voters use L∞...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        n_extreme = int(0.10 * self.baseline_n_voters)  # 10% most extreme
        
        distance_assignments = []
        for _ in range(self.n_runs):
            # Generate positions first to determine extremes
            from simulator.geometry import GeometryGenerator
            geom_gen = GeometryGenerator(config.geometry, rng)
            spatial = geom_gen.generate(1, self.baseline_n_voters, self.baseline_n_candidates)
            voter_pos = spatial.voter_positions[0]
            
            assignment = assign_metrics_by_extremity(
                voter_pos, n_extreme, 'chebyshev', 'l2'
            )
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_2_extreme_voters_linf"
        return result
    
    def test_3_radius_based_distance(self) -> TestResult:
        """Test 3: Distance rule depends on radius."""
        print("Test 3: Radius-based distance assignment...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        # r < 0.3 → L2, 0.3 ≤ r < 0.6 → L1, r ≥ 0.6 → L∞
        boundaries = [0.3, 0.6]
        metrics = ['l2', 'l1', 'chebyshev']
        
        distance_assignments = []
        for _ in range(self.n_runs):
            from simulator.geometry import GeometryGenerator
            geom_gen = GeometryGenerator(config.geometry, rng)
            spatial = geom_gen.generate(1, self.baseline_n_voters, self.baseline_n_candidates)
            voter_pos = spatial.voter_positions[0]
            
            assignment = assign_metrics_by_radius(
                voter_pos, boundaries, metrics
            )
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_3_radius_based_distance"
        return result
    
    def test_4_random_distance(self) -> TestResult:
        """Test 4: Random distance function per voter."""
        print("Test 4: Random distance assignment...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        metrics = ['l1', 'l2', 'chebyshev', 'cosine']
        distance_assignments = []
        
        for _ in range(self.n_runs):
            assignment = assign_metrics_randomly(
                self.baseline_n_voters, metrics, rng=rng
            )
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_4_random_distance"
        return result
    
    def test_5_utility_nonlinearity(self) -> TestResult:
        """Test 5: Utility nonlinearity."""
        print("Test 5: Utility nonlinearity...")
        
        config = self.get_baseline_config()
        
        # Test linear, quadratic, saturated
        utility_funcs = ['linear', 'quadratic', 'saturated']
        all_results = []
        
        for func in utility_funcs:
            config.utility.function = func
            if func == 'saturated':
                config.utility.saturation_threshold = 0.5
            
            # Run with standard L2
            preferences = generate_preferences(config)
            rule_results = {}
            for rule_name in config.voting_rules:
                result = run_simulation(preferences, rule_name, config)
                rule_results[rule_name] = result
            
            metrics = self._compute_all_metrics(preferences, rule_results, config)
            all_results.append((func, metrics))
        
        # Return quadratic result
        _, result_metrics = all_results[1]
        return TestResult(
            test_name="test_5_utility_nonlinearity",
            metrics=result_metrics,
            winners={},
            compute_time=0.0,
            config_summary={'utility_functions': utility_funcs}
        )
    
    def test_6_strategic_misreporting(self) -> TestResult:
        """Test 6: Strategic misreporting (truncate to top 2)."""
        print("Test 6: Strategic misreporting...")
        
        config = self.get_baseline_config()
        config.manipulation.enabled = True
        config.manipulation.manipulator_fraction = 0.20
        config.manipulation.strategy = 'compromise'  # Closest to truncation
        
        preferences = generate_preferences(config)
        rule_results = {}
        
        for rule_name in config.voting_rules:
            result = run_simulation(preferences, rule_name, config)
            rule_results[rule_name] = result
        
        metrics = self._compute_all_metrics(preferences, rule_results, config)
        
        return TestResult(
            test_name="test_6_strategic_misreporting",
            metrics=metrics,
            winners={rule: result.winners.tolist() for rule, result in rule_results.items()},
            compute_time=0.0,
            config_summary={'manipulation_fraction': 0.20}
        )
    
    def test_7_candidate_clustering(self) -> TestResult:
        """Test 7: Candidate clustering in one quadrant."""
        print("Test 7: Candidate clustering...")
        
        config = self.get_baseline_config()
        
        # Modify geometry to cluster candidates
        class ClusteredGeometryConfig(GeometryConfig):
            def __init__(self):
                super().__init__(method='uniform', n_dim=2)
        
        # We'll need to manually set candidate positions
        # For now, use custom generation
        rng = np.random.default_rng(42)
        
        # Generate preferences with clustered candidates
        preferences = generate_preferences(config)
        
        # Override candidate positions to cluster in [0.7, 1.0]^2
        for i in range(self.n_runs):
            candidates = rng.uniform(0.7, 1.0, (self.baseline_n_candidates, 2))
            preferences.candidate_positions[i] = candidates
        
        # Use heterogeneous L2/L∞ mixture
        distance_assignments = []
        for _ in range(self.n_runs):
            assignment = assign_metrics_by_fraction(
                self.baseline_n_voters, 0.5, 'l2', 'chebyshev', rng
            )
            distance_assignments.append(assignment)
        
        # Recompute utilities with new positions and distances
        utility_computer = UtilityComputer(config.utility)
        all_utilities = []
        all_rankings = []
        
        for i in range(self.n_runs):
            voter_pos = preferences.voter_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            metric_assignment = distance_assignments[i]
            
            distances = compute_distances_with_assignment(
                voter_pos, candidate_pos, metric_assignment
            )
            utilities = utility_computer.compute_utilities(distances, 2)
            all_utilities.append(utilities)
            rankings = utilities_to_rankings(utilities, config.epsilon)
            all_rankings.append(rankings)
        
        preferences.utilities = np.array(all_utilities)
        preferences.rankings = np.array(all_rankings)
        
        rule_results = {}
        for rule_name in config.voting_rules:
            result = run_simulation(preferences, rule_name, config)
            rule_results[rule_name] = result
        
        metrics = self._compute_all_metrics(preferences, rule_results, config)
        
        return TestResult(
            test_name="test_7_candidate_clustering",
            metrics=metrics,
            winners={rule: result.winners.tolist() for rule, result in rule_results.items()},
            compute_time=0.0,
            config_summary={'candidate_cluster': [0.7, 1.0]}
        )
    
    def test_8_thresholds(self) -> TestResult:
        """Test 8: Acceptability thresholds."""
        print("Test 8: Acceptability thresholds...")
        
        # Note: Thresholds require filtering candidates
        # For now, we'll approximate by using approval voting with threshold
        config = self.get_baseline_config()
        config.voting_rules = ['approval']  # Use approval for threshold effect
        config.voting_rule_config.approval_policy = 'threshold'
        config.voting_rule_config.approval_threshold = -0.6  # Since u = -d, threshold on utility
        
        # L2/Cosine 50/50
        rng = np.random.default_rng(42)
        distance_assignments = []
        for _ in range(self.n_runs):
            assignment = assign_metrics_by_fraction(
                self.baseline_n_voters, 0.5, 'l2', 'cosine', rng
            )
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_8_thresholds"
        return result
    
    def test_9_dimensionality_sweep(self) -> TestResult:
        """Test 9: Dimensionality sweep."""
        print("Test 9: Dimensionality sweep...")
        
        dims = [1, 2, 3, 5]
        all_results = []
        
        for d in dims:
            config = self.get_baseline_config()
            config.geometry.n_dim = d
            
            # L2/Cosine 50/50
            rng = np.random.default_rng(42)
            distance_assignments = []
            for _ in range(self.n_runs):
                assignment = assign_metrics_by_fraction(
                    self.baseline_n_voters, 0.5, 'l2', 'cosine', rng
                )
                distance_assignments.append(assignment)
            
            result = self.run_with_custom_distances(config, distance_assignments)
            all_results.append((d, result.metrics))
        
        # Return 2D result
        _, result_metrics = all_results[1]
        return TestResult(
            test_name="test_9_dimensionality_sweep",
            metrics=result_metrics,
            winners={},
            compute_time=0.0,
            config_summary={'dimensions': dims}
        )
    
    def test_10_candidate_count_sweep(self) -> TestResult:
        """Test 10: Candidate count sweep."""
        print("Test 10: Candidate count sweep...")
        
        candidate_counts = [3, 5, 10]
        all_results = []
        
        for m in candidate_counts:
            config = self.get_baseline_config()
            config.n_candidates = m
            
            # L2/Cosine 50/50
            rng = np.random.default_rng(42)
            distance_assignments = []
            for _ in range(self.n_runs):
                assignment = assign_metrics_by_fraction(
                    self.baseline_n_voters, 0.5, 'l2', 'cosine', rng
                )
                distance_assignments.append(assignment)
            
            result = self.run_with_custom_distances(config, distance_assignments)
            all_results.append((m, result.metrics))
        
        # Return M=5 result
        _, result_metrics = all_results[1] if len(all_results) > 1 else all_results[0]
        return TestResult(
            test_name="test_10_candidate_count_sweep",
            metrics=result_metrics,
            winners={},
            compute_time=0.0,
            config_summary={'candidate_counts': candidate_counts}
        )
    
    def test_11_outlier_voters(self) -> TestResult:
        """Test 11: Outlier voters."""
        print("Test 11: Outlier voters...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        # Generate preferences
        preferences = generate_preferences(config)
        
        # Place 5% of voters far outside [0,1]^2
        n_outliers = int(0.05 * self.baseline_n_voters)
        
        for i in range(self.n_runs):
            # Select random voters to be outliers
            outlier_indices = rng.choice(
                self.baseline_n_voters, n_outliers, replace=False
            )
            # Place them at [1.5, 2.0] range
            for idx in outlier_indices:
                preferences.voter_positions[i, idx] = rng.uniform(1.5, 2.0, 2)
        
        # Assign L∞ to outliers, L2 to rest
        distance_assignments = []
        for i in range(self.n_runs):
            assignment = np.full(self.baseline_n_voters, 'l2', dtype=object)
            # Find outliers (voters outside [0,1]^2)
            voter_pos = preferences.voter_positions[i]
            outliers = np.any((voter_pos < 0) | (voter_pos > 1), axis=1)
            assignment[outliers] = 'chebyshev'
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_11_outlier_voters"
        return result
    
    def test_12_noise_perception(self) -> TestResult:
        """Test 12: Noise in voter perception."""
        print("Test 12: Noise in voter perception...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        # Generate preferences
        preferences = generate_preferences(config)
        
        # Add Gaussian noise N(0, 0.05) to voter positions
        noise = rng.normal(0, 0.05, preferences.voter_positions.shape)
        noisy_positions = preferences.voter_positions + noise
        
        # Use noisy positions for distance computation
        utility_computer = UtilityComputer(config.utility)
        all_utilities = []
        all_rankings = []
        
        for i in range(self.n_runs):
            voter_pos = noisy_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            
            # Compute distances with L2
            metric_assignment = np.full(self.baseline_n_voters, 'l2', dtype=object)
            distances = compute_distances_with_assignment(
                voter_pos, candidate_pos, metric_assignment
            )
            
            utilities = utility_computer.compute_utilities(distances, 2)
            all_utilities.append(utilities)
            rankings = utilities_to_rankings(utilities, config.epsilon)
            all_rankings.append(rankings)
        
        preferences.utilities = np.array(all_utilities)
        preferences.rankings = np.array(all_rankings)
        
        rule_results = {}
        for rule_name in config.voting_rules:
            result = run_simulation(preferences, rule_name, config)
            rule_results[rule_name] = result
        
        metrics = self._compute_all_metrics(preferences, rule_results, config)
        
        return TestResult(
            test_name="test_12_noise_perception",
            metrics=metrics,
            winners={rule: result.winners.tolist() for rule, result in rule_results.items()},
            compute_time=0.0,
            config_summary={'noise_std': 0.05}
        )
    
    def test_13_hybrid_distance_switching(self) -> TestResult:
        """Test 13: Hybrid distance switching by candidate location."""
        print("Test 13: Hybrid distance by candidate location...")
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        # This is complex - distance depends on candidate location
        # For each voter-candidate pair, use L2 if candidate in center, cosine if on edge
        # We'll approximate by computing distances per candidate group
        
        preferences = generate_preferences(config)
        center = np.array([0.5, 0.5])
        
        utility_computer = UtilityComputer(config.utility)
        all_utilities = []
        all_rankings = []
        
        for i in range(self.n_runs):
            voter_pos = preferences.voter_positions[i]
            candidate_pos = preferences.candidate_positions[i]
            
            # Classify candidates: center vs edge
            candidate_distances = np.linalg.norm(candidate_pos - center, axis=1)
            center_threshold = 0.3
            candidate_in_center = candidate_distances < center_threshold
            
            # For each voter, compute distances using appropriate metric
            n_voters = voter_pos.shape[0]
            n_candidates = candidate_pos.shape[0]
            distances = np.zeros((n_voters, n_candidates))
            
            for v in range(n_voters):
                for c in range(n_candidates):
                    if candidate_in_center[c]:
                        # Use L2
                        from simulator.heterogeneous_distance import compute_l2_distance
                        dist = compute_l2_distance(
                            voter_pos[v:v+1], candidate_pos[c:c+1]
                        )
                        distances[v, c] = dist[0, 0]
                    else:
                        # Use cosine
                        from simulator.heterogeneous_distance import compute_cosine_distance
                        dist = compute_cosine_distance(
                            voter_pos[v:v+1], candidate_pos[c:c+1]
                        )
                        distances[v, c] = dist[0, 0]
            
            utilities = utility_computer.compute_utilities(distances, 2)
            all_utilities.append(utilities)
            rankings = utilities_to_rankings(utilities, config.epsilon)
            all_rankings.append(rankings)
        
        preferences.utilities = np.array(all_utilities)
        preferences.rankings = np.array(all_rankings)
        
        rule_results = {}
        for rule_name in config.voting_rules:
            result = run_simulation(preferences, rule_name, config)
            rule_results[rule_name] = result
        
        metrics = self._compute_all_metrics(preferences, rule_results, config)
        
        return TestResult(
            test_name="test_13_hybrid_distance_switching",
            metrics=metrics,
            winners={rule: result.winners.tolist() for rule, result in rule_results.items()},
            compute_time=0.0,
            config_summary={'center_threshold': 0.3}
        )
    
    def test_14_incremental_heterogeneity(self) -> TestResult:
        """Test 14: Incremental heterogeneity sweep."""
        print("Test 14: Incremental heterogeneity sweep...")
        
        # Fraction of non-L2 voters: 0%, 10%, ..., 100%
        fractions = np.linspace(0, 1, 11)
        all_results = []
        
        config = self.get_baseline_config()
        rng = np.random.default_rng(42)
        
        for frac in fractions:
            # Non-L2 voters use randomly chosen L1 or cosine
            distance_assignments = []
            for _ in range(self.n_runs):
                assignment = np.full(self.baseline_n_voters, 'l2', dtype=object)
                n_non_l2 = int(frac * self.baseline_n_voters)
                non_l2_indices = rng.choice(
                    self.baseline_n_voters, n_non_l2, replace=False
                )
                # Randomly assign L1 or cosine
                non_l2_metrics = rng.choice(['l1', 'cosine'], n_non_l2)
                assignment[non_l2_indices] = non_l2_metrics
                distance_assignments.append(assignment)
            
            result = self.run_with_custom_distances(config, distance_assignments)
            all_results.append((frac, result.metrics))
        
        # Return 50% result
        _, result_metrics = all_results[len(all_results) // 2]
        return TestResult(
            test_name="test_14_incremental_heterogeneity",
            metrics=result_metrics,
            winners={},
            compute_time=0.0,
            config_summary={'non_l2_fractions': fractions.tolist()}
        )
    
    def test_15_saturated_utility_heterogeneous(self) -> TestResult:
        """Test 15: Saturated utility + heterogeneous distance."""
        print("Test 15: Saturated utility + heterogeneous distance...")
        
        config = self.get_baseline_config()
        config.utility.function = 'saturated'
        config.utility.saturation_threshold = 0.3
        
        # Distance: L2 50%, L∞ 25%, cosine 25%
        rng = np.random.default_rng(42)
        distance_assignments = []
        
        for _ in range(self.n_runs):
            assignment = np.full(self.baseline_n_voters, 'l2', dtype=object)
            # Assign 25% to L∞
            n_linf = int(0.25 * self.baseline_n_voters)
            linf_indices = rng.choice(
                self.baseline_n_voters, n_linf, replace=False
            )
            assignment[linf_indices] = 'chebyshev'
            
            # Assign 25% to cosine
            remaining = np.setdiff1d(
                np.arange(self.baseline_n_voters), linf_indices
            )
            n_cosine = int(0.25 * self.baseline_n_voters)
            cosine_indices = rng.choice(remaining, n_cosine, replace=False)
            assignment[cosine_indices] = 'cosine'
            
            distance_assignments.append(assignment)
        
        result = self.run_with_custom_distances(config, distance_assignments)
        result.test_name = "test_15_saturated_utility_heterogeneous"
        return result
    
    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all 15 tests."""
        print("=" * 70)
        print("HETEROGENEITY TESTING SUITE")
        print("=" * 70)
        print(f"Running {self.n_runs} Monte Carlo runs per test\n")
        
        tests = [
            self.test_1_l2_cosine_fraction,
            self.test_2_extreme_voters_linf,
            self.test_3_radius_based_distance,
            self.test_4_random_distance,
            self.test_5_utility_nonlinearity,
            self.test_6_strategic_misreporting,
            self.test_7_candidate_clustering,
            self.test_8_thresholds,
            self.test_9_dimensionality_sweep,
            self.test_10_candidate_count_sweep,
            self.test_11_outlier_voters,
            self.test_12_noise_perception,
            self.test_13_hybrid_distance_switching,
            self.test_14_incremental_heterogeneity,
            self.test_15_saturated_utility_heterogeneous,
        ]
        
        for i, test_func in enumerate(tests, 1):
            print(f"\n{'='*70}")
            test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
            print(f"TEST {i}: {test_name}")
            print(f"{'='*70}")
            try:
                result = test_func()
                self.results[f"test_{i}"] = result
                print(f"[OK] Completed in {result.compute_time:.2f}s")
            except Exception as e:
                print(f"[FAILED] Failed: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results


if __name__ == "__main__":
    runner = HeterogeneityTestRunner(n_runs=100)
    results = runner.run_all_tests()
    
    # Save results
    import json
    with open('heterogenity-testing/results.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'test_name': result.test_name,
                'metrics': result.metrics,
                'compute_time': result.compute_time,
                'config_summary': result.config_summary
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("All tests completed! Results saved to heterogenity-testing/results.json")
    print(f"{'='*70}")

