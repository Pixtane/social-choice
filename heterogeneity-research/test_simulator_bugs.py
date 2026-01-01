"""
Experimental test script to check for bugs in the simulator.

Tests expected behaviors:
1. VSE should fall with increase in dimensionality (Dawson model expectation)
2. Utility normalization should be correct across dimensions
3. Distance calculations should scale properly
4. Results should match expected patterns
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig
)
from simulator.main import run_experiment
from simulator.utility import UtilityComputer, compute_distances
from simulator.metrics import MetricsComputer


def test_utility_normalization():
    """Test if utilities are correctly normalized across dimensions."""
    print("=" * 80)
    print("TEST 1: Utility Normalization Across Dimensions")
    print("=" * 80)
    
    # Generate test positions
    n_voters = 100
    n_candidates = 5
    dimensions = [1, 2, 3, 5, 10]
    
    rng = np.random.default_rng(42)
    
    results = {}
    
    for n_dim in dimensions:
        print(f"\nTesting {n_dim}D...")
        
        # Generate positions in unit hypercube
        voter_positions = rng.uniform(0, 1, (n_voters, n_dim))
        candidate_positions = rng.uniform(0, 1, (n_candidates, n_dim))
        
        # Test with linear utility
        utility_config = UtilityConfig(
            function='linear',
            distance_metric='l2',
            d_max=None  # Should auto-calculate as sqrt(n_dim)
        )
        
        utility_computer = UtilityComputer(utility_config)
        distances = utility_computer.compute_distances(voter_positions, candidate_positions)
        utilities = utility_computer.compute_utilities(distances, n_dim)
        
        # Check utility range
        min_util = np.min(utilities)
        max_util = np.max(utilities)
        mean_util = np.mean(utilities)
        
        # Check d_max calculation
        expected_d_max = np.sqrt(n_dim)
        actual_d_max = utility_config.d_max if utility_config.d_max is not None else expected_d_max
        
        print(f"  d_max (expected): {expected_d_max:.4f}")
        print(f"  d_max (actual): {actual_d_max:.4f}")
        print(f"  Utility range: [{min_util:.4f}, {max_util:.4f}]")
        print(f"  Mean utility: {mean_util:.4f}")
        
        # Check if utilities are in [0, 1] range
        if min_util < -1e-6 or max_util > 1.0 + 1e-6:
            print(f"  ⚠️  WARNING: Utilities outside [0, 1] range!")
        
        # Check maximum distance
        max_distance = np.max(distances)
        print(f"  Max distance: {max_distance:.4f} (should be <= {expected_d_max:.4f})")
        
        if max_distance > expected_d_max * 1.1:  # Allow 10% tolerance
            print(f"  ⚠️  WARNING: Max distance exceeds expected d_max!")
        
        results[n_dim] = {
            'd_max': actual_d_max,
            'min_util': min_util,
            'max_util': max_util,
            'mean_util': mean_util,
            'max_distance': max_distance
        }
    
    return results


def test_vse_vs_dimensionality():
    """Test if VSE decreases with dimensionality (Dawson model expectation)."""
    print("\n" + "=" * 80)
    print("TEST 2: VSE vs Dimensionality (Dawson Model)")
    print("=" * 80)
    print("Expected: VSE should decrease as dimensionality increases")
    print()
    
    dimensions = [1, 2, 3, 5, 7, 10]
    n_profiles = 200
    n_voters = 100
    n_candidates = 5
    voting_rules = ['plurality', 'borda', 'irv', 'score']
    
    results = {}
    
    for n_dim in dimensions:
        print(f"Testing {n_dim}D...", end=" ", flush=True)
        
        config = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method='uniform', n_dim=n_dim),
            utility=UtilityConfig(
                function='linear',
                distance_metric='l2',
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            ),
            rng_seed=42  # Fixed seed for reproducibility
        )
        
        experiment = run_experiment(config, save_results=False, verbose=False)
        
        dim_results = {}
        for rule in voting_rules:
            if rule in experiment.rule_results:
                vse = experiment.rule_results[rule].aggregate_metrics.vse_mean
                dim_results[rule] = vse
        
        results[n_dim] = dim_results
        print(f"Done (VSE: {dim_results.get('plurality', 0):.4f})")
    
    # Analyze trend
    print("\n" + "-" * 80)
    print("VSE Trend Analysis:")
    print("-" * 80)
    
    for rule in voting_rules:
        vse_values = [results[d].get(rule, 0) for d in dimensions if rule in results.get(d, {})]
        if len(vse_values) < 2:
            continue
        
        # Check if VSE decreases
        first_vse = vse_values[0]
        last_vse = vse_values[-1]
        change = last_vse - first_vse
        pct_change = 100 * change / first_vse if first_vse > 0 else 0
        
        trend = "DECREASING ✓" if change < -0.01 else "INCREASING ⚠️" if change > 0.01 else "STABLE"
        
        print(f"\n{rule.upper()}:")
        print(f"  1D → {dimensions[-1]}D: {first_vse:.4f} → {last_vse:.4f} ({change:+.4f}, {pct_change:+.1f}%)")
        print(f"  Trend: {trend}")
        
        # Check for monotonic decrease
        decreasing = all(vse_values[i] >= vse_values[i+1] for i in range(len(vse_values)-1))
        if not decreasing:
            print(f"  ⚠️  WARNING: Not monotonically decreasing!")
            print(f"  Values: {[f'{v:.4f}' for v in vse_values]}")
    
    return results


def test_distance_scaling():
    """Test if distances scale correctly with dimensionality."""
    print("\n" + "=" * 80)
    print("TEST 3: Distance Scaling with Dimensionality")
    print("=" * 80)
    
    dimensions = [1, 2, 3, 5, 10]
    n_voters = 100
    n_candidates = 5
    
    rng = np.random.default_rng(42)
    
    results = {}
    
    for n_dim in dimensions:
        print(f"\nTesting {n_dim}D...")
        
        # Generate positions
        voter_positions = rng.uniform(0, 1, (n_voters, n_dim))
        candidate_positions = rng.uniform(0, 1, (n_candidates, n_dim))
        
        # Test L2 distance
        distances_l2 = compute_distances(voter_positions, candidate_positions, metric='l2')
        max_dist_l2 = np.max(distances_l2)
        expected_max_l2 = np.sqrt(n_dim)  # Diagonal of unit hypercube
        
        print(f"  L2 max distance: {max_dist_l2:.4f} (expected: {expected_max_l2:.4f})")
        
        # Test L1 distance
        distances_l1 = compute_distances(voter_positions, candidate_positions, metric='l1')
        max_dist_l1 = np.max(distances_l1)
        expected_max_l1 = n_dim  # Sum of max differences
        
        print(f"  L1 max distance: {max_dist_l1:.4f} (expected: {expected_max_l1:.4f})")
        
        # Test Chebyshev distance
        distances_cheb = compute_distances(voter_positions, candidate_positions, metric='chebyshev')
        max_dist_cheb = np.max(distances_cheb)
        expected_max_cheb = 1.0  # Max coordinate difference in [0,1]
        
        print(f"  Chebyshev max distance: {max_dist_cheb:.4f} (expected: {expected_max_cheb:.4f})")
        
        results[n_dim] = {
            'l2': {'actual': max_dist_l2, 'expected': expected_max_l2},
            'l1': {'actual': max_dist_l1, 'expected': expected_max_l1},
            'chebyshev': {'actual': max_dist_cheb, 'expected': expected_max_cheb}
        }
    
    return results


def test_utility_function_consistency():
    """Test if utility functions produce consistent results."""
    print("\n" + "=" * 80)
    print("TEST 4: Utility Function Consistency")
    print("=" * 80)
    
    n_dim = 2
    n_voters = 100
    n_candidates = 5
    
    rng = np.random.default_rng(42)
    voter_positions = rng.uniform(0, 1, (n_voters, n_dim))
    candidate_positions = rng.uniform(0, 1, (n_candidates, n_dim))
    
    # Test different utility functions
    utility_functions = ['linear', 'quadratic', 'gaussian']
    
    results = {}
    
    for func in utility_functions:
        print(f"\nTesting {func} utility...")
        
        config = UtilityConfig(
            function=func,
            distance_metric='l2',
            sigma_factor=0.5 if func == 'gaussian' else None
        )
        
        utility_computer = UtilityComputer(config)
        distances = utility_computer.compute_distances(voter_positions, candidate_positions)
        utilities = utility_computer.compute_utilities(distances, n_dim)
        
        min_util = np.min(utilities)
        max_util = np.max(utilities)
        mean_util = np.mean(utilities)
        
        print(f"  Utility range: [{min_util:.4f}, {max_util:.4f}]")
        print(f"  Mean utility: {mean_util:.4f}")
        
        # Check for negative utilities (shouldn't happen for these functions)
        if min_util < -1e-6:
            print(f"  ⚠️  WARNING: Negative utilities found!")
        
        results[func] = {
            'min': min_util,
            'max': max_util,
            'mean': mean_util
        }
    
    return results


def test_heterogeneous_distance_consistency():
    """Test if heterogeneous distance produces expected behavior."""
    print("\n" + "=" * 80)
    print("TEST 5: Heterogeneous Distance Consistency")
    print("=" * 80)
    
    n_profiles = 100
    n_voters = 100
    n_candidates = 5
    n_dim = 2
    
    # Test homogeneous vs heterogeneous
    print("\nTesting homogeneous (L2) vs heterogeneous (L2 center, Cosine extreme)...")
    
    # Homogeneous
    config_homo = SimulationConfig(
        n_profiles=n_profiles,
        n_voters=n_voters,
        n_candidates=n_candidates,
        voting_rules=['plurality', 'borda'],
        geometry=GeometryConfig(method='uniform', n_dim=n_dim),
        utility=UtilityConfig(
            function='linear',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
        ),
        rng_seed=42
    )
    
    result_homo = run_experiment(config_homo, save_results=False, verbose=False)
    
    # Heterogeneous
    config_het = SimulationConfig(
        n_profiles=n_profiles,
        n_voters=n_voters,
        n_candidates=n_candidates,
        voting_rules=['plurality', 'borda'],
        geometry=GeometryConfig(method='uniform', n_dim=n_dim),
        utility=UtilityConfig(
            function='linear',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(
                enabled=True,
                strategy='center_extreme',
                center_metric='l2',
                extreme_metric='cosine',
                extreme_threshold=0.5
            )
        ),
        rng_seed=42  # Same seed for fair comparison
    )
    
    result_het = run_experiment(config_het, save_results=False, verbose=False)
    
    # Compare results
    print("\nComparison:")
    for rule in ['plurality', 'borda']:
        if rule in result_homo.rule_results and rule in result_het.rule_results:
            vse_homo = result_homo.rule_results[rule].aggregate_metrics.vse_mean
            vse_het = result_het.rule_results[rule].aggregate_metrics.vse_mean
            
            # Check winner disagreement
            winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
            winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
            disagreement = np.mean(winners_homo != winners_het) * 100
            
            print(f"\n{rule.upper()}:")
            print(f"  VSE (homogeneous): {vse_homo:.4f}")
            print(f"  VSE (heterogeneous): {vse_het:.4f}")
            print(f"  Winner disagreement: {disagreement:.1f}%")
            
            if disagreement < 1.0:
                print(f"  ⚠️  WARNING: Very low disagreement - heterogeneity may not be working!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SIMULATOR BUG CHECK - EXPERIMENTAL TESTS")
    print("=" * 80)
    print()
    
    all_results = {}
    
    # Test 1: Utility normalization
    all_results['utility_normalization'] = test_utility_normalization()
    
    # Test 2: VSE vs dimensionality (key test)
    all_results['vse_vs_dimensionality'] = test_vse_vs_dimensionality()
    
    # Test 3: Distance scaling
    all_results['distance_scaling'] = test_distance_scaling()
    
    # Test 4: Utility function consistency
    all_results['utility_consistency'] = test_utility_function_consistency()
    
    # Test 5: Heterogeneous distance
    test_heterogeneous_distance_consistency()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("- Check VSE trend: Should DECREASE with dimensionality")
    print("- Check utility ranges: Should be in [0, 1] for linear/quadratic")
    print("- Check distance scaling: Should match expected max distances")
    print("- Check heterogeneous effects: Should show disagreement > 0")
    
    return all_results


if __name__ == '__main__':
    results = main()

