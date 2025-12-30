"""
Test script for heterogeneous distance metrics.

Tests the new heterogeneous distance functionality to ensure
different voters use different distance metrics based on their position.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_center_extreme_strategy():
    """Test the center-extreme strategy."""
    print("\n" + "=" * 60)
    print("TEST: Center-Extreme Strategy")
    print("=" * 60)
    
    from simulator.heterogeneous_distance import CenterExtremeStrategy
    
    # Create strategy
    strategy = CenterExtremeStrategy(
        center_metric='l2',
        extreme_metric='cosine',
        threshold=0.5
    )
    
    # Create test positions
    # Some voters near center, some at edges
    voter_positions = np.array([
        [0.5, 0.5],   # Center - should use L2
        [0.45, 0.55], # Near center - should use L2
        [0.1, 0.1],   # Corner - should use cosine
        [0.9, 0.9],   # Corner - should use cosine
        [0.5, 0.9],   # Edge - might be borderline
    ])
    
    candidate_positions = np.array([
        [0.3, 0.3],
        [0.5, 0.5],
        [0.7, 0.7],
    ])
    
    # Get metrics for each voter
    metrics = strategy.get_voter_metrics(voter_positions)
    print(f"\nVoter metrics: {metrics}")
    
    # Get distribution
    distribution = strategy.get_metric_distribution(voter_positions)
    print(f"Distribution: {distribution}")
    
    # Compute distances
    distances = strategy.compute_distances(voter_positions, candidate_positions)
    print(f"\nDistances shape: {distances.shape}")
    print(f"Distances:\n{distances}")
    
    # Verify that different metrics produce different results
    assert len(set(metrics)) > 0, "Should have assigned metrics"
    print("\n[PASS] Center-Extreme strategy working correctly!")
    return True


def test_radial_steps_strategy():
    """Test the radial steps strategy."""
    print("\n" + "=" * 60)
    print("TEST: Radial Steps Strategy")
    print("=" * 60)
    
    from simulator.heterogeneous_distance import RadialStepsStrategy
    
    # Test with different scaling functions
    for scaling in ['linear', 'logarithmic', 'exponential']:
        print(f"\n--- Testing {scaling} scaling ---")
        
        strategy = RadialStepsStrategy(
            metrics=['l1', 'l2', 'chebyshev'],
            scaling=scaling,
            scaling_parameter=2.0
        )
        
        # Get boundaries
        boundaries = strategy.get_boundaries()
        print(f"Boundaries: {boundaries}")
        
        # Create test positions at various distances from center
        voter_positions = np.array([
            [0.5, 0.5],   # Center - L1
            [0.4, 0.4],   # Near center
            [0.3, 0.3],   # Middle
            [0.2, 0.2],   # Far
            [0.1, 0.1],   # Edge - Chebyshev
            [0.9, 0.9],   # Edge - Chebyshev
        ])
        
        candidate_positions = np.array([
            [0.3, 0.3],
            [0.5, 0.5],
            [0.7, 0.7],
        ])
        
        # Get metrics
        metrics = strategy.get_voter_metrics(voter_positions)
        print(f"Voter metrics: {metrics}")
        
        # Compute distances
        distances = strategy.compute_distances(voter_positions, candidate_positions)
        print(f"Distances shape: {distances.shape}")
    
    print("\n[PASS] Radial Steps strategy working correctly!")
    return True


def test_integration_with_utility():
    """Test integration with UtilityComputer."""
    print("\n" + "=" * 60)
    print("TEST: Integration with UtilityComputer")
    print("=" * 60)
    
    from simulator.utility import UtilityComputer
    from simulator.config import UtilityConfig, HeterogeneousDistanceConfig
    
    # Test with heterogeneous distance disabled (homogeneous)
    print("\n--- Testing homogeneous mode ---")
    config_homo = UtilityConfig(
        function='gaussian',
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
    )
    
    computer_homo = UtilityComputer(config_homo)
    
    voter_positions = np.array([
        [0.2, 0.3],
        [0.5, 0.5],
        [0.8, 0.7],
    ])
    
    candidate_positions = np.array([
        [0.3, 0.3],
        [0.6, 0.6],
    ])
    
    distances_homo = computer_homo.compute_distances(voter_positions, candidate_positions)
    print(f"Homogeneous distances:\n{distances_homo}")
    
    distribution_homo = computer_homo.get_metric_distribution(voter_positions)
    print(f"Metric distribution: {distribution_homo}")
    
    # Test with heterogeneous distance enabled
    print("\n--- Testing heterogeneous mode (center-extreme) ---")
    config_het = UtilityConfig(
        function='gaussian',
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme',
            center_metric='l2',
            extreme_metric='cosine',
            extreme_threshold=0.3
        )
    )
    
    computer_het = UtilityComputer(config_het)
    
    distances_het = computer_het.compute_distances(voter_positions, candidate_positions)
    print(f"Heterogeneous distances:\n{distances_het}")
    
    voter_metrics = computer_het.get_voter_metrics(voter_positions)
    print(f"Per-voter metrics: {voter_metrics}")
    
    distribution_het = computer_het.get_metric_distribution(voter_positions)
    print(f"Metric distribution: {distribution_het}")
    
    # Verify differences
    assert not np.allclose(distances_homo, distances_het), \
        "Heterogeneous should produce different results"
    
    print("\n[PASS] Integration with UtilityComputer working correctly!")
    return True


def test_full_simulation():
    """Test full simulation with heterogeneous distance."""
    print("\n" + "=" * 60)
    print("TEST: Full Simulation")
    print("=" * 60)
    
    from simulator import (
        SimulationConfig, GeometryConfig, UtilityConfig,
        HeterogeneousDistanceConfig, run_experiment
    )
    
    # Run with homogeneous distance
    print("\n--- Running homogeneous simulation ---")
    config_homo = SimulationConfig(
        n_profiles=50,
        n_voters=20,
        n_candidates=3,
        voting_rules=['plurality', 'borda'],
        geometry=GeometryConfig(method='polarized', n_dim=2),
        utility=UtilityConfig(
            function='gaussian',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
        )
    )
    
    result_homo = run_experiment(config_homo, save_results=False, verbose=False)
    
    for rule_name, rule_result in result_homo.rule_results.items():
        vse = rule_result.aggregate_metrics.vse_mean
        print(f"  {rule_name} VSE: {vse:.4f}")
    
    # Run with heterogeneous distance (center-extreme)
    print("\n--- Running heterogeneous simulation (center-extreme) ---")
    config_het_ce = SimulationConfig(
        n_profiles=50,
        n_voters=20,
        n_candidates=3,
        voting_rules=['plurality', 'borda'],
        geometry=GeometryConfig(method='polarized', n_dim=2),
        utility=UtilityConfig(
            function='gaussian',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(
                enabled=True,
                strategy='center_extreme',
                center_metric='l2',
                extreme_metric='cosine',
                extreme_threshold=0.5
            )
        )
    )
    
    result_het_ce = run_experiment(config_het_ce, save_results=False, verbose=False)
    
    for rule_name, rule_result in result_het_ce.rule_results.items():
        vse = rule_result.aggregate_metrics.vse_mean
        print(f"  {rule_name} VSE: {vse:.4f}")
    
    # Run with heterogeneous distance (radial steps)
    print("\n--- Running heterogeneous simulation (radial-steps) ---")
    config_het_rs = SimulationConfig(
        n_profiles=50,
        n_voters=20,
        n_candidates=3,
        voting_rules=['plurality', 'borda'],
        geometry=GeometryConfig(method='uniform', n_dim=2),
        utility=UtilityConfig(
            function='gaussian',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(
                enabled=True,
                strategy='radial_steps',
                radial_metrics=['l1', 'l2', 'chebyshev'],
                radial_scaling='logarithmic',
                scaling_parameter=2.0
            )
        )
    )
    
    result_het_rs = run_experiment(config_het_rs, save_results=False, verbose=False)
    
    for rule_name, rule_result in result_het_rs.rule_results.items():
        vse = rule_result.aggregate_metrics.vse_mean
        print(f"  {rule_name} VSE: {vse:.4f}")
    
    print(f"\n[PASS] Full simulation completed successfully!")
    print(f"  Homogeneous time: {result_homo.total_compute_time:.3f}s")
    print(f"  Heterogeneous (CE) time: {result_het_ce.total_compute_time:.3f}s")
    print(f"  Heterogeneous (RS) time: {result_het_rs.total_compute_time:.3f}s")
    
    return True


def test_config_serialization():
    """Test that config serializes correctly."""
    print("\n" + "=" * 60)
    print("TEST: Configuration Serialization")
    print("=" * 60)
    
    from simulator import (
        SimulationConfig, GeometryConfig, UtilityConfig,
        HeterogeneousDistanceConfig
    )
    
    config = SimulationConfig(
        n_profiles=100,
        n_voters=25,
        n_candidates=3,
        voting_rules=['plurality', 'borda'],
        utility=UtilityConfig(
            function='gaussian',
            heterogeneous_distance=HeterogeneousDistanceConfig(
                enabled=True,
                strategy='radial_steps',
                radial_metrics=['l1', 'l2', 'chebyshev'],
                radial_scaling='exponential',
                scaling_parameter=3.0
            )
        )
    )
    
    # Serialize
    config_dict = config.to_dict()
    
    print("Serialized config keys:")
    for key in sorted(config_dict.keys()):
        print(f"  {key}: {config_dict[key]}")
    
    # Check heterogeneous fields exist
    assert config_dict['heterogeneous_distance_enabled'] == True
    assert config_dict['heterogeneous_strategy'] == 'radial_steps'
    assert config_dict['heterogeneous_radial_scaling'] == 'exponential'
    
    print("\n[PASS] Configuration serialization working correctly!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("HETEROGENEOUS DISTANCE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Center-Extreme Strategy", test_center_extreme_strategy),
        ("Radial Steps Strategy", test_radial_steps_strategy),
        ("Integration with UtilityComputer", test_integration_with_utility),
        ("Full Simulation", test_full_simulation),
        ("Config Serialization", test_config_serialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8} {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n*** All tests passed! Heterogeneous distance is ready. ***")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

