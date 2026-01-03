"""
Systematic test suite for heterogeneity research.

Implements the research framework tests:
1. Baseline characterization
2. Systematic heterogeneity exploration
3. Context-dependent effects
4. Voting rule sensitivity
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig, VotingRuleConfig
)
from simulator.main import run_experiment, ExperimentResult


@dataclass
class TestResult:
    """Result from a single test configuration."""
    config_name: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    compute_time: float
    timestamp: str


class HeterogeneityTestSuite:
    """Systematic test suite for heterogeneity research."""

    def __init__(
        self,
        base_n_profiles: int = 100,
        base_n_voters: int = 100,
        base_n_candidates: int = 5,
        output_dir: str = "heterogeneity-research/results"
    ):
        """
        Initialize test suite.

        Args:
            base_n_profiles: Default number of profiles per test
            base_n_voters: Default number of voters
            base_n_candidates: Default number of candidates
            output_dir: Directory to save results
        """
        self.base_n_profiles = base_n_profiles
        self.base_n_voters = base_n_voters
        self.base_n_candidates = base_n_candidates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[TestResult] = []
        self.base_rng_seed = 42
        self._seed_counter = 0  # Counter for generating unique seeds

    def _get_next_seed(self) -> int:
        """Get next unique seed for independent Monte Carlo runs."""
        seed = self.base_rng_seed + self._seed_counter
        self._seed_counter += 1
        return seed

    def test_baseline_characterization(
        self,
        voting_rules: Optional[List[str]] = None,
        n_profiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Characterize baseline homogeneous distance metrics.

        Tests all 4 metrics (L1, L2, Cosine, Chebyshev) separately.
        """
        if voting_rules is None:
            voting_rules = ['plurality', 'borda', 'irv', 'approval', 'star', 'schulze']

        if n_profiles is None:
            n_profiles = self.base_n_profiles

        print("=" * 80)
        print("PHASE 1: BASELINE CHARACTERIZATION")
        print("=" * 80)
        print(f"Testing {len(voting_rules)} voting rules with {n_profiles} profiles")
        print()

        baseline_results = {}
        metrics = ['l1', 'l2', 'cosine', 'chebyshev']

        for metric in metrics:
            print(f"Testing {metric.upper()}...", end=" ", flush=True)
            start_time = time.perf_counter()

            config = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                ),
                rng_seed=self._get_next_seed()
            )

            result = run_experiment(config, save_results=False, verbose=False)

            # Extract metrics
            metric_data = {}
            for rule in voting_rules:
                if rule in result.rule_results:
                    rule_result = result.rule_results[rule]
                    metric_data[rule] = {
                        'vse_mean': rule_result.aggregate_metrics.vse_mean,
                        'vse_std': rule_result.aggregate_metrics.vse_std,
                        'condorcet_efficiency': rule_result.aggregate_metrics.condorcet_efficiency,
                        'cycle_percentage': rule_result.aggregate_metrics.cycle_percentage,
                    }

            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.2f}s)")

            baseline_results[metric] = metric_data

        # Save results
        output_file = self.output_dir / "baseline_characterization.json"
        with open(output_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return baseline_results

    def test_threshold_sweep(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        thresholds: Optional[np.ndarray] = None,
        voting_rules: Optional[List[str]] = None,
        n_profiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 2a: Sweep threshold parameter for center-extreme strategy.

        Args:
            center_metric: Metric for center voters
            extreme_metric: Metric for extreme voters
            thresholds: Array of threshold values (default: 0.1 to 0.9)
            voting_rules: List of rules to test
            n_profiles: Number of profiles
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        if voting_rules is None:
            voting_rules = ['plurality', 'borda', 'irv', 'approval', 'star']

        if n_profiles is None:
            n_profiles = self.base_n_profiles

        print("=" * 80)
        print(f"PHASE 2a: THRESHOLD SWEEP ({center_metric.upper()} + {extreme_metric.upper()})")
        print("=" * 80)
        print(f"Testing {len(thresholds)} thresholds with {n_profiles} profiles")
        print()

        sweep_results = {
            'center_metric': center_metric,
            'extreme_metric': extreme_metric,
            'thresholds': thresholds.tolist(),
            'results': {}
        }

        for threshold in thresholds:
            print(f"Threshold {threshold:.1f}...", end=" ", flush=True)
            start_time = time.perf_counter()

            # Get unique seeds for this iteration
            seed_het = self._get_next_seed()
            seed_homo = self._get_next_seed()

            config = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',  # Fallback
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric=center_metric,
                        extreme_metric=extreme_metric,
                        extreme_threshold=float(threshold)
                    )
                ),
                rng_seed=seed_het
            )

            result = run_experiment(config, save_results=False, verbose=False)

            # Also run homogeneous baseline for comparison
            config_homo = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=center_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                ),
                rng_seed=seed_homo
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)

            # Compute comparison metrics
            threshold_data = {}
            for rule in voting_rules:
                if rule in result.rule_results and rule in result_homo.rule_results:
                    het_result = result.rule_results[rule]
                    homo_result = result_homo.rule_results[rule]

                    # Compute disagreement
                    winners_het = np.array([m.winner_index for m in het_result.profile_metrics])
                    winners_homo = np.array([m.winner_index for m in homo_result.profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100

                    threshold_data[rule] = {
                        'vse_heterogeneous': het_result.aggregate_metrics.vse_mean,
                        'vse_homogeneous': homo_result.aggregate_metrics.vse_mean,
                        'vse_difference': (het_result.aggregate_metrics.vse_mean -
                                         homo_result.aggregate_metrics.vse_mean),
                        'disagreement_rate': disagreement,
                        'condorcet_efficiency_het': het_result.aggregate_metrics.condorcet_efficiency,
                        'condorcet_efficiency_homo': homo_result.aggregate_metrics.condorcet_efficiency,
                    }

            sweep_results['results'][f"{threshold:.1f}"] = threshold_data

            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.2f}s)")

        # Save results
        output_file = self.output_dir / f"threshold_sweep_{center_metric}_{extreme_metric}.json"
        with open(output_file, 'w') as f:
            json.dump(sweep_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return sweep_results

    def test_all_metric_pairs(
        self,
        thresholds: Optional[np.ndarray] = None,
        voting_rules: Optional[List[str]] = None,
        n_profiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 2b: Test all metric pairs with threshold sweep.

        Tests all combinations of center/extreme metrics.
        """
        if thresholds is None:
            thresholds = np.array([0.3, 0.5, 0.7])  # Key thresholds

        if voting_rules is None:
            voting_rules = ['plurality', 'borda', 'irv']

        if n_profiles is None:
            n_profiles = 50  # Smaller for comprehensive test

        metrics = ['l1', 'l2', 'cosine', 'chebyshev']
        metric_pairs = [(c, e) for c in metrics for e in metrics if c != e]

        print("=" * 80)
        print("PHASE 2b: ALL METRIC PAIRS")
        print("=" * 80)
        print(f"Testing {len(metric_pairs)} metric pairs")
        print(f"Each with {len(thresholds)} thresholds")
        print(f"{n_profiles} profiles per configuration")
        print()

        all_results = {}

        for center_metric, extreme_metric in metric_pairs:
            pair_name = f"{center_metric}_{extreme_metric}"
            print(f"\nTesting {pair_name}...")

            pair_results = self.test_threshold_sweep(
                center_metric=center_metric,
                extreme_metric=extreme_metric,
                thresholds=thresholds,
                voting_rules=voting_rules,
                n_profiles=n_profiles
            )

            all_results[pair_name] = pair_results

        # Save combined results
        output_file = self.output_dir / "all_metric_pairs.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nAll results saved to: {output_file}")
        return all_results

    def test_geometry_effects(
        self,
        geometries: Optional[List[str]] = None,
        het_config: Optional[HeterogeneousDistanceConfig] = None,
        voting_rules: Optional[List[str]] = None,
        n_profiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 3a: Test heterogeneity effects across different geometries.

        Args:
            geometries: List of geometry methods to test
            het_config: Heterogeneous distance configuration
            voting_rules: List of voting rules
            n_profiles: Number of profiles
        """
        if geometries is None:
            geometries = ['uniform', 'polarized', 'clustered', 'single_peaked']

        if het_config is None:
            het_config = HeterogeneousDistanceConfig(
                enabled=True,
                strategy='center_extreme',
                center_metric='l2',
                extreme_metric='cosine',
                extreme_threshold=0.5
            )

        if voting_rules is None:
            voting_rules = ['plurality', 'borda', 'irv']

        if n_profiles is None:
            n_profiles = self.base_n_profiles

        print("=" * 80)
        print("PHASE 3a: GEOMETRY EFFECTS")
        print("=" * 80)
        print(f"Testing {len(geometries)} geometries with {n_profiles} profiles")
        print()

        geometry_results = {}

        for geometry in geometries:
            print(f"Testing {geometry}...", end=" ", flush=True)
            start_time = time.perf_counter()

            # Get unique seeds for this iteration
            seed_het = self._get_next_seed()
            seed_homo = self._get_next_seed()

            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method=geometry, n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=het_config
                ),
                rng_seed=seed_het
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)

            # Homogeneous baseline
            config_homo = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method=geometry, n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                ),
                rng_seed=seed_homo
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)

            # Compare
            geometry_data = {}
            for rule in voting_rules:
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    het_result = result_het.rule_results[rule]
                    homo_result = result_homo.rule_results[rule]

                    winners_het = np.array([m.winner_index for m in het_result.profile_metrics])
                    winners_homo = np.array([m.winner_index for m in homo_result.profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100

                    geometry_data[rule] = {
                        'disagreement_rate': disagreement,
                        'vse_difference': (het_result.aggregate_metrics.vse_mean -
                                         homo_result.aggregate_metrics.vse_mean),
                    }

            geometry_results[geometry] = geometry_data

            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.2f}s)")

        # Save results
        output_file = self.output_dir / "geometry_effects.json"
        with open(output_file, 'w') as f:
            json.dump(geometry_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return geometry_results

    def test_dimensionality_effects(
        self,
        dimensions: Optional[List[int]] = None,
        het_config: Optional[HeterogeneousDistanceConfig] = None,
        voting_rules: Optional[List[str]] = None,
        n_profiles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Phase 3c: Test how heterogeneity effects change with dimensionality.

        Args:
            dimensions: List of dimensions to test
            het_config: Heterogeneous distance configuration
            voting_rules: List of voting rules
            n_profiles: Number of profiles
        """
        if dimensions is None:
            dimensions = [1, 2, 3, 5, 10]

        if het_config is None:
            het_config = HeterogeneousDistanceConfig(
                enabled=True,
                strategy='center_extreme',
                center_metric='l2',
                extreme_metric='cosine',
                extreme_threshold=0.5
            )

        if voting_rules is None:
            voting_rules = ['plurality', 'borda', 'irv']

        if n_profiles is None:
            n_profiles = self.base_n_profiles

        print("=" * 80)
        print("PHASE 3c: DIMENSIONALITY EFFECTS")
        print("=" * 80)
        print(f"Testing {len(dimensions)} dimensions with {n_profiles} profiles")
        print()

        dim_results = {}

        for n_dim in dimensions:
            print(f"Testing {n_dim}D...", end=" ", flush=True)
            start_time = time.perf_counter()

            # Get unique seeds for this iteration
            seed_het = self._get_next_seed()
            seed_homo = self._get_next_seed()

            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=n_dim),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=het_config
                ),
                rng_seed=seed_het
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)

            # Homogeneous baseline
            config_homo = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=n_dim),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                ),
                rng_seed=seed_homo
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)

            # Compare
            dim_data = {}
            for rule in voting_rules:
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    het_result = result_het.rule_results[rule]
                    homo_result = result_homo.rule_results[rule]

                    winners_het = np.array([m.winner_index for m in het_result.profile_metrics])
                    winners_homo = np.array([m.winner_index for m in homo_result.profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100

                    dim_data[rule] = {
                        'disagreement_rate': disagreement,
                        'vse_difference': (het_result.aggregate_metrics.vse_mean -
                                         homo_result.aggregate_metrics.vse_mean),
                    }

            dim_results[n_dim] = dim_data

            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.2f}s)")

        # Save results
        output_file = self.output_dir / "dimensionality_effects.json"
        with open(output_file, 'w') as f:
            json.dump(dim_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return dim_results

    def run_full_suite(
        self,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete test suite.

        Args:
            quick_mode: If True, use smaller parameters for faster execution
        """
        print("=" * 80)
        print("HETEROGENEITY RESEARCH - FULL TEST SUITE")
        print("=" * 80)
        print()

        if quick_mode:
            n_profiles = 20
            print("QUICK MODE: Using reduced parameters")
        else:
            n_profiles = self.base_n_profiles

        all_results = {}

        # Phase 1: Baseline
        print("\n" + "=" * 80)
        all_results['baseline'] = self.test_baseline_characterization(
            n_profiles=n_profiles
        )

        # Phase 2: Threshold sweep (key pair)
        print("\n" + "=" * 80)
        all_results['threshold_sweep'] = self.test_threshold_sweep(
            center_metric='l2',
            extreme_metric='cosine',
            n_profiles=n_profiles
        )

        # Phase 3: Context effects
        print("\n" + "=" * 80)
        all_results['geometry_effects'] = self.test_geometry_effects(
            n_profiles=n_profiles
        )

        all_results['dimensionality_effects'] = self.test_dimensionality_effects(
            n_profiles=n_profiles
        )

        # Save combined results
        output_file = self.output_dir / "full_suite_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 80)
        print("FULL SUITE COMPLETE")
        print(f"All results saved to: {output_file}")
        print("=" * 80)

        return all_results


def main():
    """Main entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run heterogeneity research tests")
    parser.add_argument(
        '--quick', action='store_true',
        help='Run in quick mode with reduced parameters'
    )
    parser.add_argument(
        '--phase', type=str,
        choices=['baseline', 'threshold', 'geometry', 'dimensions', 'all'],
        default='all',
        help='Which phase to run'
    )

    args = parser.parse_args()

    suite = HeterogeneityTestSuite()

    if args.phase == 'all':
        suite.run_full_suite(quick_mode=args.quick)
    elif args.phase == 'baseline':
        suite.test_baseline_characterization()
    elif args.phase == 'threshold':
        suite.test_threshold_sweep()
    elif args.phase == 'geometry':
        suite.test_geometry_effects()
    elif args.phase == 'dimensions':
        suite.test_dimensionality_effects()


if __name__ == '__main__':
    main()
