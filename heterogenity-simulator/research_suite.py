"""
Comprehensive research suite for heterogeneous distance metrics.

This script systematically investigates phenomena in heterogeneous distance metrics
with rigorous methodology:
- Minimum 200 profiles, 100 voters for conclusions
- Voter scaling tests (10-500 voters)
- Final verification with 500 voters
- Systematic parameter sweeps
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig
)
from simulator.main import run_experiment, ExperimentResult


@dataclass
class ResearchConfig:
    """Configuration for research experiments."""
    # Base parameters
    base_n_profiles: int = 200
    base_n_voters: int = 100
    base_n_candidates: int = 5
    
    # Voter scaling test range
    voter_scaling_range: List[int] = None
    
    # Final verification
    verification_n_voters: int = 500
    verification_n_profiles: int = 200
    
    # Dimensions to test
    dimensions: List[int] = None
    
    # Thresholds to test
    thresholds: np.ndarray = None
    
    # Metrics to test
    metrics: List[str] = None
    
    # Voting rules
    voting_rules: List[str] = None
    
    # Output directory
    output_dir: str = "heterogenity-simulator/results"
    
    # Random seed
    rng_seed: int = 42
    
    def __post_init__(self):
        if self.voter_scaling_range is None:
            self.voter_scaling_range = [10, 25, 50, 100, 200, 300, 400, 500]
        if self.dimensions is None:
            self.dimensions = [1, 2, 3, 4, 5, 7, 10]
        if self.thresholds is None:
            self.thresholds = np.linspace(0.05, 0.95, 19)
        if self.metrics is None:
            self.metrics = ['l1', 'l2', 'cosine', 'chebyshev']
        if self.voting_rules is None:
            self.voting_rules = ['plurality', 'borda', 'irv']


class HeterogeneityResearcher:
    """Comprehensive researcher for heterogeneous distance phenomena."""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.rng_seed)
        
        # Results storage
        self.results = {
            'config': asdict(config),
            'timestamp': datetime.now().isoformat(),
            'experiments': {}
        }
    
    def compute_disagreement(
        self,
        het_result: ExperimentResult,
        homo_result: ExperimentResult,
        rule: str
    ) -> float:
        """Compute disagreement rate between heterogeneous and homogeneous results."""
        if rule not in het_result.rule_results or rule not in homo_result.rule_results:
            return 0.0
        
        het_winners = het_result.rule_results[rule].winners
        homo_winners = homo_result.rule_results[rule].winners
        disagreement = np.mean(het_winners != homo_winners) * 100
        return disagreement

    def _run_homogeneous_baseline(
        self,
        *,
        n_profiles: int,
        n_voters: int,
        n_candidates: int,
        voting_rules: List[str],
        dimension: int,
        baseline_metric: str,
    ) -> ExperimentResult:
        """Run a homogeneous simulation using a single baseline distance metric."""
        config_homo = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
            utility=UtilityConfig(
                function='linear',
                distance_metric=baseline_metric,
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            ),
            rng_seed=self.config.rng_seed
        )
        return run_experiment(config_homo, save_results=False, verbose=False)
    
    def compute_condorcet_metrics(
        self,
        het_result: ExperimentResult,
        homo_result: ExperimentResult,
        rule: str
    ) -> Dict[str, float]:
        """Compute Condorcet-related metrics."""
        if rule not in het_result.rule_results or rule not in homo_result.rule_results:
            return {}
        
        het_metrics = het_result.rule_results[rule].aggregate_metrics
        homo_metrics = homo_result.rule_results[rule].aggregate_metrics
        
        return {
            'cycle_rate_het': het_metrics.cycle_percentage,
            'cycle_rate_homo': homo_metrics.cycle_percentage,
            'cycle_rate_delta': (het_metrics.cycle_percentage - 
                               homo_metrics.cycle_percentage),
            'condorcet_efficiency_het': het_metrics.condorcet_efficiency,
            'condorcet_efficiency_homo': homo_metrics.condorcet_efficiency,
            'condorcet_efficiency_delta': (het_metrics.condorcet_efficiency - 
                                         homo_metrics.condorcet_efficiency),
        }
    
    def experiment_voter_scaling(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        threshold: float = 0.5,
        dimension: int = 2
    ) -> Dict[str, Any]:
        """
        Experiment: Test how voter count affects heterogeneity effects.
        
        Tests with 10-500 voters to see if effects are stable or change.
        """
        print("=" * 80)
        print(f"EXPERIMENT: VOTER SCALING")
        print(f"Metrics: {center_metric} (center) -> {extreme_metric} (extreme)")
        print(f"Threshold: {threshold}, Dimension: {dimension}")
        print("=" * 80)
        
        results = {
            'center_metric': center_metric,
            'extreme_metric': extreme_metric,
            'threshold': threshold,
            'dimension': dimension,
            'voter_counts': [],
            'data': {}
        }
        
        for n_voters in self.config.voter_scaling_range:
            print(f"\nTesting with {n_voters} voters...", end=" ", flush=True)
            start_time = time.perf_counter()
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric=center_metric,
                        extreme_metric=extreme_metric,
                        extreme_threshold=threshold
                    )
                ),
                rng_seed=self.config.rng_seed
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous baseline (center metric)
            result_homo_center = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=center_metric,
            )
            result_homo_extreme = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=extreme_metric,
            )
            
            voter_data = {}
            for rule in self.config.voting_rules:
                disagreement = self.compute_disagreement(result_het, result_homo_center, rule)
                condorcet_metrics = self.compute_condorcet_metrics(result_het, result_homo_center, rule)
                disagreement_extreme_baseline = self.compute_disagreement(result_het, result_homo_extreme, rule)
                condorcet_metrics_extreme_baseline = self.compute_condorcet_metrics(
                    result_het, result_homo_extreme, rule
                )
                
                het_metrics = result_het.rule_results[rule].aggregate_metrics
                homo_metrics = result_homo_center.rule_results[rule].aggregate_metrics
                homo_metrics_extreme = result_homo_extreme.rule_results[rule].aggregate_metrics
                
                voter_data[rule] = {
                    'disagreement_rate': disagreement,
                    'vse_het': het_metrics.vse_mean,
                    'vse_homo': homo_metrics.vse_mean,
                    'vse_difference': het_metrics.vse_mean - homo_metrics.vse_mean,
                    'disagreement_rate_extreme_baseline': disagreement_extreme_baseline,
                    'vse_homo_extreme_baseline': homo_metrics_extreme.vse_mean,
                    'vse_difference_extreme_baseline': het_metrics.vse_mean - homo_metrics_extreme.vse_mean,
                    **{f'{k}_extreme_baseline': v for k, v in condorcet_metrics_extreme_baseline.items()},
                    **condorcet_metrics
                }
            
            results['voter_counts'].append(n_voters)
            results['data'][str(n_voters)] = voter_data
            
            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.1f}s)")
        
        # Save results
        output_file = self.output_dir / f"voter_scaling_{center_metric}_{extreme_metric}_d{dimension}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    
    def experiment_threshold_sweep(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        dimension: int = 2,
        n_voters: int = 100
    ) -> Dict[str, Any]:
        """
        Experiment: Fine-grained threshold sweep to detect phase transitions.
        """
        print("=" * 80)
        print(f"EXPERIMENT: THRESHOLD SWEEP")
        print(f"Metrics: {center_metric} (center) -> {extreme_metric} (extreme)")
        print(f"Dimension: {dimension}, Voters: {n_voters}")
        print("=" * 80)
        
        results = {
            'center_metric': center_metric,
            'extreme_metric': extreme_metric,
            'dimension': dimension,
            'n_voters': n_voters,
            'thresholds': self.config.thresholds.tolist(),
            'data': {}
        }
        
        for threshold in self.config.thresholds:
            print(f"Threshold {threshold:.2f}...", end=" ", flush=True)
            start_time = time.perf_counter()
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric=center_metric,
                        extreme_metric=extreme_metric,
                        extreme_threshold=float(threshold)
                    )
                ),
                rng_seed=self.config.rng_seed
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous baseline
            result_homo_center = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=center_metric,
            )
            result_homo_extreme = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=extreme_metric,
            )
            
            threshold_data = {}
            for rule in self.config.voting_rules:
                disagreement = self.compute_disagreement(result_het, result_homo_center, rule)
                condorcet_metrics = self.compute_condorcet_metrics(result_het, result_homo_center, rule)
                disagreement_extreme_baseline = self.compute_disagreement(result_het, result_homo_extreme, rule)
                condorcet_metrics_extreme_baseline = self.compute_condorcet_metrics(
                    result_het, result_homo_extreme, rule
                )
                
                het_metrics = result_het.rule_results[rule].aggregate_metrics
                homo_metrics = result_homo_center.rule_results[rule].aggregate_metrics
                homo_metrics_extreme = result_homo_extreme.rule_results[rule].aggregate_metrics
                
                threshold_data[rule] = {
                    'disagreement_rate': disagreement,
                    'vse_het': het_metrics.vse_mean,
                    'vse_homo': homo_metrics.vse_mean,
                    'vse_difference': het_metrics.vse_mean - homo_metrics.vse_mean,
                    'disagreement_rate_extreme_baseline': disagreement_extreme_baseline,
                    'vse_homo_extreme_baseline': homo_metrics_extreme.vse_mean,
                    'vse_difference_extreme_baseline': het_metrics.vse_mean - homo_metrics_extreme.vse_mean,
                    **{f'{k}_extreme_baseline': v for k, v in condorcet_metrics_extreme_baseline.items()},
                    **condorcet_metrics
                }
            
            results['data'][f"{threshold:.2f}"] = threshold_data
            
            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.1f}s)")
        
        # Save results
        output_file = self.output_dir / f"threshold_sweep_{center_metric}_{extreme_metric}_d{dimension}_v{n_voters}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    
    def experiment_dimensional_scaling(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        threshold: float = 0.5,
        n_voters: int = 100
    ) -> Dict[str, Any]:
        """
        Experiment: Test how dimensionality affects heterogeneity effects.
        """
        print("=" * 80)
        print(f"EXPERIMENT: DIMENSIONAL SCALING")
        print(f"Metrics: {center_metric} (center) -> {extreme_metric} (extreme)")
        print(f"Threshold: {threshold}, Voters: {n_voters}")
        print("=" * 80)
        
        results = {
            'center_metric': center_metric,
            'extreme_metric': extreme_metric,
            'threshold': threshold,
            'n_voters': n_voters,
            'dimensions': [],
            'data': {}
        }
        
        for dimension in self.config.dimensions:
            print(f"\nTesting dimension {dimension}...", end=" ", flush=True)
            start_time = time.perf_counter()
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric=center_metric,
                        extreme_metric=extreme_metric,
                        extreme_threshold=threshold
                    )
                ),
                rng_seed=self.config.rng_seed
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous baseline
            result_homo_center = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=center_metric,
            )
            result_homo_extreme = self._run_homogeneous_baseline(
                n_profiles=self.config.base_n_profiles,
                n_voters=n_voters,
                n_candidates=self.config.base_n_candidates,
                voting_rules=self.config.voting_rules,
                dimension=dimension,
                baseline_metric=extreme_metric,
            )
            
            dim_data = {}
            for rule in self.config.voting_rules:
                disagreement = self.compute_disagreement(result_het, result_homo_center, rule)
                condorcet_metrics = self.compute_condorcet_metrics(result_het, result_homo_center, rule)
                disagreement_extreme_baseline = self.compute_disagreement(result_het, result_homo_extreme, rule)
                condorcet_metrics_extreme_baseline = self.compute_condorcet_metrics(
                    result_het, result_homo_extreme, rule
                )
                
                het_metrics = result_het.rule_results[rule].aggregate_metrics
                homo_metrics = result_homo_center.rule_results[rule].aggregate_metrics
                homo_metrics_extreme = result_homo_extreme.rule_results[rule].aggregate_metrics
                
                dim_data[rule] = {
                    'disagreement_rate': disagreement,
                    'vse_het': het_metrics.vse_mean,
                    'vse_homo': homo_metrics.vse_mean,
                    'vse_difference': het_metrics.vse_mean - homo_metrics.vse_mean,
                    'disagreement_rate_extreme_baseline': disagreement_extreme_baseline,
                    'vse_homo_extreme_baseline': homo_metrics_extreme.vse_mean,
                    'vse_difference_extreme_baseline': het_metrics.vse_mean - homo_metrics_extreme.vse_mean,
                    **{f'{k}_extreme_baseline': v for k, v in condorcet_metrics_extreme_baseline.items()},
                    **condorcet_metrics
                }
            
            results['dimensions'].append(dimension)
            results['data'][str(dimension)] = dim_data
            
            elapsed = time.perf_counter() - start_time
            print(f"Done ({elapsed:.1f}s)")
        
        # Save results
        output_file = self.output_dir / f"dimensional_scaling_{center_metric}_{extreme_metric}_v{n_voters}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    
    def experiment_metric_pairs(
        self,
        threshold: float = 0.5,
        dimension: int = 2,
        n_voters: int = 100
    ) -> Dict[str, Any]:
        """
        Experiment: Test all metric pairs for asymmetric interactions.
        """
        print("=" * 80)
        print(f"EXPERIMENT: METRIC PAIR INTERACTIONS")
        print(f"Threshold: {threshold}, Dimension: {dimension}, Voters: {n_voters}")
        print("=" * 80)
        
        results = {
            'threshold': threshold,
            'dimension': dimension,
            'n_voters': n_voters,
            'pairs': {}
        }
        
        # Test all ordered pairs
        for center_metric in self.config.metrics:
            for extreme_metric in self.config.metrics:
                if center_metric == extreme_metric:
                    continue
                
                pair_name = f"{center_metric}_{extreme_metric}"
                print(f"\nTesting {pair_name}...", end=" ", flush=True)
                start_time = time.perf_counter()
                
                # Heterogeneous (A -> B)
                config_het = SimulationConfig(
                    n_profiles=self.config.base_n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.config.base_n_candidates,
                    voting_rules=self.config.voting_rules,
                    geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric='l2',
                        heterogeneous_distance=HeterogeneousDistanceConfig(
                            enabled=True,
                            strategy='center_extreme',
                            center_metric=center_metric,
                            extreme_metric=extreme_metric,
                            extreme_threshold=threshold
                        )
                    ),
                    rng_seed=self.config.rng_seed
                )
                result_het_ab = run_experiment(config_het, save_results=False, verbose=False)
                
                # Heterogeneous (B -> A) - reversed
                config_het_rev = SimulationConfig(
                    n_profiles=self.config.base_n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.config.base_n_candidates,
                    voting_rules=self.config.voting_rules,
                    geometry=GeometryConfig(method='uniform', n_dim=dimension, position_min=-1.0, position_max=1.0),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric='l2',
                        heterogeneous_distance=HeterogeneousDistanceConfig(
                            enabled=True,
                            strategy='center_extreme',
                            center_metric=extreme_metric,
                            extreme_metric=center_metric,
                            extreme_threshold=threshold
                        )
                    ),
                    rng_seed=self.config.rng_seed
                )
                result_het_ba = run_experiment(config_het_rev, save_results=False, verbose=False)

                # Homogeneous baselines
                # Baseline aligned with each heterogeneous run's center metric (default comparison)
                result_homo_center_ab = self._run_homogeneous_baseline(
                    n_profiles=self.config.base_n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.config.base_n_candidates,
                    voting_rules=self.config.voting_rules,
                    dimension=dimension,
                    baseline_metric=center_metric,
                )
                result_homo_center_ba = self._run_homogeneous_baseline(
                    n_profiles=self.config.base_n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.config.base_n_candidates,
                    voting_rules=self.config.voting_rules,
                    dimension=dimension,
                    baseline_metric=extreme_metric,
                )
                # Secondary comparison baseline aligned with each heterogeneous run's extreme metric.
                # These are the same two homogeneous runs as above, just swapped.
                result_homo_extreme_ab = result_homo_center_ba
                result_homo_extreme_ba = result_homo_center_ab
                
                pair_data = {}
                for rule in self.config.voting_rules:
                    # Default: compare each heterogeneous run to the homogeneous run using its center metric
                    disagreement_ab = self.compute_disagreement(result_het_ab, result_homo_center_ab, rule)
                    disagreement_ba = self.compute_disagreement(result_het_ba, result_homo_center_ba, rule)
                    asymmetry = abs(disagreement_ab - disagreement_ba)

                    # Secondary: compare to homogeneous run using the extreme metric
                    disagreement_ab_extreme_baseline = self.compute_disagreement(
                        result_het_ab, result_homo_extreme_ab, rule
                    )
                    disagreement_ba_extreme_baseline = self.compute_disagreement(
                        result_het_ba, result_homo_extreme_ba, rule
                    )
                    asymmetry_extreme_baseline = abs(disagreement_ab_extreme_baseline - disagreement_ba_extreme_baseline)
                    
                    condorcet_ab = self.compute_condorcet_metrics(result_het_ab, result_homo_center_ab, rule)
                    condorcet_ba = self.compute_condorcet_metrics(result_het_ba, result_homo_center_ba, rule)
                    condorcet_ab_extreme_baseline = self.compute_condorcet_metrics(
                        result_het_ab, result_homo_extreme_ab, rule
                    )
                    condorcet_ba_extreme_baseline = self.compute_condorcet_metrics(
                        result_het_ba, result_homo_extreme_ba, rule
                    )
                    
                    pair_data[rule] = {
                        'disagreement_ab': disagreement_ab,
                        'disagreement_ba': disagreement_ba,
                        'asymmetry': asymmetry,
                        'disagreement_ab_extreme_baseline': disagreement_ab_extreme_baseline,
                        'disagreement_ba_extreme_baseline': disagreement_ba_extreme_baseline,
                        'asymmetry_extreme_baseline': asymmetry_extreme_baseline,
                        **{f'{k}_ab': v for k, v in condorcet_ab.items()},
                        **{f'{k}_ba': v for k, v in condorcet_ba.items()},
                        **{f'{k}_ab_extreme_baseline': v for k, v in condorcet_ab_extreme_baseline.items()},
                        **{f'{k}_ba_extreme_baseline': v for k, v in condorcet_ba_extreme_baseline.items()},
                    }
                
                results['pairs'][pair_name] = pair_data
                
                elapsed = time.perf_counter() - start_time
                print(f"Done ({elapsed:.1f}s)")
        
        # Save results
        output_file = self.output_dir / f"metric_pairs_d{dimension}_v{n_voters}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return results
    
    def run_full_research_suite(self):
        """Run the complete research suite."""
        print("=" * 80)
        print("COMPREHENSIVE HETEROGENEITY RESEARCH SUITE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Base config: {self.config.base_n_profiles} profiles, {self.config.base_n_voters} voters")
        print()
        
        overall_start = time.perf_counter()
        
        # 1. Voter scaling experiment (key metric pair)
        print("\n" + "=" * 80)
        print("PHASE 1: VOTER SCALING ANALYSIS")
        print("=" * 80)
        voter_scaling = self.experiment_voter_scaling(
            center_metric='l2',
            extreme_metric='cosine',
            threshold=0.5,
            dimension=2
        )
        self.results['experiments']['voter_scaling'] = voter_scaling
        
        # 2. Threshold sweep (key metric pair)
        print("\n" + "=" * 80)
        print("PHASE 2: THRESHOLD SWEEP")
        print("=" * 80)
        threshold_sweep = self.experiment_threshold_sweep(
            center_metric='l2',
            extreme_metric='cosine',
            dimension=2,
            n_voters=100
        )
        self.results['experiments']['threshold_sweep'] = threshold_sweep
        
        # 3. Dimensional scaling
        print("\n" + "=" * 80)
        print("PHASE 3: DIMENSIONAL SCALING")
        print("=" * 80)
        dimensional_scaling = self.experiment_dimensional_scaling(
            center_metric='l2',
            extreme_metric='cosine',
            threshold=0.5,
            n_voters=100
        )
        self.results['experiments']['dimensional_scaling'] = dimensional_scaling
        
        # 4. Metric pair interactions
        print("\n" + "=" * 80)
        print("PHASE 4: METRIC PAIR INTERACTIONS")
        print("=" * 80)
        metric_pairs = self.experiment_metric_pairs(
            threshold=0.5,
            dimension=2,
            n_voters=100
        )
        self.results['experiments']['metric_pairs'] = metric_pairs
        
        # 5. Final verification with 500 voters
        print("\n" + "=" * 80)
        print("PHASE 5: FINAL VERIFICATION (500 VOTERS)")
        print("=" * 80)
        verification = self.experiment_metric_pairs(
            threshold=0.5,
            dimension=2,
            n_voters=self.config.verification_n_voters
        )
        self.results['experiments']['verification_500_voters'] = verification
        
        # Save all results
        output_file = self.output_dir / "full_research_suite.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        total_time = time.perf_counter() - overall_start
        print("\n" + "=" * 80)
        print("RESEARCH SUITE COMPLETE")
        print("=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {output_file}")
        
        return self.results


if __name__ == "__main__":
    config = ResearchConfig(
        base_n_profiles=200,
        base_n_voters=100,
        voter_scaling_range=[10, 25, 50, 100, 200, 300, 400, 500],
        verification_n_voters=500,
        verification_n_profiles=200
    )
    
    researcher = HeterogeneityResearcher(config)
    results = researcher.run_full_research_suite()

