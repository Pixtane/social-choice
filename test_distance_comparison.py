"""
Comprehensive tests comparing homogeneous and heterogeneous distance metrics.

This test suite systematically compares:
- Homogeneous distance metrics (l1, l2, cosine, chebyshev)
- Heterogeneous strategies (center_extreme, radial_steps)
- Different parameter settings
- Impact on voting outcomes and metrics
"""

import numpy as np
from simulator import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig, run_experiment
)
from simulator.utility import UtilityComputer
from simulator.geometry import GeometryGenerator
from simulator.heterogeneous_distance import (
    CenterExtremeStrategy, RadialStepsStrategy,
    compute_voter_centrality
)
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of comparing two configurations."""
    config_name: str
    vse_mean: float
    vse_std: float
    cycle_percentage: float
    condorcet_efficiency: float
    winner_rank_1st_pct: float
    compute_time: float
    metric_distribution: Dict[str, float] = None


def run_comparison_suite(
    n_profiles: int = 500,
    n_voters: int = 25,
    n_candidates: int = 3,
    voting_rules: List[str] = None,
    geometry_method: str = 'uniform',
    n_dim: int = 2,
    rng_seed: int = 42
) -> Dict[str, Any]:
    """
    Run comprehensive comparison of distance metrics.
    
    Returns:
        Dictionary with all comparison results
    """
    if voting_rules is None:
        voting_rules = ['plurality', 'borda', 'irv', 'approval', 'star', 'schulze']
    
    results = {}
    
    print("\n" + "=" * 80)
    print("DISTANCE METRIC COMPARISON SUITE")
    print("=" * 80)
    print(f"Profiles: {n_profiles} | Voters: {n_voters} | Candidates: {n_candidates}")
    print(f"Geometry: {geometry_method} | Dimensions: {n_dim}")
    print("=" * 80)
    
    # ========================================================================
    # 1. HOMOGENEOUS DISTANCE METRICS
    # ========================================================================
    print("\n" + "-" * 80)
    print("1. HOMOGENEOUS DISTANCE METRICS")
    print("-" * 80)
    
    homogeneous_metrics = ['l1', 'l2', 'cosine', 'chebyshev']
    
    for metric in homogeneous_metrics:
        print(f"\nTesting homogeneous {metric}...")
        
        config = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
            utility=UtilityConfig(
                function='gaussian',
                distance_metric=metric,
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            ),
            rng_seed=rng_seed
        )
        
        start_time = time.perf_counter()
        result = run_experiment(config, save_results=False, verbose=False)
        compute_time = time.perf_counter() - start_time
        
        # Store results for each voting rule
        for rule_name in voting_rules:
            key = f"homo_{metric}_{rule_name}"
            rule_result = result.rule_results[rule_name]
            agg = rule_result.aggregate_metrics
            
            results[key] = ComparisonResult(
                config_name=f"Homogeneous {metric}",
                vse_mean=agg.vse_mean,
                vse_std=agg.vse_std,
                cycle_percentage=agg.cycle_percentage,
                condorcet_efficiency=agg.condorcet_efficiency,
                winner_rank_1st_pct=agg.winner_rank_1st_pct,
                compute_time=compute_time / len(voting_rules),
                metric_distribution={metric: 1.0}
            )
            
            print(f"  {rule_name:12s} VSE: {agg.vse_mean:.4f} ± {agg.vse_std:.4f} | "
                  f"Cycles: {agg.cycle_percentage:.1f}% | "
                  f"CW Eff: {agg.condorcet_efficiency:.1f}%")
    
    # ========================================================================
    # 2. HETEROGENEOUS: CENTER-EXTREME STRATEGY
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. HETEROGENEOUS: CENTER-EXTREME STRATEGY")
    print("-" * 80)
    
    # Test different threshold values
    thresholds = [0.3, 0.5, 0.7]
    center_metrics = ['l2', 'l1']
    extreme_metrics = ['cosine', 'l2', 'chebyshev']
    
    for threshold in thresholds:
        for center_metric in center_metrics:
            for extreme_metric in extreme_metrics:
                if center_metric == extreme_metric:
                    continue  # Skip identical metrics
                
                config_name = f"CE_t{threshold:.1f}_{center_metric}_{extreme_metric}"
                print(f"\nTesting {config_name}...")
                
                config = SimulationConfig(
                    n_profiles=n_profiles,
                    n_voters=n_voters,
                    n_candidates=n_candidates,
                    voting_rules=voting_rules,
                    geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
                    utility=UtilityConfig(
                        function='gaussian',
                        heterogeneous_distance=HeterogeneousDistanceConfig(
                            enabled=True,
                            strategy='center_extreme',
                            center_metric=center_metric,
                            extreme_metric=extreme_metric,
                            extreme_threshold=threshold
                        )
                    ),
                    rng_seed=rng_seed
                )
                
                # Get metric distribution
                utility_computer = UtilityComputer(config.utility)
                # Generate sample positions to check distribution
                geometry_gen = GeometryGenerator(config.geometry, np.random.default_rng(rng_seed))
                sample_profile = geometry_gen.generate(1, n_voters, n_candidates)
                metric_dist = utility_computer.get_metric_distribution(
                    sample_profile.voter_positions[0]
                )
                
                start_time = time.perf_counter()
                result = run_experiment(config, save_results=False, verbose=False)
                compute_time = time.perf_counter() - start_time
                
                for rule_name in voting_rules:
                    key = f"{config_name}_{rule_name}"
                    rule_result = result.rule_results[rule_name]
                    agg = rule_result.aggregate_metrics
                    
                    results[key] = ComparisonResult(
                        config_name=config_name,
                        vse_mean=agg.vse_mean,
                        vse_std=agg.vse_std,
                        cycle_percentage=agg.cycle_percentage,
                        condorcet_efficiency=agg.condorcet_efficiency,
                        winner_rank_1st_pct=agg.winner_rank_1st_pct,
                        compute_time=compute_time / len(voting_rules),
                        metric_distribution=metric_dist
                    )
                    
                    print(f"  {rule_name:12s} VSE: {agg.vse_mean:.4f} ± {agg.vse_std:.4f} | "
                          f"Cycles: {agg.cycle_percentage:.1f}% | "
                          f"Dist: {metric_dist}")
    
    # ========================================================================
    # 3. HETEROGENEOUS: RADIAL STEPS STRATEGY
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. HETEROGENEOUS: RADIAL STEPS STRATEGY")
    print("-" * 80)
    
    radial_configs = [
        {
            'name': 'RS_linear_3',
            'metrics': ['l1', 'l2', 'chebyshev'],
            'scaling': 'linear',
            'param': 2.0
        },
        {
            'name': 'RS_log_3',
            'metrics': ['l1', 'l2', 'chebyshev'],
            'scaling': 'logarithmic',
            'param': 2.0
        },
        {
            'name': 'RS_exp_3',
            'metrics': ['l1', 'l2', 'chebyshev'],
            'scaling': 'exponential',
            'param': 2.0
        },
        {
            'name': 'RS_linear_4',
            'metrics': ['l1', 'l2', 'cosine', 'chebyshev'],
            'scaling': 'linear',
            'param': 2.0
        },
        {
            'name': 'RS_log_4',
            'metrics': ['l1', 'l2', 'cosine', 'chebyshev'],
            'scaling': 'logarithmic',
            'param': 3.0
        },
    ]
    
    for radial_config in radial_configs:
        config_name = radial_config['name']
        print(f"\nTesting {config_name}...")
        
        config = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
            utility=UtilityConfig(
                function='gaussian',
                heterogeneous_distance=HeterogeneousDistanceConfig(
                    enabled=True,
                    strategy='radial_steps',
                    radial_metrics=radial_config['metrics'],
                    radial_scaling=radial_config['scaling'],
                    scaling_parameter=radial_config['param']
                )
            ),
            rng_seed=rng_seed
        )
        
        # Get metric distribution
        utility_computer = UtilityComputer(config.utility)
        geometry_gen = GeometryGenerator(config.geometry, np.random.default_rng(rng_seed))
        sample_profile = geometry_gen.generate(1, n_voters, n_candidates)
        metric_dist = utility_computer.get_metric_distribution(
            sample_profile.voter_positions[0]
        )
        
        start_time = time.perf_counter()
        result = run_experiment(config, save_results=False, verbose=False)
        compute_time = time.perf_counter() - start_time
        
        for rule_name in voting_rules:
            key = f"{config_name}_{rule_name}"
            rule_result = result.rule_results[rule_name]
            agg = rule_result.aggregate_metrics
            
            results[key] = ComparisonResult(
                config_name=config_name,
                vse_mean=agg.vse_mean,
                vse_std=agg.vse_std,
                cycle_percentage=agg.cycle_percentage,
                condorcet_efficiency=agg.condorcet_efficiency,
                winner_rank_1st_pct=agg.winner_rank_1st_pct,
                compute_time=compute_time / len(voting_rules),
                metric_distribution=metric_dist
            )
            
            print(f"  {rule_name:12s} VSE: {agg.vse_mean:.4f} ± {agg.vse_std:.4f} | "
                  f"Cycles: {agg.cycle_percentage:.1f}% | "
                  f"Dist: {metric_dist}")
    
    # ========================================================================
    # 4. GEOMETRY-SPECIFIC COMPARISONS
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. GEOMETRY-SPECIFIC COMPARISONS")
    print("-" * 80)
    
    geometries = ['uniform', 'polarized', 'clustered']
    
    for geom_method in geometries:
        print(f"\nTesting geometry: {geom_method}")
        
        # Homogeneous L2 baseline
        config_homo = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method=geom_method, n_dim=n_dim),
            utility=UtilityConfig(
                function='gaussian',
                distance_metric='l2',
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            ),
            rng_seed=rng_seed
        )
        
        result_homo = run_experiment(config_homo, save_results=False, verbose=False)
        
        # Heterogeneous center-extreme
        config_het = SimulationConfig(
            n_profiles=n_profiles,
            n_voters=n_voters,
            n_candidates=n_candidates,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method=geom_method, n_dim=n_dim),
            utility=UtilityConfig(
                function='gaussian',
                heterogeneous_distance=HeterogeneousDistanceConfig(
                    enabled=True,
                    strategy='center_extreme',
                    center_metric='l2',
                    extreme_metric='cosine',
                    extreme_threshold=0.5
                )
            ),
            rng_seed=rng_seed
        )
        
        result_het = run_experiment(config_het, save_results=False, verbose=False)
        
        print(f"  Geometry: {geom_method}")
        for rule_name in voting_rules[:3]:  # Show first 3 rules
            vse_homo = result_homo.rule_results[rule_name].aggregate_metrics.vse_mean
            vse_het = result_het.rule_results[rule_name].aggregate_metrics.vse_mean
            diff = vse_het - vse_homo
            print(f"    {rule_name:12s} Homo: {vse_homo:.4f} | Het: {vse_het:.4f} | "
                  f"Diff: {diff:+.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON SUITE COMPLETE")
    print("=" * 80)
    
    return results


def analyze_results(results: Dict[str, ComparisonResult]) -> Dict[str, Any]:
    """
    Analyze comparison results and extract patterns.
    
    Returns:
        Dictionary with analysis findings
    """
    analysis = {
        'homogeneous_comparison': {},
        'heterogeneous_effects': {},
        'best_configs': {},
        'interesting_patterns': []
    }
    
    # Group results by voting rule
    by_rule = {}
    for key, result in results.items():
        rule = key.split('_')[-1]  # Last part is rule name
        if rule not in by_rule:
            by_rule[rule] = []
        by_rule[rule].append((key, result))
    
    # Find best homogeneous metric per rule
    for rule, rule_results in by_rule.items():
        homo_results = [(k, r) for k, r in rule_results if k.startswith('homo_')]
        if homo_results:
            best = max(homo_results, key=lambda x: x[1].vse_mean)
            analysis['homogeneous_comparison'][rule] = {
                'best_metric': best[0],
                'best_vse': best[1].vse_mean,
                'all_metrics': {k: r.vse_mean for k, r in homo_results}
            }
    
    # Compare homogeneous vs heterogeneous
    for rule, rule_results in by_rule.items():
        homo_l2 = next((r for k, r in rule_results if k == f'homo_l2_{rule}'), None)
        het_results = [(k, r) for k, r in rule_results if not k.startswith('homo_')]
        
        if homo_l2 and het_results:
            best_het = max(het_results, key=lambda x: x[1].vse_mean)
            worst_het = min(het_results, key=lambda x: x[1].vse_mean)
            
            analysis['heterogeneous_effects'][rule] = {
                'homo_l2_vse': homo_l2.vse_mean,
                'best_het_vse': best_het[1].vse_mean,
                'worst_het_vse': worst_het[1].vse_mean,
                'best_het_config': best_het[0],
                'improvement': best_het[1].vse_mean - homo_l2.vse_mean,
                'degradation': worst_het[1].vse_mean - homo_l2.vse_mean
            }
    
    # Find overall best configurations
    for rule, rule_results in by_rule.items():
        best = max(rule_results, key=lambda x: x[1].vse_mean)
        analysis['best_configs'][rule] = {
            'config': best[0],
            'vse': best[1].vse_mean,
            'cycle_pct': best[1].cycle_percentage,
            'cw_eff': best[1].condorcet_efficiency
        }
    
    return analysis


def print_analysis_report(analysis: Dict[str, Any]):
    """Print formatted analysis report."""
    print("\n" + "=" * 80)
    print("ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n1. HOMOGENEOUS METRIC COMPARISON")
    print("-" * 80)
    for rule, data in analysis['homogeneous_comparison'].items():
        print(f"\n{rule.upper()}:")
        print(f"  Best: {data['best_metric']} (VSE: {data['best_vse']:.4f})")
        for metric, vse in sorted(data['all_metrics'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {metric:30s} VSE: {vse:.4f}")
    
    print("\n2. HETEROGENEOUS EFFECTS")
    print("-" * 80)
    for rule, data in analysis['heterogeneous_effects'].items():
        print(f"\n{rule.upper()}:")
        print(f"  Homogeneous L2:     {data['homo_l2_vse']:.4f}")
        print(f"  Best Heterogeneous: {data['best_het_vse']:.4f} ({data['best_het_config']})")
        print(f"  Worst Heterogeneous: {data['worst_het_vse']:.4f}")
        print(f"  Improvement:       {data['improvement']:+.4f}")
        print(f"  Max Degradation:    {data['degradation']:+.4f}")
    
    print("\n3. BEST CONFIGURATIONS PER RULE")
    print("-" * 80)
    for rule, data in analysis['best_configs'].items():
        print(f"{rule:12s} | {data['config']:40s} | "
              f"VSE: {data['vse']:.4f} | "
              f"Cycles: {data['cycle_pct']:.1f}% | "
              f"CW Eff: {data['cw_eff']:.1f}%")


def main():
    """Main test function."""
    print("Starting distance metric comparison tests...")
    
    # Run comparison suite
    results = run_comparison_suite(
        n_profiles=500,
        n_voters=25,
        n_candidates=3,
        voting_rules=['plurality', 'borda', 'irv', 'approval', 'star', 'schulze'],
        geometry_method='uniform',
        n_dim=2,
        rng_seed=42
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print report
    print_analysis_report(analysis)
    
    # Return for documentation
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()

