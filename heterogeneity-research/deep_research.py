"""
Deep research script to discover novel phenomena in heterogeneous distance metrics.

This script performs systematic exploration to find:
1. Non-linear threshold effects
2. Metric interaction patterns
3. Dimensional scaling laws
4. Preference structure changes
5. Voting rule sensitivity patterns
"""

import numpy as np
import json
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
from simulator.heterogeneous_distance import compute_voter_centrality


@dataclass
class DeepResearchResult:
    """Result from deep research experiment."""
    experiment_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str


class DeepResearcher:
    """Deep research into heterogeneous distance phenomena."""
    
    def __init__(
        self,
        base_n_profiles: int = 200,
        base_n_voters: int = 100,
        base_n_candidates: int = 5,
        output_dir: str = "heterogeneity-research/results"
    ):
        self.base_n_profiles = base_n_profiles
        self.base_n_voters = base_n_voters
        self.base_n_candidates = base_n_candidates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(42)
    
    def experiment_threshold_nonlinearity(
        self,
        center_metric: str = 'l2',
        extreme_metric: str = 'cosine',
        n_profiles: int = 200
    ) -> Dict[str, Any]:
        """
        Experiment 1: Test for non-linear threshold effects.
        
        Uses fine-grained threshold sweep to detect:
        - Phase transitions
        - Critical thresholds
        - Discontinuities
        """
        print("=" * 80)
        print("EXPERIMENT 1: THRESHOLD NON-LINEARITY")
        print("=" * 80)
        
        # Fine-grained threshold sweep
        thresholds = np.linspace(0.05, 0.95, 19)  # 19 points for smooth curves
        
        results = {
            'center_metric': center_metric,
            'extreme_metric': extreme_metric,
            'thresholds': thresholds.tolist(),
            'data': {}
        }
        
        voting_rules = ['plurality', 'borda', 'irv', 'approval', 'star']
        
        for threshold in thresholds:
            print(f"Threshold {threshold:.2f}...", end=" ", flush=True)
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
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
                )
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous baseline (center metric)
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
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            threshold_data = {}
            for rule in voting_rules:
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    het_result = result_het.rule_results[rule]
                    homo_result = result_homo.rule_results[rule]
                    
                    winners_het = np.array([m.winner_index for m in het_result.profile_metrics])
                    winners_homo = np.array([m.winner_index for m in homo_result.profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100
                    
                    # Compute second derivative proxy (curvature)
                    threshold_data[rule] = {
                        'disagreement_rate': disagreement,
                        'vse_het': het_result.aggregate_metrics.vse_mean,
                        'vse_homo': homo_result.aggregate_metrics.vse_mean,
                        'vse_difference': (het_result.aggregate_metrics.vse_mean - 
                                         homo_result.aggregate_metrics.vse_mean),
                        'vse_std_het': het_result.aggregate_metrics.vse_std,
                        'vse_std_homo': homo_result.aggregate_metrics.vse_std,
                    }
            
            results['data'][f"{threshold:.2f}"] = threshold_data
            print("Done")
        
        # Analyze for non-linearity
        analysis = self._analyze_nonlinearity(results)
        results['analysis'] = analysis
        
        # Save
        output_file = self.output_dir / "threshold_nonlinearity.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def _analyze_nonlinearity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threshold data for non-linear patterns."""
        thresholds = np.array(results['thresholds'])
        analysis = {}
        
        for rule in ['plurality', 'borda', 'irv']:
            if rule not in results['data'].get('0.50', {}):
                continue
            
            disagreements = []
            vse_diffs = []
            
            for t in thresholds:
                key = f"{t:.2f}"
                if key in results['data'] and rule in results['data'][key]:
                    disagreements.append(results['data'][key][rule]['disagreement_rate'])
                    vse_diffs.append(results['data'][key][rule]['vse_difference'])
            
            if len(disagreements) < 3:
                continue
            
            disagreements = np.array(disagreements)
            vse_diffs = np.array(vse_diffs)
            
            # Compute first and second derivatives (discrete)
            d1_disagreement = np.diff(disagreements)
            d2_disagreement = np.diff(d1_disagreement)
            
            # Find inflection points (where second derivative changes sign)
            inflection_points = []
            for i in range(1, len(d2_disagreement)):
                if d2_disagreement[i-1] * d2_disagreement[i] < 0:
                    inflection_points.append(float(thresholds[i+1]))
            
            # Find maximum curvature
            curvature = np.abs(d2_disagreement)
            max_curvature_idx = np.argmax(curvature) if len(curvature) > 0 else None
            
            analysis[rule] = {
                'max_disagreement': float(np.max(disagreements)),
                'max_disagreement_threshold': float(thresholds[np.argmax(disagreements)]),
                'inflection_points': inflection_points,
                'max_curvature_threshold': float(thresholds[max_curvature_idx + 2]) if max_curvature_idx is not None else None,
                'mean_curvature': float(np.mean(np.abs(curvature))) if len(curvature) > 0 else 0.0,
            }
        
        return analysis
    
    def experiment_metric_interaction_matrix(
        self,
        n_profiles: int = 150
    ) -> Dict[str, Any]:
        """
        Experiment 2: Test all metric pairs systematically.
        
        Creates interaction matrix showing which pairs create
        the strongest heterogeneity effects.
        """
        print("=" * 80)
        print("EXPERIMENT 2: METRIC INTERACTION MATRIX")
        print("=" * 80)
        
        metrics = ['l1', 'l2', 'cosine', 'chebyshev']
        threshold = 0.5  # Fixed threshold
        voting_rules = ['plurality', 'borda', 'irv']
        
        interaction_matrix = {}
        
        for center_metric in metrics:
            for extreme_metric in metrics:
                if center_metric == extreme_metric:
                    continue
                
                pair_name = f"{center_metric}_{extreme_metric}"
                print(f"Testing {pair_name}...", end=" ", flush=True)
                
                # Heterogeneous
                config_het = SimulationConfig(
                    n_profiles=n_profiles,
                    n_voters=self.base_n_voters,
                    n_candidates=self.base_n_candidates,
                    voting_rules=voting_rules,
                    geometry=GeometryConfig(method='uniform', n_dim=2),
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
                    )
                )
                result_het = run_experiment(config_het, save_results=False, verbose=False)
                
                # Homogeneous (center)
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
                    )
                )
                result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                
                pair_data = {}
                for rule in voting_rules:
                    if rule in result_het.rule_results and rule in result_homo.rule_results:
                        het_result = result_het.rule_results[rule]
                        homo_result = result_homo.rule_results[rule]
                        
                        winners_het = np.array([m.winner_index for m in het_result.profile_metrics])
                        winners_homo = np.array([m.winner_index for m in homo_result.profile_metrics])
                        disagreement = np.mean(winners_het != winners_homo) * 100
                        
                        pair_data[rule] = {
                            'disagreement_rate': disagreement,
                            'vse_difference': (het_result.aggregate_metrics.vse_mean - 
                                             homo_result.aggregate_metrics.vse_mean),
                        }
                
                interaction_matrix[pair_name] = pair_data
                print("Done")
        
        # Analyze interaction patterns
        analysis = self._analyze_interactions(interaction_matrix)
        
        results = {
            'interaction_matrix': interaction_matrix,
            'analysis': analysis
        }
        
        # Save
        output_file = self.output_dir / "metric_interaction_matrix.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def _analyze_interactions(self, matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction matrix for patterns."""
        analysis = {
            'strongest_pairs': {},
            'weakest_pairs': {},
            'asymmetry_analysis': {}
        }
        
        for rule in ['plurality', 'borda', 'irv']:
            pairs_with_disagreement = []
            for pair_name, pair_data in matrix.items():
                if rule in pair_data:
                    pairs_with_disagreement.append((
                        pair_name,
                        pair_data[rule]['disagreement_rate']
                    ))
            
            if pairs_with_disagreement:
                pairs_with_disagreement.sort(key=lambda x: -x[1])
                analysis['strongest_pairs'][rule] = pairs_with_disagreement[:3]
                analysis['weakest_pairs'][rule] = pairs_with_disagreement[-3:]
        
        # Check for asymmetry (A_B vs B_A)
        for rule in ['plurality', 'borda', 'irv']:
            asymmetry_scores = []
            for center in ['l1', 'l2', 'cosine', 'chebyshev']:
                for extreme in ['l1', 'l2', 'cosine', 'chebyshev']:
                    if center == extreme:
                        continue
                    pair1 = f"{center}_{extreme}"
                    pair2 = f"{extreme}_{center}"
                    
                    if pair1 in matrix and pair2 in matrix:
                        if rule in matrix[pair1] and rule in matrix[pair2]:
                            d1 = matrix[pair1][rule]['disagreement_rate']
                            d2 = matrix[pair2][rule]['disagreement_rate']
                            asymmetry = abs(d1 - d2)
                            asymmetry_scores.append({
                                'pair1': pair1,
                                'pair2': pair2,
                                'asymmetry': asymmetry,
                                'd1': d1,
                                'd2': d2
                            })
            
            if asymmetry_scores:
                asymmetry_scores.sort(key=lambda x: -x['asymmetry'])
                analysis['asymmetry_analysis'][rule] = asymmetry_scores[:5]
        
        return analysis
    
    def experiment_dimensional_scaling(
        self,
        dimensions: List[int] = [1, 2, 3, 4, 5, 7, 10],
        n_profiles: int = 150
    ) -> Dict[str, Any]:
        """
        Experiment 3: How does heterogeneity effect scale with dimensionality?
        
        Tests hypothesis that effects increase with dimensions up to a point,
        then plateau or decrease.
        """
        print("=" * 80)
        print("EXPERIMENT 3: DIMENSIONAL SCALING")
        print("=" * 80)
        
        het_config = HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme',
            center_metric='l2',
            extreme_metric='cosine',
            extreme_threshold=0.5
        )
        
        voting_rules = ['plurality', 'borda', 'irv']
        
        results = {
            'dimensions': dimensions,
            'data': {}
        }
        
        for n_dim in dimensions:
            print(f"Testing {n_dim}D...", end=" ", flush=True)
            
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
                )
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous
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
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
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
            
            results['data'][n_dim] = dim_data
            print("Done")
        
        # Analyze scaling law
        analysis = self._analyze_scaling(results)
        results['analysis'] = analysis
        
        # Save
        output_file = self.output_dir / "dimensional_scaling.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def _analyze_scaling(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dimensional scaling patterns."""
        dimensions = np.array(results['dimensions'])
        analysis = {}
        
        for rule in ['plurality', 'borda', 'irv']:
            disagreements = []
            vse_diffs = []
            
            for dim in dimensions:
                if dim in results['data'] and rule in results['data'][dim]:
                    disagreements.append(results['data'][dim][rule]['disagreement_rate'])
                    vse_diffs.append(results['data'][dim][rule]['vse_difference'])
            
            if len(disagreements) < 3:
                continue
            
            disagreements = np.array(disagreements)
            vse_diffs = np.array(vse_diffs)
            
            # Find peak dimension
            peak_idx = np.argmax(disagreements)
            peak_dim = int(dimensions[peak_idx])
            
            # Test for power law: disagreement ~ dim^alpha
            # Use log-log regression
            log_dims = np.log(dimensions[:peak_idx+1] + 1)
            log_disagreements = np.log(disagreements[:peak_idx+1] + 1)
            
            if len(log_dims) > 1:
                # Linear regression in log space
                coeffs = np.polyfit(log_dims, log_disagreements, 1)
                alpha = coeffs[0]  # Power law exponent
            else:
                alpha = 0.0
            
            analysis[rule] = {
                'peak_dimension': peak_dim,
                'peak_disagreement': float(disagreements[peak_idx]),
                'scaling_exponent': float(alpha),
                'final_disagreement': float(disagreements[-1]),
                'scaling_ratio': float(disagreements[-1] / disagreements[0]) if disagreements[0] > 0 else 0.0
            }
        
        return analysis
    
    def experiment_preference_structure_changes(
        self,
        n_profiles: int = 200
    ) -> Dict[str, Any]:
        """
        Experiment 4: How does heterogeneity change preference structures?
        
        Analyzes:
        - Condorcet cycle rates
        - Preference transitivity
        - Utility distributions
        """
        print("=" * 80)
        print("EXPERIMENT 4: PREFERENCE STRUCTURE CHANGES")
        print("=" * 80)
        
        thresholds = [0.3, 0.5, 0.7]
        voting_rules = ['plurality', 'borda', 'irv', 'schulze']
        
        results = {
            'thresholds': thresholds,
            'data': {}
        }
        
        for threshold in thresholds:
            print(f"Threshold {threshold}...", end=" ", flush=True)
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric='l2',
                        extreme_metric='cosine',
                        extreme_threshold=threshold
                    )
                )
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous
            config_homo = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=self.base_n_voters,
                n_candidates=self.base_n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            threshold_data = {}
            for rule in voting_rules:
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    het_result = result_het.rule_results[rule]
                    homo_result = result_homo.rule_results[rule]
                    
                    threshold_data[rule] = {
                        'cycle_rate_het': het_result.aggregate_metrics.cycle_percentage,
                        'cycle_rate_homo': homo_result.aggregate_metrics.cycle_percentage,
                        'cycle_rate_change': (het_result.aggregate_metrics.cycle_percentage - 
                                            homo_result.aggregate_metrics.cycle_percentage),
                        'condorcet_efficiency_het': het_result.aggregate_metrics.condorcet_efficiency,
                        'condorcet_efficiency_homo': homo_result.aggregate_metrics.condorcet_efficiency,
                        'condorcet_efficiency_change': (het_result.aggregate_metrics.condorcet_efficiency - 
                                                       homo_result.aggregate_metrics.condorcet_efficiency),
                    }
            
            results['data'][f"{threshold}"] = threshold_data
            print("Done")
        
        # Save
        output_file = self.output_dir / "preference_structure_changes.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all deep research experiments."""
        print("=" * 80)
        print("DEEP RESEARCH - ALL EXPERIMENTS")
        print("=" * 80)
        print()
        
        all_results = {}
        
        # Experiment 1: Threshold non-linearity
        all_results['threshold_nonlinearity'] = self.experiment_threshold_nonlinearity()
        
        # Experiment 2: Metric interactions
        all_results['metric_interactions'] = self.experiment_metric_interaction_matrix()
        
        # Experiment 3: Dimensional scaling
        all_results['dimensional_scaling'] = self.experiment_dimensional_scaling()
        
        # Experiment 4: Preference structure
        all_results['preference_structure'] = self.experiment_preference_structure_changes()
        
        # Save combined
        output_file = self.output_dir / "deep_research_all.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print(f"Results saved to: {output_file}")
        print("=" * 80)
        
        return all_results


if __name__ == '__main__':
    researcher = DeepResearcher()
    researcher.run_all_experiments()


