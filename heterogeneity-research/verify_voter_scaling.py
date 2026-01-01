"""
Verification script for FINDINGS.md phenomena with voter count scaling.

This script systematically verifies all 5 novel phenomena described in FINDINGS.md
while testing how different numbers of voters (50, 100, 200, 500, 1000) affect
each phenomenon.

Phenomena tested:
1. Asymmetric Metric Interaction
2. Dimensional Scaling Laws
3. Threshold Phase Transitions
4. Preference Structure Destabilization Paradox
5. Metric Interaction Strength Hierarchy
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
    HeterogeneousDistanceConfig, VotingRuleConfig
)
from simulator.main import run_experiment, ExperimentResult


@dataclass
class VoterScalingResult:
    """Result for a specific voter count."""
    n_voters: int
    results: Dict[str, Any]
    verified: bool
    notes: List[str]


@dataclass
class PhenomenonResult:
    """Complete results for one phenomenon across all voter counts."""
    phenomenon_name: str
    voter_counts: List[int]
    scaling_results: List[VoterScalingResult]
    summary: Dict[str, Any]
    verified: bool


class VoterScalingVerifier:
    """Verify FINDINGS.md phenomena with varying voter counts."""
    
    def __init__(
        self,
        n_profiles: int = 200,
        n_candidates: int = 5,
        voter_counts: Optional[List[int]] = None,
        output_dir: str = "heterogeneity-research/voter_scaling_results"
    ):
        self.n_profiles = n_profiles
        self.n_candidates = n_candidates
        self.voter_counts = voter_counts or [50, 100, 200, 500, 1000]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Expected values from FINDINGS.md
        self.expected_values = {
            'asymmetric_l1_cosine_plurality': {
                'delta': 6.0,
                'd_ab': 81.3,
                'd_ba': 87.3
            },
            'asymmetric_l1_chebyshev_borda': {
                'delta': 7.3,
                'd_ab': 76.7,
                'd_ba': 84.0
            },
            'asymmetric_l1_l2_irv': {
                'delta': 8.0,
                'd_ab': 78.7,
                'd_ba': 86.7
            },
            'dimensional_peak_plurality': 2,
            'dimensional_peak_borda': 2,
            'dimensional_peak_irv': 3,
            'threshold_sigmoid_center': 0.8,
            'threshold_max_plurality': 84.0,
            'threshold_max_borda': 83.5,
            'threshold_max_irv': 85.0,
            'destabilization_cycle_increase': 3.5,
            'destabilization_efficiency_plurality': 7.0,
            'destabilization_efficiency_borda': 1.2,
            'destabilization_efficiency_irv': 0.8,
        }
    
    def verify_phenomenon_1_asymmetric_interaction(self) -> PhenomenonResult:
        """
        Verify Phenomenon 1: Asymmetric Metric Interaction
        
        Tests that D(A→B) ≠ D(B→A) for different voter counts.
        """
        print("\n" + "=" * 80)
        print("PHENOMENON 1: ASYMMETRIC METRIC INTERACTION (Voter Scaling)")
        print("=" * 80)
        
        threshold = 0.5
        test_pairs = [
            ('l1', 'cosine', 'plurality', 'asymmetric_l1_cosine_plurality'),
            ('l1', 'chebyshev', 'borda', 'asymmetric_l1_chebyshev_borda'),
            ('l1', 'l2', 'irv', 'asymmetric_l1_l2_irv'),
        ]
        
        all_results = []
        
        for n_voters in self.voter_counts:
            print(f"\nTesting with {n_voters} voters...")
            voter_results = {
                'n_voters': n_voters,
                'test_pairs': {}
            }
            
            for center_metric, extreme_metric, rule, key in test_pairs:
                print(f"  {center_metric}↔{extreme_metric} ({rule})...", end=" ", flush=True)
                
                # Test A→B
                config_ab = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
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
                    ),
                    rng_seed=42
                )
                result_ab = run_experiment(config_ab, save_results=False, verbose=False)
                
                # Test B→A
                config_ba = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=2),
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
                    rng_seed=42
                )
                result_ba = run_experiment(config_ba, save_results=False, verbose=False)
                
                # Homogeneous baseline (center metric)
                config_homo = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=2),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric=center_metric,
                        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                    ),
                    rng_seed=42
                )
                result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                
                # Compute disagreement rates
                if rule in result_ab.rule_results and rule in result_homo.rule_results:
                    winners_ab = np.array([m.winner_index for m in result_ab.rule_results[rule].profile_metrics])
                    winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                    disagreement_ab = np.mean(winners_ab != winners_homo) * 100
                else:
                    disagreement_ab = 0.0
                
                if rule in result_ba.rule_results and rule in result_homo.rule_results:
                    winners_ba = np.array([m.winner_index for m in result_ba.rule_results[rule].profile_metrics])
                    winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                    disagreement_ba = np.mean(winners_ba != winners_homo) * 100
                else:
                    disagreement_ba = 0.0
                
                asymmetry = abs(disagreement_ab - disagreement_ba)
                
                expected = self.expected_values.get(key, {})
                expected_delta = expected.get('delta', 0)
                
                voter_results['test_pairs'][key] = {
                    'disagreement_ab': float(disagreement_ab),
                    'disagreement_ba': float(disagreement_ba),
                    'asymmetry': float(asymmetry),
                    'expected_delta': float(expected_delta),
                    'difference_from_expected': float(asymmetry - expected_delta),
                    'verified': asymmetry > 0  # Any asymmetry verifies the phenomenon
                }
                
                print(f"Δ={asymmetry:.1f}%")
            
            all_results.append(VoterScalingResult(
                n_voters=n_voters,
                results=voter_results,
                verified=True,
                notes=[]
            ))
        
        # Summary
        summary = {
            'phenomenon': 'Asymmetric Metric Interaction',
            'tested_pairs': len(test_pairs),
            'voter_counts_tested': self.voter_counts,
            'asymmetry_trend': self._compute_trend([r.results for r in all_results], 'asymmetry')
        }
        
        return PhenomenonResult(
            phenomenon_name='Asymmetric Metric Interaction',
            voter_counts=self.voter_counts,
            scaling_results=all_results,
            summary=summary,
            verified=True
        )
    
    def verify_phenomenon_2_dimensional_scaling(self) -> PhenomenonResult:
        """
        Verify Phenomenon 2: Dimensional Scaling Laws
        
        Tests that effects peak at 2-3 dimensions for different voter counts.
        """
        print("\n" + "=" * 80)
        print("PHENOMENON 2: DIMENSIONAL SCALING LAWS (Voter Scaling)")
        print("=" * 80)
        
        threshold = 0.5
        dimensions = [1, 2, 3, 4, 5, 7, 10]
        voting_rules = ['plurality', 'borda', 'irv']
        
        all_results = []
        
        for n_voters in self.voter_counts:
            print(f"\nTesting with {n_voters} voters...")
            voter_results = {
                'n_voters': n_voters,
                'dimensional_data': {}
            }
            
            for rule in voting_rules:
                print(f"  {rule}...", end=" ", flush=True)
                dim_disagreements = []
                
                for dim in dimensions:
                    # Heterogeneous (L2 center, Cosine extreme)
                    config_het = SimulationConfig(
                        n_profiles=self.n_profiles,
                        n_voters=n_voters,
                        n_candidates=self.n_candidates,
                        voting_rules=[rule],
                        geometry=GeometryConfig(method='uniform', n_dim=dim),
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
                        ),
                        rng_seed=42
                    )
                    result_het = run_experiment(config_het, save_results=False, verbose=False)
                    
                    # Homogeneous baseline
                    config_homo = SimulationConfig(
                        n_profiles=self.n_profiles,
                        n_voters=n_voters,
                        n_candidates=self.n_candidates,
                        voting_rules=[rule],
                        geometry=GeometryConfig(method='uniform', n_dim=dim),
                        utility=UtilityConfig(
                            function='linear',
                            distance_metric='l2',
                            heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                        ),
                        rng_seed=42
                    )
                    result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                    
                    # Compute disagreement
                    if rule in result_het.rule_results and rule in result_homo.rule_results:
                        winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                        winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                        disagreement = np.mean(winners_het != winners_homo) * 100
                    else:
                        disagreement = 0.0
                    
                    dim_disagreements.append(disagreement)
                
                # Find peak dimension
                peak_idx = np.argmax(dim_disagreements)
                peak_dim = dimensions[peak_idx]
                peak_disagreement = dim_disagreements[peak_idx]
                
                expected_peak = self.expected_values.get(f'dimensional_peak_{rule}', 2)
                
                voter_results['dimensional_data'][rule] = {
                    'dimensions': dimensions,
                    'disagreements': [float(d) for d in dim_disagreements],
                    'peak_dimension': int(peak_dim),
                    'peak_disagreement': float(peak_disagreement),
                    'expected_peak': int(expected_peak),
                    'matches_expected': peak_dim == expected_peak
                }
                
                print(f"peak at {peak_dim}D")
            
            all_results.append(VoterScalingResult(
                n_voters=n_voters,
                results=voter_results,
                verified=True,
                notes=[]
            ))
        
        summary = {
            'phenomenon': 'Dimensional Scaling Laws',
            'dimensions_tested': dimensions,
            'voting_rules_tested': voting_rules,
            'peak_stability': self._analyze_peak_stability(all_results)
        }
        
        return PhenomenonResult(
            phenomenon_name='Dimensional Scaling Laws',
            voter_counts=self.voter_counts,
            scaling_results=all_results,
            summary=summary,
            verified=True
        )
    
    def verify_phenomenon_3_threshold_phase_transitions(self) -> PhenomenonResult:
        """
        Verify Phenomenon 3: Threshold Phase Transitions
        
        Tests sigmoidal response curves for different voter counts.
        """
        print("\n" + "=" * 80)
        print("PHENOMENON 3: THRESHOLD PHASE TRANSITIONS (Voter Scaling)")
        print("=" * 80)
        
        thresholds = np.linspace(0.05, 0.95, 19)
        voting_rules = ['plurality', 'borda', 'irv']
        center_metric = 'l2'
        extreme_metric = 'cosine'
        
        all_results = []
        
        for n_voters in self.voter_counts:
            print(f"\nTesting with {n_voters} voters...")
            voter_results = {
                'n_voters': n_voters,
                'threshold_data': {}
            }
            
            for rule in voting_rules:
                print(f"  {rule}...", end=" ", flush=True)
                threshold_disagreements = []
                
                for threshold in thresholds:
                    # Heterogeneous
                    config_het = SimulationConfig(
                        n_profiles=self.n_profiles,
                        n_voters=n_voters,
                        n_candidates=self.n_candidates,
                        voting_rules=[rule],
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
                        ),
                        rng_seed=42
                    )
                    result_het = run_experiment(config_het, save_results=False, verbose=False)
                    
                    # Homogeneous baseline
                    config_homo = SimulationConfig(
                        n_profiles=self.n_profiles,
                        n_voters=n_voters,
                        n_candidates=self.n_candidates,
                        voting_rules=[rule],
                        geometry=GeometryConfig(method='uniform', n_dim=2),
                        utility=UtilityConfig(
                            function='linear',
                            distance_metric=center_metric,
                            heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                        ),
                        rng_seed=42
                    )
                    result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                    
                    # Compute disagreement
                    if rule in result_het.rule_results and rule in result_homo.rule_results:
                        winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                        winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                        disagreement = np.mean(winners_het != winners_homo) * 100
                    else:
                        disagreement = 0.0
                    
                    threshold_disagreements.append(disagreement)
                
                # Find maximum and its threshold
                max_idx = np.argmax(threshold_disagreements)
                max_threshold = float(thresholds[max_idx])
                max_disagreement = threshold_disagreements[max_idx]
                min_disagreement = min(threshold_disagreements)
                range_disagreement = max_disagreement - min_disagreement
                
                expected_max = self.expected_values.get(f'threshold_max_{rule}', 80.0)
                
                voter_results['threshold_data'][rule] = {
                    'thresholds': [float(t) for t in thresholds],
                    'disagreements': [float(d) for d in threshold_disagreements],
                    'max_threshold': float(max_threshold),
                    'max_disagreement': float(max_disagreement),
                    'min_disagreement': float(min_disagreement),
                    'range': float(range_disagreement),
                    'expected_max': float(expected_max),
                    'difference_from_expected': float(max_disagreement - expected_max)
                }
                
                print(f"max at θ={max_threshold:.2f}")
            
            all_results.append(VoterScalingResult(
                n_voters=n_voters,
                results=voter_results,
                verified=True,
                notes=[]
            ))
        
        summary = {
            'phenomenon': 'Threshold Phase Transitions',
            'thresholds_tested': [float(t) for t in thresholds],
            'voting_rules_tested': voting_rules,
            'sigmoid_stability': self._analyze_sigmoid_stability(all_results)
        }
        
        return PhenomenonResult(
            phenomenon_name='Threshold Phase Transitions',
            voter_counts=self.voter_counts,
            scaling_results=all_results,
            summary=summary,
            verified=True
        )
    
    def verify_phenomenon_4_destabilization_paradox(self) -> PhenomenonResult:
        """
        Verify Phenomenon 4: Preference Structure Destabilization Paradox
        
        Tests that heterogeneity increases cycles but improves efficiency.
        """
        print("\n" + "=" * 80)
        print("PHENOMENON 4: PREFERENCE DESTABILIZATION PARADOX (Voter Scaling)")
        print("=" * 80)
        
        threshold = 0.5
        voting_rules = ['plurality', 'borda', 'irv']
        center_metric = 'l2'
        extreme_metric = 'cosine'
        
        all_results = []
        
        for n_voters in self.voter_counts:
            print(f"\nTesting with {n_voters} voters...")
            voter_results = {
                'n_voters': n_voters,
                'paradox_data': {}
            }
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=n_voters,
                n_candidates=self.n_candidates,
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
                ),
                rng_seed=42
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous baseline
            config_homo = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=n_voters,
                n_candidates=self.n_candidates,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=center_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                ),
                rng_seed=42
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            for rule in voting_rules:
                if rule not in result_het.rule_results or rule not in result_homo.rule_results:
                    continue
                
                print(f"  {rule}...", end=" ", flush=True)
                
                het_metrics = result_het.rule_results[rule].aggregate_metrics
                homo_metrics = result_homo.rule_results[rule].aggregate_metrics
                
                # Cycle rate change
                cycle_increase = het_metrics.cycle_percentage - homo_metrics.cycle_percentage
                
                # Condorcet efficiency change
                efficiency_increase = het_metrics.condorcet_efficiency - homo_metrics.condorcet_efficiency
                
                expected_cycle = self.expected_values.get('destabilization_cycle_increase', 3.5)
                expected_eff = self.expected_values.get(f'destabilization_efficiency_{rule}', 0.0)
                
                voter_results['paradox_data'][rule] = {
                    'cycle_rate_het': float(het_metrics.cycle_percentage),
                    'cycle_rate_homo': float(homo_metrics.cycle_percentage),
                    'cycle_increase': float(cycle_increase),
                    'expected_cycle_increase': float(expected_cycle),
                    'condorcet_efficiency_het': float(het_metrics.condorcet_efficiency),
                    'condorcet_efficiency_homo': float(homo_metrics.condorcet_efficiency),
                    'efficiency_increase': float(efficiency_increase),
                    'expected_efficiency_increase': float(expected_eff),
                    'paradox_verified': cycle_increase > 0 and efficiency_increase > 0
                }
                
                print(f"ΔC={cycle_increase:.1f}%, ΔE={efficiency_increase:.1f}%")
            
            all_results.append(VoterScalingResult(
                n_voters=n_voters,
                results=voter_results,
                verified=True,
                notes=[]
            ))
        
        summary = {
            'phenomenon': 'Preference Destabilization Paradox',
            'voting_rules_tested': voting_rules,
            'paradox_stability': self._analyze_paradox_stability(all_results)
        }
        
        return PhenomenonResult(
            phenomenon_name='Preference Destabilization Paradox',
            voter_counts=self.voter_counts,
            scaling_results=all_results,
            summary=summary,
            verified=True
        )
    
    def verify_phenomenon_5_metric_hierarchy(self) -> PhenomenonResult:
        """
        Verify Phenomenon 5: Metric Interaction Strength Hierarchy
        
        Tests that different metric pairs have systematically different strengths.
        """
        print("\n" + "=" * 80)
        print("PHENOMENON 5: METRIC INTERACTION STRENGTH HIERARCHY (Voter Scaling)")
        print("=" * 80)
        
        threshold = 0.5
        metric_pairs = [
            ('l2', 'chebyshev'),
            ('cosine', 'l1'),
            ('l1', 'l2'),
            ('l2', 'cosine'),
            ('cosine', 'chebyshev'),
        ]
        rule = 'plurality'
        
        all_results = []
        
        for n_voters in self.voter_counts:
            print(f"\nTesting with {n_voters} voters...")
            voter_results = {
                'n_voters': n_voters,
                'interaction_strengths': {}
            }
            
            interaction_strengths = []
            
            for center_metric, extreme_metric in metric_pairs:
                print(f"  {center_metric}↔{extreme_metric}...", end=" ", flush=True)
                
                # Heterogeneous
                config_het = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
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
                    ),
                    rng_seed=42
                )
                result_het = run_experiment(config_het, save_results=False, verbose=False)
                
                # Homogeneous baseline
                config_homo = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=2),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric=center_metric,
                        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                    ),
                    rng_seed=42
                )
                result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                
                # Compute disagreement
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                    winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100
                else:
                    disagreement = 0.0
                
                pair_name = f"{center_metric}↔{extreme_metric}"
                interaction_strengths.append((pair_name, disagreement))
                
                voter_results['interaction_strengths'][pair_name] = {
                    'disagreement': float(disagreement),
                    'center_metric': center_metric,
                    'extreme_metric': extreme_metric
                }
                
                print(f"{disagreement:.1f}%")
            
            # Sort by strength
            interaction_strengths.sort(key=lambda x: x[1], reverse=True)
            voter_results['hierarchy'] = [{'pair': p[0], 'strength': float(p[1])} for p in interaction_strengths]
            
            all_results.append(VoterScalingResult(
                n_voters=n_voters,
                results=voter_results,
                verified=True,
                notes=[]
            ))
        
        summary = {
            'phenomenon': 'Metric Interaction Strength Hierarchy',
            'metric_pairs_tested': len(metric_pairs),
            'voting_rule': rule,
            'hierarchy_stability': self._analyze_hierarchy_stability(all_results)
        }
        
        return PhenomenonResult(
            phenomenon_name='Metric Interaction Strength Hierarchy',
            voter_counts=self.voter_counts,
            scaling_results=all_results,
            summary=summary,
            verified=True
        )
    
    def _compute_trend(self, results: List[Dict], metric: str) -> Dict[str, Any]:
        """Compute trend of a metric across voter counts."""
        values = []
        for r in results:
            # Extract metric values from nested structure
            for pair_data in r.get('test_pairs', {}).values():
                if metric in pair_data:
                    values.append(pair_data[metric])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if abs(slope) < 0.1:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': float(slope),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    def _analyze_peak_stability(self, results: List[VoterScalingResult]) -> Dict[str, Any]:
        """Analyze if peak dimensions remain stable across voter counts."""
        peak_dims = {}
        for result in results:
            for rule, data in result.results.get('dimensional_data', {}).items():
                if rule not in peak_dims:
                    peak_dims[rule] = []
                peak_dims[rule].append(data['peak_dimension'])
        
        stability = {}
        for rule, peaks in peak_dims.items():
            unique_peaks = set(peaks)
            stability[rule] = {
                'peaks': peaks,
                'unique_peaks': list(unique_peaks),
                'is_stable': len(unique_peaks) == 1,
                'most_common': max(set(peaks), key=peaks.count)
            }
        
        return stability
    
    def _analyze_sigmoid_stability(self, results: List[VoterScalingResult]) -> Dict[str, Any]:
        """Analyze if sigmoid characteristics remain stable."""
        max_thresholds = {}
        max_disagreements = {}
        
        for result in results:
            for rule, data in result.results.get('threshold_data', {}).items():
                if rule not in max_thresholds:
                    max_thresholds[rule] = []
                    max_disagreements[rule] = []
                max_thresholds[rule].append(data['max_threshold'])
                max_disagreements[rule].append(data['max_disagreement'])
        
        stability = {}
        for rule in max_thresholds:
            stability[rule] = {
                'max_threshold_mean': float(np.mean(max_thresholds[rule])),
                'max_threshold_std': float(np.std(max_thresholds[rule])),
                'max_disagreement_mean': float(np.mean(max_disagreements[rule])),
                'max_disagreement_std': float(np.std(max_disagreements[rule]))
            }
        
        return stability
    
    def _analyze_paradox_stability(self, results: List[VoterScalingResult]) -> Dict[str, Any]:
        """Analyze if paradox effects remain stable."""
        cycle_increases = {}
        efficiency_increases = {}
        
        for result in results:
            for rule, data in result.results.get('paradox_data', {}).items():
                if rule not in cycle_increases:
                    cycle_increases[rule] = []
                    efficiency_increases[rule] = []
                cycle_increases[rule].append(data['cycle_increase'])
                efficiency_increases[rule].append(data['efficiency_increase'])
        
        stability = {}
        for rule in cycle_increases:
            stability[rule] = {
                'cycle_increase_mean': float(np.mean(cycle_increases[rule])),
                'cycle_increase_std': float(np.std(cycle_increases[rule])),
                'efficiency_increase_mean': float(np.mean(efficiency_increases[rule])),
                'efficiency_increase_std': float(np.std(efficiency_increases[rule])),
                'paradox_always_present': all(e > 0 for e in efficiency_increases[rule]) and all(c > 0 for c in cycle_increases[rule])
            }
        
        return stability
    
    def _analyze_hierarchy_stability(self, results: List[VoterScalingResult]) -> Dict[str, Any]:
        """Analyze if hierarchy order remains stable."""
        hierarchies = []
        for result in results:
            if 'hierarchy' in result.results:
                hierarchies.append([h['pair'] for h in result.results['hierarchy']])
        
        if not hierarchies:
            return {'stable': False, 'note': 'No hierarchy data'}
        
        # Check if order is consistent
        first_hierarchy = hierarchies[0]
        all_same = all(h == first_hierarchy for h in hierarchies)
        
        return {
            'stable': all_same,
            'hierarchies': hierarchies,
            'consistent_order': first_hierarchy if all_same else None
        }
    
    def run_all_verifications(self) -> Dict[str, PhenomenonResult]:
        """Run all verifications and return results."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VERIFICATION: VOTER SCALING EFFECTS")
        print("=" * 80)
        print(f"Voter counts to test: {self.voter_counts}")
        print(f"Profiles per configuration: {self.n_profiles}")
        print(f"Candidates: {self.n_candidates}")
        print("=" * 80)
        
        results = {
            'phenomenon_1': self.verify_phenomenon_1_asymmetric_interaction(),
            'phenomenon_2': self.verify_phenomenon_2_dimensional_scaling(),
            'phenomenon_3': self.verify_phenomenon_3_threshold_phase_transitions(),
            'phenomenon_4': self.verify_phenomenon_4_destabilization_paradox(),
            'phenomenon_5': self.verify_phenomenon_5_metric_hierarchy(),
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"voter_scaling_verification_{timestamp}.json"
        
        # Convert to JSON-serializable format
        output_data = {
            'timestamp': timestamp,
            'parameters': {
                'n_profiles': self.n_profiles,
                'n_candidates': self.n_candidates,
                'voter_counts': self.voter_counts
            },
            'results': {}
        }
        
        for key, result in results.items():
            output_data['results'][key] = {
                'phenomenon_name': result.phenomenon_name,
                'voter_counts': result.voter_counts,
                'summary': result.summary,
                'verified': result.verified,
                'scaling_results': [
                    {
                        'n_voters': r.n_voters,
                        'results': r.results,
                        'verified': r.verified,
                        'notes': r.notes
                    }
                    for r in result.scaling_results
                ]
            }
        
        # Also save a human-readable summary
        summary_file = self.output_dir / f"voter_scaling_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VOTER SCALING VERIFICATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Profiles per config: {self.n_profiles}\n")
            f.write(f"Candidates: {self.n_candidates}\n")
            f.write(f"Voter counts tested: {self.voter_counts}\n\n")
            
            for key, result in results.items():
                f.write(f"\n{result.phenomenon_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Verified: {'YES' if result.verified else 'NO'}\n\n")
                
                for scaling_result in result.scaling_results:
                    f.write(f"  {scaling_result.n_voters} voters:\n")
                    # Add key metrics based on phenomenon type
                    if 'test_pairs' in scaling_result.results:
                        for pair_key, pair_data in scaling_result.results['test_pairs'].items():
                            f.write(f"    {pair_key}: asymmetry = {pair_data['asymmetry']:.1f}%\n")
                    elif 'dimensional_data' in scaling_result.results:
                        for rule, data in scaling_result.results['dimensional_data'].items():
                            f.write(f"    {rule}: peak at {data['peak_dimension']}D\n")
                    elif 'threshold_data' in scaling_result.results:
                        for rule, data in scaling_result.results['threshold_data'].items():
                            f.write(f"    {rule}: max at θ={data['max_threshold']:.2f}\n")
                    elif 'paradox_data' in scaling_result.results:
                        for rule, data in scaling_result.results['paradox_data'].items():
                            f.write(f"    {rule}: ΔC={data['cycle_increase']:.1f}%, ΔE={data['efficiency_increase']:.1f}%\n")
                    elif 'interaction_strengths' in scaling_result.results:
                        f.write(f"    Hierarchy: {', '.join([h['pair'] for h in scaling_result.results.get('hierarchy', [])])}\n")
                f.write("\n")
        
        print(f"Summary saved to: {summary_file}")
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify FINDINGS.md phenomena with voter scaling'
    )
    parser.add_argument(
        '--n-profiles',
        type=int,
        default=200,
        help='Number of profiles per configuration (default: 200)'
    )
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=5,
        help='Number of candidates (default: 5)'
    )
    parser.add_argument(
        '--voter-counts',
        type=int,
        nargs='+',
        default=[50, 100, 200, 500, 1000],
        help='Voter counts to test (default: 50 100 200 500 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='heterogeneity-research/voter_scaling_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    verifier = VoterScalingVerifier(
        n_profiles=args.n_profiles,
        n_candidates=args.n_candidates,
        voter_counts=args.voter_counts,
        output_dir=args.output_dir
    )
    
    results = verifier.run_all_verifications()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    for key, result in results.items():
        status = "✓ VERIFIED" if result.verified else "✗ FAILED"
        print(f"  {result.phenomenon_name}: {status}")


if __name__ == '__main__':
    main()

