"""
Experimental verification of findings documented in FINDINGS.md.

This script systematically tests each of the 5 novel phenomena to verify
they actually occur in simulations and compares results with documented values.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
class VerificationResult:
    """Result of verifying a specific finding."""
    finding_name: str
    verified: bool
    expected_values: Dict[str, Any]
    observed_values: Dict[str, Any]
    differences: Dict[str, float]
    notes: List[str]


class FindingsVerifier:
    """Verify findings from FINDINGS.md through experimental testing."""
    
    def __init__(
        self,
        n_profiles: int = 200,
        n_voters: int = 100,
        n_candidates: int = 5,
        output_dir: str = "heterogeneity-research/verification_results"
    ):
        self.n_profiles = n_profiles
        self.n_voters = n_voters
        self.n_candidates = n_candidates
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    def verify_finding_1_asymmetric_interaction(self) -> VerificationResult:
        """
        Verify Finding 1: Asymmetric Metric Interaction
        
        Tests that D(A→B) ≠ D(B→A) for metric pairs.
        Expected asymmetries from FINDINGS.md:
        - L1 ↔ Cosine (Plurality): Δ = 6.0% (81.3% vs 87.3%)
        - L1 ↔ Chebyshev (Borda): Δ = 7.3% (76.7% vs 84.0%)
        - L1 ↔ L2 (IRV): Δ = 8.0% (78.7% vs 86.7%)
        """
        print("\n" + "=" * 80)
        print("VERIFYING FINDING 1: ASYMMETRIC METRIC INTERACTION")
        print("=" * 80)
        
        threshold = 0.5
        voting_rules = ['plurality', 'borda', 'irv']
        test_pairs = [
            ('l1', 'cosine', 'plurality', 6.0, 81.3, 87.3),
            ('l1', 'chebyshev', 'borda', 7.3, 76.7, 84.0),
            ('l1', 'l2', 'irv', 8.0, 78.7, 86.7),
        ]
        
        results = {}
        verified_pairs = []
        failed_pairs = []
        
        for center_metric, extreme_metric, rule, expected_delta, expected_d1, expected_d2 in test_pairs:
            print(f"\nTesting {center_metric}->{extreme_metric} vs {extreme_metric}->{center_metric} ({rule})...")
            
            # Test A→B
            config_ab = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
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
                )
            )
            result_ab = run_experiment(config_ab, save_results=False, verbose=False)
            
            # Test B→A
            config_ba = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
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
                )
            )
            result_ba = run_experiment(config_ba, save_results=False, verbose=False)
            
            # Homogeneous baseline (center metric)
            config_homo = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
                n_candidates=self.n_candidates,
                voting_rules=[rule],
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=center_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
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
            
            results[f"{center_metric}_{extreme_metric}"] = {
                'disagreement_ab': disagreement_ab,
                'disagreement_ba': disagreement_ba,
                'asymmetry': asymmetry,
                'expected_delta': expected_delta,
                'expected_d1': expected_d1,
                'expected_d2': expected_d2
            }
            
            # Check if asymmetry exists and is in reasonable range
            tolerance = 3.0  # Allow 3% tolerance
            if asymmetry > 1.0:  # Asymmetry exists
                if abs(asymmetry - expected_delta) < tolerance:
                    verified_pairs.append(f"{center_metric}<->{extreme_metric} ({rule})")
                    print(f"  [OK] Verified: Delta = {asymmetry:.1f}% (expected {expected_delta:.1f}%)")
                else:
                    failed_pairs.append(f"{center_metric}<->{extreme_metric} ({rule}): Delta = {asymmetry:.1f}% (expected {expected_delta:.1f}%)")
                    print(f"  [FAIL] Mismatch: Delta = {asymmetry:.1f}% (expected {expected_delta:.1f}%)")
            else:
                failed_pairs.append(f"{center_metric}<->{extreme_metric} ({rule}): No significant asymmetry (Delta = {asymmetry:.1f}%)")
                print(f"  [FAIL] No asymmetry detected: Delta = {asymmetry:.1f}%")
        
        # Overall verification
        verified = len(verified_pairs) >= 2  # At least 2 out of 3 should match
        
        return VerificationResult(
            finding_name="Asymmetric Metric Interaction",
            verified=verified,
            expected_values={
                'mean_asymmetry': 4.0,
                'std_asymmetry': 1.7,
                'test_pairs': test_pairs
            },
            observed_values=results,
            differences={},
            notes=[
                f"Verified pairs: {len(verified_pairs)}/{len(test_pairs)}",
                f"Failed pairs: {len(failed_pairs)}",
                *[f"  - {p}" for p in failed_pairs]
            ]
        )
    
    def verify_finding_2_dimensional_scaling(self) -> VerificationResult:
        """
        Verify Finding 2: Dimensional Scaling Laws
        
        Tests that effects peak at 2-3 dimensions and follow power law.
        Expected from FINDINGS.md:
        - Plurality: peak at d=2, α=0.263, D(2)=84.7%
        - Borda: peak at d=2, α=0.099, D(2)=83.3%
        - IRV: peak at d=3, α=0.008, D(3)=82.0%
        """
        print("\n" + "=" * 80)
        print("VERIFYING FINDING 2: DIMENSIONAL SCALING LAWS")
        print("=" * 80)
        
        dimensions = [1, 2, 3, 4, 5, 7, 10]
        voting_rules = ['plurality', 'borda', 'irv']
        threshold = 0.5
        
        results = {}
        verified_rules = []
        failed_rules = []
        
        for rule in voting_rules:
            print(f"\nTesting {rule}...")
            disagreements_by_dim = {}
            
            for n_dim in dimensions:
                print(f"  Dimension {n_dim}D...", end=" ", flush=True)
                
                # Heterogeneous
                config_het = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=self.n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=n_dim),
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
                    n_profiles=self.n_profiles,
                    n_voters=self.n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=n_dim),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric='l2',
                        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                    )
                )
                result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                
                # Compute disagreement
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                    winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100
                else:
                    disagreement = 0.0
                
                disagreements_by_dim[n_dim] = disagreement
                print(f"D = {disagreement:.1f}%")
            
            # Analyze scaling
            dims_array = np.array(dimensions)
            disagrees_array = np.array([disagreements_by_dim[d] for d in dimensions])
            
            # Find peak
            peak_idx = np.argmax(disagrees_array)
            peak_dim = int(dims_array[peak_idx])
            peak_disagreement = disagrees_array[peak_idx]
            
            # Fit power law up to peak
            log_dims = np.log(dims_array[:peak_idx+1] + 1)
            log_disagrees = np.log(disagrees_array[:peak_idx+1] + 1)
            
            if len(log_dims) > 1:
                coeffs = np.polyfit(log_dims, log_disagrees, 1)
                alpha = coeffs[0]
                r_squared = np.corrcoef(log_dims, log_disagrees)[0,1]**2
            else:
                alpha = 0.0
                r_squared = 0.0
            
            # Expected values
            if rule == 'plurality':
                expected_peak_dim = 2
                expected_peak_d = 84.7
                expected_alpha = 0.263
            elif rule == 'borda':
                expected_peak_dim = 2
                expected_peak_d = 83.3
                expected_alpha = 0.099
            else:  # irv
                expected_peak_dim = 3
                expected_peak_d = 82.0
                expected_alpha = 0.008
            
            results[rule] = {
                'peak_dimension': peak_dim,
                'peak_disagreement': peak_disagreement,
                'scaling_exponent': alpha,
                'r_squared': r_squared,
                'disagreements_by_dim': disagreements_by_dim,
                'expected_peak_dim': expected_peak_dim,
                'expected_peak_d': expected_peak_d,
                'expected_alpha': expected_alpha
            }
            
            # Verify
            peak_match = peak_dim == expected_peak_dim
            peak_d_close = abs(peak_disagreement - expected_peak_d) < 5.0  # 5% tolerance
            alpha_close = abs(alpha - expected_alpha) < 0.1  # 0.1 tolerance for exponent
            
            if peak_match and peak_d_close:
                verified_rules.append(rule)
                print(f"  [OK] Verified: peak at d={peak_dim} (D={peak_disagreement:.1f}%), alpha={alpha:.3f}")
            else:
                failed_rules.append(f"{rule}: peak at d={peak_dim} (expected {expected_peak_dim}), D={peak_disagreement:.1f}% (expected {expected_peak_d:.1f}%)")
                print(f"  [FAIL] Mismatch: peak at d={peak_dim} (expected {expected_peak_dim}), D={peak_disagreement:.1f}% (expected {expected_peak_d:.1f}%)")
        
        verified = len(verified_rules) >= 2  # At least 2 out of 3 should match
        
        return VerificationResult(
            finding_name="Dimensional Scaling Laws",
            verified=verified,
            expected_values={
                'plurality': {'peak_dim': 2, 'peak_d': 84.7, 'alpha': 0.263},
                'borda': {'peak_dim': 2, 'peak_d': 83.3, 'alpha': 0.099},
                'irv': {'peak_dim': 3, 'peak_d': 82.0, 'alpha': 0.008}
            },
            observed_values=results,
            differences={},
            notes=[
                f"Verified rules: {len(verified_rules)}/{len(voting_rules)}",
                *[f"  - {r}" for r in failed_rules]
            ]
        )
    
    def verify_finding_3_threshold_phase_transitions(self) -> VerificationResult:
        """
        Verify Finding 3: Threshold Phase Transitions
        
        Tests sigmoidal response curves with critical thresholds.
        Expected from FINDINGS.md:
        - Plurality: theta_0=0.8, k=5.0, D_max=84.0% at theta=0.6, D_min=70.5%
        - Borda: theta_0=0.8, k=5.0, D_max=83.5% at theta=0.7
        - IRV: D_max=85.0% at theta=0.15 (early peak)
        """
        print("\n" + "=" * 80)
        print("VERIFYING FINDING 3: THRESHOLD PHASE TRANSITIONS")
        print("=" * 80)
        
        thresholds = np.linspace(0.05, 0.95, 19)
        voting_rules = ['plurality', 'borda', 'irv']
        center_metric = 'l2'
        extreme_metric = 'cosine'
        
        results = {}
        verified_rules = []
        failed_rules = []
        
        for rule in voting_rules:
            print(f"\nTesting {rule}...")
            disagreements = []
            
            for threshold in thresholds:
                print(f"  theta={threshold:.2f}...", end=" ", flush=True)
                
                # Heterogeneous
                config_het = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=self.n_voters,
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
                    )
                )
                result_het = run_experiment(config_het, save_results=False, verbose=False)
                
                # Homogeneous
                config_homo = SimulationConfig(
                    n_profiles=self.n_profiles,
                    n_voters=self.n_voters,
                    n_candidates=self.n_candidates,
                    voting_rules=[rule],
                    geometry=GeometryConfig(method='uniform', n_dim=2),
                    utility=UtilityConfig(
                        function='linear',
                        distance_metric=center_metric,
                        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                    )
                )
                result_homo = run_experiment(config_homo, save_results=False, verbose=False)
                
                # Compute disagreement
                if rule in result_het.rule_results and rule in result_homo.rule_results:
                    winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                    winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                    disagreement = np.mean(winners_het != winners_homo) * 100
                else:
                    disagreement = 0.0
                
                disagreements.append(disagreement)
                print(f"D = {disagreement:.1f}%")
            
            disagreements = np.array(disagreements)
            
            # Find max and min
            max_idx = np.argmax(disagreements)
            max_threshold = float(thresholds[max_idx])
            max_disagreement = float(disagreements[max_idx])
            min_disagreement = float(np.min(disagreements))
            range_val = max_disagreement - min_disagreement
            
            # Check for sigmoidal pattern (compute curvature)
            d1 = np.diff(disagreements)
            d2 = np.diff(d1)
            max_curvature_idx = np.argmax(np.abs(d2)) if len(d2) > 0 else None
            max_curvature_threshold = float(thresholds[max_curvature_idx + 2]) if max_curvature_idx is not None else None
            
            # Expected values
            if rule == 'plurality':
                expected_max_d = 84.0
                expected_max_theta = 0.6
                expected_min_d = 70.5
                expected_range = 13.5
            elif rule == 'borda':
                expected_max_d = 83.5
                expected_max_theta = 0.7
                expected_min_d = None
                expected_range = 7.5
            else:  # irv
                expected_max_d = 85.0
                expected_max_theta = 0.15
                expected_min_d = None
                expected_range = 8.5
            
            results[rule] = {
                'max_disagreement': max_disagreement,
                'max_threshold': max_threshold,
                'min_disagreement': min_disagreement,
                'range': range_val,
                'max_curvature_threshold': max_curvature_threshold,
                'disagreements': disagreements.tolist(),
                'thresholds': thresholds.tolist(),
                'expected_max_d': expected_max_d,
                'expected_max_theta': expected_max_theta,
                'expected_range': expected_range
            }
            
            # Verify
            max_d_close = abs(max_disagreement - expected_max_d) < 5.0
            max_theta_close = abs(max_threshold - expected_max_theta) < 0.15  # 0.15 tolerance
            range_close = abs(range_val - expected_range) < 3.0 if expected_range else True
            
            if max_d_close and max_theta_close and range_close:
                verified_rules.append(rule)
                print(f"  [OK] Verified: D_max={max_disagreement:.1f}% at theta={max_threshold:.2f}")
            else:
                failed_rules.append(f"{rule}: D_max={max_disagreement:.1f}% at theta={max_threshold:.2f} (expected {expected_max_d:.1f}% at theta={expected_max_theta:.2f})")
                print(f"  [FAIL] Mismatch: D_max={max_disagreement:.1f}% at theta={max_threshold:.2f} (expected {expected_max_d:.1f}% at theta={expected_max_theta:.2f})")
        
        verified = len(verified_rules) >= 2
        
        return VerificationResult(
            finding_name="Threshold Phase Transitions",
            verified=verified,
            expected_values={
                'plurality': {'max_d': 84.0, 'max_theta': 0.6, 'range': 13.5},
                'borda': {'max_d': 83.5, 'max_theta': 0.7, 'range': 7.5},
                'irv': {'max_d': 85.0, 'max_theta': 0.15, 'range': 8.5}
            },
            observed_values=results,
            differences={},
            notes=[
                f"Verified rules: {len(verified_rules)}/{len(voting_rules)}",
                *[f"  - {r}" for r in failed_rules]
            ]
        )
    
    def verify_finding_4_preference_destabilization(self) -> VerificationResult:
        """
        Verify Finding 4: Preference Structure Destabilization Paradox
        
        Tests that heterogeneity increases cycles but improves Condorcet efficiency.
        Expected from FINDINGS.md (theta=0.5):
        - Plurality: ΔC = +3.5%, ΔE = +7.0% (48.2% → 55.2%)
        - Borda: ΔC = +3.5%, ΔE = +1.2% (90.1% → 91.2%)
        - IRV: ΔC = +3.5%, ΔE = +0.8% (81.2% → 82.0%)
        """
        print("\n" + "=" * 80)
        print("VERIFYING FINDING 4: PREFERENCE DESTABILIZATION PARADOX")
        print("=" * 80)
        
        threshold = 0.5
        voting_rules = ['plurality', 'borda', 'irv']
        
        results = {}
        verified_rules = []
        failed_rules = []
        
        for rule in voting_rules:
            print(f"\nTesting {rule}...")
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
                n_candidates=self.n_candidates,
                voting_rules=[rule],
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
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
                n_candidates=self.n_candidates,
                voting_rules=[rule],
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            if rule in result_het.rule_results and rule in result_homo.rule_results:
                het_metrics = result_het.rule_results[rule].aggregate_metrics
                homo_metrics = result_homo.rule_results[rule].aggregate_metrics
                
                cycle_rate_het = het_metrics.cycle_percentage
                cycle_rate_homo = homo_metrics.cycle_percentage
                cycle_change = cycle_rate_het - cycle_rate_homo
                
                condorcet_eff_het = het_metrics.condorcet_efficiency
                condorcet_eff_homo = homo_metrics.condorcet_efficiency
                condorcet_change = condorcet_eff_het - condorcet_eff_homo
            else:
                cycle_rate_het = cycle_rate_homo = cycle_change = 0.0
                condorcet_eff_het = condorcet_eff_homo = condorcet_change = 0.0
            
            # Expected values
            if rule == 'plurality':
                expected_cycle_change = 3.5
                expected_condorcet_change = 7.0
            elif rule == 'borda':
                expected_cycle_change = 3.5
                expected_condorcet_change = 1.2
            else:  # irv
                expected_cycle_change = 3.5
                expected_condorcet_change = 0.8
            
            results[rule] = {
                'cycle_rate_het': cycle_rate_het,
                'cycle_rate_homo': cycle_rate_homo,
                'cycle_change': cycle_change,
                'condorcet_eff_het': condorcet_eff_het,
                'condorcet_eff_homo': condorcet_eff_homo,
                'condorcet_change': condorcet_change,
                'expected_cycle_change': expected_cycle_change,
                'expected_condorcet_change': expected_condorcet_change
            }
            
            # Verify paradox: cycles increase AND efficiency improves
            cycle_increases = cycle_change > 0
            efficiency_improves = condorcet_change > 0
            cycle_close = abs(cycle_change - expected_cycle_change) < 2.0
            condorcet_close = abs(condorcet_change - expected_condorcet_change) < 3.0
            
            paradox_holds = cycle_increases and efficiency_improves
            
            if paradox_holds and cycle_close and condorcet_close:
                verified_rules.append(rule)
                print(f"  [OK] Verified: Delta_C = +{cycle_change:.1f}%, Delta_E = +{condorcet_change:.1f}%")
            else:
                status = []
                if not cycle_increases:
                    status.append("cycles don't increase")
                if not efficiency_improves:
                    status.append("efficiency doesn't improve")
                if not cycle_close:
                    status.append(f"cycle change mismatch ({cycle_change:.1f}% vs {expected_cycle_change:.1f}%)")
                if not condorcet_close:
                    status.append(f"efficiency change mismatch ({condorcet_change:.1f}% vs {expected_condorcet_change:.1f}%)")
                
                failed_rules.append(f"{rule}: {', '.join(status)}")
                print(f"  [FAIL] Mismatch: Delta_C = +{cycle_change:.1f}%, Delta_E = +{condorcet_change:.1f}%")
        
        verified = len(verified_rules) >= 2
        
        return VerificationResult(
            finding_name="Preference Destabilization Paradox",
            verified=verified,
            expected_values={
                'plurality': {'cycle_change': 3.5, 'condorcet_change': 7.0},
                'borda': {'cycle_change': 3.5, 'condorcet_change': 1.2},
                'irv': {'cycle_change': 3.5, 'condorcet_change': 0.8}
            },
            observed_values=results,
            differences={},
            notes=[
                f"Verified rules: {len(verified_rules)}/{len(voting_rules)}",
                *[f"  - {r}" for r in failed_rules]
            ]
        )
    
    def verify_finding_5_metric_hierarchy(self) -> VerificationResult:
        """
        Verify Finding 5: Metric Interaction Strength Hierarchy
        
        Tests that different metric pairs have systematically different interaction strengths.
        Expected from FINDINGS.md (Plurality, descending order):
        1. Cosine ↔ L1: 87.3%
        2. L2 ↔ Chebyshev: 86.0%
        3. L1 ↔ L2: 84.0%
        4. L2 ↔ Cosine: 82.7%
        5. Cosine ↔ Chebyshev: 79.3%
        """
        print("\n" + "=" * 80)
        print("VERIFYING FINDING 5: METRIC INTERACTION STRENGTH HIERARCHY")
        print("=" * 80)
        
        threshold = 0.5
        rule = 'plurality'
        metric_pairs = [
            ('cosine', 'l1', 87.3),
            ('l2', 'chebyshev', 86.0),
            ('l1', 'l2', 84.0),
            ('l2', 'cosine', 82.7),
            ('cosine', 'chebyshev', 79.3),
        ]
        
        results = {}
        verified_pairs = []
        failed_pairs = []
        
        for center_metric, extreme_metric, expected_d in metric_pairs:
            print(f"\nTesting {center_metric}->{extreme_metric}...", end=" ", flush=True)
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
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
                )
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous
            config_homo = SimulationConfig(
                n_profiles=self.n_profiles,
                n_voters=self.n_voters,
                n_candidates=self.n_candidates,
                voting_rules=[rule],
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=center_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            # Compute disagreement
            if rule in result_het.rule_results and rule in result_homo.rule_results:
                winners_het = np.array([m.winner_index for m in result_het.rule_results[rule].profile_metrics])
                winners_homo = np.array([m.winner_index for m in result_homo.rule_results[rule].profile_metrics])
                disagreement = np.mean(winners_het != winners_homo) * 100
            else:
                disagreement = 0.0
            
            results[f"{center_metric}_{extreme_metric}"] = {
                'disagreement': disagreement,
                'expected': expected_d
            }
            
            # Verify
            if abs(disagreement - expected_d) < 5.0:  # 5% tolerance
                verified_pairs.append(f"{center_metric}<->{extreme_metric}")
                print(f"D = {disagreement:.1f}% (expected {expected_d:.1f}%) [OK]")
            else:
                failed_pairs.append(f"{center_metric}<->{extreme_metric}: D = {disagreement:.1f}% (expected {expected_d:.1f}%)")
                print(f"D = {disagreement:.1f}% (expected {expected_d:.1f}%) [FAIL]")
        
        # Check hierarchy ordering
        sorted_pairs = sorted(results.items(), key=lambda x: -x[1]['disagreement'])
        expected_order = [f"{m1}_{m2}" for m1, m2, _ in metric_pairs]
        observed_order = [pair[0] for pair in sorted_pairs]
        
        order_correct = observed_order == expected_order
        
        verified = len(verified_pairs) >= 3 and order_correct
        
        return VerificationResult(
            finding_name="Metric Interaction Strength Hierarchy",
            verified=verified,
            expected_values={
                'expected_order': expected_order,
                'expected_values': {f"{m1}_{m2}": d for m1, m2, d in metric_pairs}
            },
            observed_values={
                'observed_order': observed_order,
                'values': results
            },
            differences={},
            notes=[
                f"Verified pairs: {len(verified_pairs)}/{len(metric_pairs)}",
                f"Order correct: {order_correct}",
                *[f"  - {p}" for p in failed_pairs]
            ]
        )
    
    def run_all_verifications(self) -> Dict[str, VerificationResult]:
        """Run all verification tests."""
        print("=" * 80)
        print("EXPERIMENTAL VERIFICATION OF FINDINGS")
        print("=" * 80)
        print(f"Profiles per test: {self.n_profiles}")
        print(f"Voters per profile: {self.n_voters}")
        print(f"Candidates per profile: {self.n_candidates}")
        print()
        
        results = {}
        
        results['finding_1'] = self.verify_finding_1_asymmetric_interaction()
        results['finding_2'] = self.verify_finding_2_dimensional_scaling()
        results['finding_3'] = self.verify_finding_3_threshold_phase_transitions()
        results['finding_4'] = self.verify_finding_4_preference_destabilization()
        results['finding_5'] = self.verify_finding_5_metric_hierarchy()
        
        # Summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        verified_count = sum(1 for r in results.values() if r.verified)
        total_count = len(results)
        
        for key, result in results.items():
            status = "[VERIFIED]" if result.verified else "[FAILED]"
            print(f"\n{status}: {result.finding_name}")
            for note in result.notes:
                print(f"  {note}")
        
        print(f"\n{'=' * 80}")
        print(f"Overall: {verified_count}/{total_count} findings verified")
        print("=" * 80)
        
        # Save results
        output_file = self.output_dir / "verification_results.json"
        with open(output_file, 'w') as f:
            # Convert to dict for JSON serialization
            results_dict = {
                key: {
                    'finding_name': r.finding_name,
                    'verified': r.verified,
                    'expected_values': r.expected_values,
                    'observed_values': r.observed_values,
                    'notes': r.notes
                }
                for key, r in results.items()
            }
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        return results


if __name__ == '__main__':
    verifier = FindingsVerifier(n_profiles=200, n_voters=100, n_candidates=5)
    verifier.run_all_verifications()

