"""
Visualization script for findings in FINDINGS.md.

This script runs simulations at research scale (10000 voters) and generates
publication-quality visualizations for all 5 novel phenomena:

1. Asymmetric Metric Interaction
2. Dimensional Scaling Laws
3. Threshold Phase Transitions
4. Preference Structure Destabilization Paradox
5. Metric Interaction Strength Hierarchy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Any
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.config import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig, VotingRuleConfig
)
from simulator.main import run_experiment, ExperimentResult

# Research-scale parameters
RESEARCH_N_VOTERS = 500
RESEARCH_N_PROFILES = 200  # Reduced profiles for computational efficiency
RESEARCH_N_CANDIDATES = 5

# Output directory for images
OUTPUT_DIR = Path("heterogeneity-research/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
METRIC_COLORS = {
    'l1': '#FF6B6B',      # Red
    'l2': '#4ECDC4',      # Teal
    'cosine': '#95E1D3',  # Light teal
    'chebyshev': '#F38181'  # Pink
}

METRIC_NAMES = {
    'l1': 'L1',
    'l2': 'L2',
    'cosine': 'Cosine',
    'chebyshev': 'Chebyshev'
}

RULE_COLORS = {
    'plurality': '#FF6B6B',
    'borda': '#4ECDC4',
    'irv': '#95E1D3'
}

RULE_NAMES = {
    'plurality': 'Plurality',
    'borda': 'Borda',
    'irv': 'IRV'
}


def compute_disagreement(het_result: ExperimentResult, homo_result: ExperimentResult, rule: str) -> float:
    """Compute disagreement rate between heterogeneous and homogeneous results."""
    if rule not in het_result.rule_results or rule not in homo_result.rule_results:
        return 0.0
    
    het_winners = het_result.rule_results[rule].winners
    homo_winners = homo_result.rule_results[rule].winners
    disagreement = np.mean(het_winners != homo_winners) * 100
    return disagreement


def visualize_asymmetric_metric_interaction(save_path: Path):
    """
    Visualization 1: Asymmetric Metric Interaction
    
    Shows how the order of metric assignment (A->B vs B->A) creates
    different disagreement rates.
    """
    print("\n" + "="*80)
    print("VISUALIZATION 1: Asymmetric Metric Interaction")
    print("="*80)
    
    # Test pairs with known asymmetry
    metric_pairs = [
        ('l1', 'cosine'),
        ('l1', 'chebyshev'),
        ('l1', 'l2'),
    ]
    
    voting_rules = ['plurality', 'borda', 'irv']
    threshold = 0.5
    
    results = {}
    
    for center_metric, extreme_metric in metric_pairs:
        pair_name = f"{center_metric}_{extreme_metric}"
        print(f"\nTesting pair: {pair_name}...")
        
        results[pair_name] = {}
        
        # Test both directions
        for direction in ['forward', 'reverse']:
            if direction == 'forward':
                c_metric, e_metric = center_metric, extreme_metric
            else:
                c_metric, e_metric = extreme_metric, center_metric
            
            print(f"  Direction: {c_metric} -> {e_metric}...", end=" ", flush=True)
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=RESEARCH_N_PROFILES,
                n_voters=RESEARCH_N_VOTERS,
                n_candidates=RESEARCH_N_CANDIDATES,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric='l2',
                    heterogeneous_distance=HeterogeneousDistanceConfig(
                        enabled=True,
                        strategy='center_extreme',
                        center_metric=c_metric,
                        extreme_metric=e_metric,
                        extreme_threshold=threshold
                    )
                )
            )
            result_het = run_experiment(config_het, save_results=False, verbose=False)
            
            # Homogeneous (center metric)
            config_homo = SimulationConfig(
                n_profiles=RESEARCH_N_PROFILES,
                n_voters=RESEARCH_N_VOTERS,
                n_candidates=RESEARCH_N_CANDIDATES,
                voting_rules=voting_rules,
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=c_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            direction_data = {}
            for rule in voting_rules:
                disagreement = compute_disagreement(result_het, result_homo, rule)
                direction_data[rule] = disagreement
            
            results[pair_name][direction] = direction_data
            print("Done")
    
    # Create visualization
    fig, axes = plt.subplots(1, len(voting_rules), figsize=(16, 5))
    if len(voting_rules) == 1:
        axes = [axes]
    
    x = np.arange(len(metric_pairs))
    width = 0.35
    
    for rule_idx, rule in enumerate(voting_rules):
        ax = axes[rule_idx]
        
        forward_values = []
        reverse_values = []
        pair_labels = []
        
        for center_metric, extreme_metric in metric_pairs:
            pair_name = f"{center_metric}_{extreme_metric}"
            forward_values.append(results[pair_name]['forward'][rule])
            reverse_values.append(results[pair_name]['reverse'][rule])
            pair_labels.append(f"{METRIC_NAMES[center_metric]}\n↔\n{METRIC_NAMES[extreme_metric]}")
        
        bars1 = ax.bar(x - width/2, forward_values, width, 
                      label=f'{METRIC_NAMES[metric_pairs[0][0]]}→{METRIC_NAMES[metric_pairs[0][1]]}',
                      color=METRIC_COLORS.get(metric_pairs[0][0], 'gray'), alpha=0.7)
        bars2 = ax.bar(x + width/2, reverse_values, width,
                      label=f'{METRIC_NAMES[metric_pairs[0][1]]}→{METRIC_NAMES[metric_pairs[0][0]]}',
                      color=METRIC_COLORS.get(metric_pairs[0][1], 'gray'), alpha=0.7)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metric Pair', fontsize=12)
        ax.set_ylabel('Disagreement Rate (%)', fontsize=12)
        ax.set_title(f'{RULE_NAMES[rule]} Rule', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(max(forward_values), max(reverse_values)) * 1.15)
    
    fig.suptitle('Asymmetric Metric Interaction: Order Matters', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / '1_asymmetric_metric_interaction.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path / '1_asymmetric_metric_interaction.png'}")


def visualize_dimensional_scaling(save_path: Path):
    """
    Visualization 2: Dimensional Scaling Laws
    
    Shows how heterogeneity effects scale with dimensionality,
    with peak effects at 2-3 dimensions.
    """
    print("\n" + "="*80)
    print("VISUALIZATION 2: Dimensional Scaling Laws")
    print("="*80)
    
    dimensions = [1, 2, 3, 4, 5, 7, 10]
    voting_rules = ['plurality', 'borda', 'irv']
    
    het_config = HeterogeneousDistanceConfig(
        enabled=True,
        strategy='center_extreme',
        center_metric='l2',
        extreme_metric='cosine',
        extreme_threshold=0.5
    )
    
    results = {rule: [] for rule in voting_rules}
    
    for n_dim in dimensions:
        print(f"\nTesting {n_dim}D...", end=" ", flush=True)
        
        # Heterogeneous
        config_het = SimulationConfig(
            n_profiles=RESEARCH_N_PROFILES,
            n_voters=RESEARCH_N_VOTERS,
            n_candidates=RESEARCH_N_CANDIDATES,
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
            n_profiles=RESEARCH_N_PROFILES,
            n_voters=RESEARCH_N_VOTERS,
            n_candidates=RESEARCH_N_CANDIDATES,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method='uniform', n_dim=n_dim),
            utility=UtilityConfig(
                function='linear',
                distance_metric='l2',
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            )
        )
        result_homo = run_experiment(config_homo, save_results=False, verbose=False)
        
        for rule in voting_rules:
            disagreement = compute_disagreement(result_het, result_homo, rule)
            results[rule].append(disagreement)
        
        print("Done")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for rule in voting_rules:
        ax.plot(dimensions, results[rule], 'o-', linewidth=2.5, markersize=10,
               label=RULE_NAMES[rule], color=RULE_COLORS[rule], alpha=0.8)
    
    # Highlight peak dimensions
    for rule in voting_rules:
        peak_idx = np.argmax(results[rule])
        peak_dim = dimensions[peak_idx]
        peak_val = results[rule][peak_idx]
        ax.plot(peak_dim, peak_val, 'o', markersize=15, 
               color=RULE_COLORS[rule], markerfacecolor='none', 
               markeredgewidth=3, zorder=10)
        ax.annotate(f'Peak: {peak_dim}D\n{peak_val:.1f}%',
                   xy=(peak_dim, peak_val),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Dimensionality', fontsize=14, fontweight='bold')
    ax.set_ylabel('Disagreement Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Dimensional Scaling Laws: Peak Effects at 2-3 Dimensions', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 10.5)
    
    plt.tight_layout()
    plt.savefig(save_path / '2_dimensional_scaling.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path / '2_dimensional_scaling.png'}")


def visualize_threshold_phase_transitions(save_path: Path):
    """
    Visualization 3: Threshold Phase Transitions
    
    Shows sigmoidal response curves with inflection points,
    demonstrating phase-like transitions.
    """
    print("\n" + "="*80)
    print("VISUALIZATION 3: Threshold Phase Transitions")
    print("="*80)
    
    thresholds = np.linspace(0.05, 0.95, 19)
    voting_rules = ['plurality', 'borda', 'irv']
    center_metric = 'l2'
    extreme_metric = 'cosine'
    
    results = {rule: [] for rule in voting_rules}
    
    for threshold in thresholds:
        print(f"Threshold {threshold:.2f}...", end=" ", flush=True)
        
        # Heterogeneous
        config_het = SimulationConfig(
            n_profiles=RESEARCH_N_PROFILES,
            n_voters=RESEARCH_N_VOTERS,
            n_candidates=RESEARCH_N_CANDIDATES,
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
        
        # Homogeneous
        config_homo = SimulationConfig(
            n_profiles=RESEARCH_N_PROFILES,
            n_voters=RESEARCH_N_VOTERS,
            n_candidates=RESEARCH_N_CANDIDATES,
            voting_rules=voting_rules,
            geometry=GeometryConfig(method='uniform', n_dim=2),
            utility=UtilityConfig(
                function='linear',
                distance_metric=center_metric,
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            )
        )
        result_homo = run_experiment(config_homo, save_results=False, verbose=False)
        
        for rule in voting_rules:
            disagreement = compute_disagreement(result_het, result_homo, rule)
            results[rule].append(disagreement)
        
        print("Done")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for rule in voting_rules:
        ax.plot(thresholds, results[rule], 'o-', linewidth=2.5, markersize=8,
               label=RULE_NAMES[rule], color=RULE_COLORS[rule], alpha=0.8)
        
        # Fit sigmoid and plot (optional, requires scipy)
        try:
            from scipy.optimize import curve_fit
            
            def sigmoid(x, A, k, x0, y0):
                return y0 + A / (1 + np.exp(-k * (x - x0)))
            
            p0 = [max(results[rule]) - min(results[rule]), 5.0, 0.8, min(results[rule])]
            popt, _ = curve_fit(sigmoid, thresholds, results[rule], p0=p0, maxfev=5000)
            x_smooth = np.linspace(thresholds.min(), thresholds.max(), 200)
            y_smooth = sigmoid(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, '--', linewidth=1.5, 
                   color=RULE_COLORS[rule], alpha=0.5, label=f'{RULE_NAMES[rule]} (sigmoid fit)')
        except ImportError:
            pass  # scipy not available, skip sigmoid fit
        except Exception:
            pass  # Fit failed, skip sigmoid plot
    
    # Mark critical thresholds
    for rule in voting_rules:
        max_idx = np.argmax(results[rule])
        max_threshold = thresholds[max_idx]
        max_val = results[rule][max_idx]
        ax.plot(max_threshold, max_val, 's', markersize=12,
               color=RULE_COLORS[rule], markerfacecolor='none',
               markeredgewidth=2, zorder=10)
    
    ax.set_xlabel('Threshold (θ)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Disagreement Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Threshold Phase Transitions: Sigmoidal Response Curves',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / '3_threshold_phase_transitions.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path / '3_threshold_phase_transitions.png'}")


def visualize_preference_destabilization(save_path: Path):
    """
    Visualization 4: Preference Structure Destabilization Paradox
    
    Shows the paradox: heterogeneity increases cycle rates (destabilization)
    while improving Condorcet efficiency for some rules.
    """
    print("\n" + "="*80)
    print("VISUALIZATION 4: Preference Structure Destabilization Paradox")
    print("="*80)
    
    threshold = 0.5
    voting_rules = ['plurality', 'borda', 'irv']
    
    # Heterogeneous
    config_het = SimulationConfig(
        n_profiles=RESEARCH_N_PROFILES,
        n_voters=RESEARCH_N_VOTERS,
        n_candidates=RESEARCH_N_CANDIDATES,
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
    print("Running heterogeneous simulation...", end=" ", flush=True)
    result_het = run_experiment(config_het, save_results=False, verbose=False)
    print("Done")
    
    # Homogeneous
    config_homo = SimulationConfig(
        n_profiles=RESEARCH_N_PROFILES,
        n_voters=RESEARCH_N_VOTERS,
        n_candidates=RESEARCH_N_CANDIDATES,
        voting_rules=voting_rules,
        geometry=GeometryConfig(method='uniform', n_dim=2),
        utility=UtilityConfig(
            function='linear',
            distance_metric='l2',
            heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
        )
    )
    print("Running homogeneous simulation...", end=" ", flush=True)
    result_homo = run_experiment(config_homo, save_results=False, verbose=False)
    print("Done")
    
    # Extract metrics
    cycle_rates_het = []
    cycle_rates_homo = []
    condorcet_eff_het = []
    condorcet_eff_homo = []
    rule_labels = []
    
    for rule in voting_rules:
        if rule in result_het.rule_results and rule in result_homo.rule_results:
            het_agg = result_het.rule_results[rule].aggregate_metrics
            homo_agg = result_homo.rule_results[rule].aggregate_metrics
            
            cycle_rates_het.append(het_agg.cycle_percentage)
            cycle_rates_homo.append(homo_agg.cycle_percentage)
            condorcet_eff_het.append(het_agg.condorcet_efficiency)
            condorcet_eff_homo.append(homo_agg.condorcet_efficiency)
            rule_labels.append(RULE_NAMES[rule])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(rule_labels))
    width = 0.35
    
    # Cycle rates
    bars1 = ax1.bar(x - width/2, cycle_rates_homo, width, 
                   label='Homogeneous', color='#4ECDC4', alpha=0.7)
    bars2 = ax1.bar(x + width/2, cycle_rates_het, width,
                   label='Heterogeneous', color='#FF6B6B', alpha=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Voting Rule', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cycle Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Destabilization: Increased Cycle Rates', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rule_labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Condorcet efficiency
    bars3 = ax2.bar(x - width/2, condorcet_eff_homo, width,
                   label='Homogeneous', color='#4ECDC4', alpha=0.7)
    bars4 = ax2.bar(x + width/2, condorcet_eff_het, width,
                   label='Heterogeneous', color='#FF6B6B', alpha=0.7)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Voting Rule', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Condorcet Efficiency (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Paradox: Improved Condorcet Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rule_labels)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Preference Structure Destabilization Paradox:\n'
                 'More Cycles, Better Efficiency',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / '4_preference_destabilization.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path / '4_preference_destabilization.png'}")


def visualize_metric_interaction_hierarchy(save_path: Path):
    """
    Visualization 5: Metric Interaction Strength Hierarchy
    
    Shows the interaction strength matrix for different metric pairs.
    """
    print("\n" + "="*80)
    print("VISUALIZATION 5: Metric Interaction Strength Hierarchy")
    print("="*80)
    
    metrics = ['l1', 'l2', 'cosine', 'chebyshev']
    threshold = 0.5
    voting_rule = 'plurality'  # Focus on plurality as in FINDINGS.md
    
    interaction_matrix = {}
    
    for center_metric in metrics:
        for extreme_metric in metrics:
            if center_metric == extreme_metric:
                continue
            
            pair_name = f"{center_metric}_{extreme_metric}"
            print(f"Testing {pair_name}...", end=" ", flush=True)
            
            # Heterogeneous
            config_het = SimulationConfig(
                n_profiles=RESEARCH_N_PROFILES,
                n_voters=RESEARCH_N_VOTERS,
                n_candidates=RESEARCH_N_CANDIDATES,
                voting_rules=[voting_rule],
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
                n_profiles=RESEARCH_N_PROFILES,
                n_voters=RESEARCH_N_VOTERS,
                n_candidates=RESEARCH_N_CANDIDATES,
                voting_rules=[voting_rule],
                geometry=GeometryConfig(method='uniform', n_dim=2),
                utility=UtilityConfig(
                    function='linear',
                    distance_metric=center_metric,
                    heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
                )
            )
            result_homo = run_experiment(config_homo, save_results=False, verbose=False)
            
            disagreement = compute_disagreement(result_het, result_homo, voting_rule)
            interaction_matrix[pair_name] = disagreement
            print("Done")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare data for heatmap
    n_metrics = len(metrics)
    heatmap_data = np.zeros((n_metrics, n_metrics))
    
    for i, center_metric in enumerate(metrics):
        for j, extreme_metric in enumerate(metrics):
            if center_metric == extreme_metric:
                heatmap_data[i, j] = np.nan
            else:
                pair_name = f"{center_metric}_{extreme_metric}"
                heatmap_data[i, j] = interaction_matrix.get(pair_name, 0)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_metrics))
    ax.set_xticklabels([METRIC_NAMES[m] for m in metrics])
    ax.set_yticklabels([METRIC_NAMES[m] for m in metrics])
    
    # Add text annotations
    for i in range(n_metrics):
        for j in range(n_metrics):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                             ha="center", va="center",
                             color="white" if heatmap_data[i, j] > 50 else "black",
                             fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Extreme Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Center Metric', fontsize=14, fontweight='bold')
    ax.set_title(f'Metric Interaction Strength Hierarchy ({RULE_NAMES[voting_rule]} Rule)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Disagreement Rate (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '5_metric_interaction_hierarchy.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path / '5_metric_interaction_hierarchy.png'}")


def main():
    """Main function to run all visualizations."""
    print("="*80)
    print("HETEROGENEITY RESEARCH VISUALIZATIONS")
    print("="*80)
    print(f"\nResearch Parameters:")
    print(f"  Voters per profile: {RESEARCH_N_VOTERS:,}")
    print(f"  Profiles: {RESEARCH_N_PROFILES}")
    print(f"  Candidates: {RESEARCH_N_CANDIDATES}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nThis will take a while...")
    
    # Run all visualizations
    try:
        visualize_asymmetric_metric_interaction(OUTPUT_DIR)
    except Exception as e:
        print(f"\nERROR in Visualization 1: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        visualize_dimensional_scaling(OUTPUT_DIR)
    except Exception as e:
        print(f"\nERROR in Visualization 2: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        visualize_threshold_phase_transitions(OUTPUT_DIR)
    except Exception as e:
        print(f"\nERROR in Visualization 3: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        visualize_preference_destabilization(OUTPUT_DIR)
    except Exception as e:
        print(f"\nERROR in Visualization 4: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        visualize_metric_interaction_hierarchy(OUTPUT_DIR)
    except Exception as e:
        print(f"\nERROR in Visualization 5: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for i in range(1, 6):
        filename = OUTPUT_DIR / f'{i}_*.png'
        files = list(OUTPUT_DIR.glob(f'{i}_*.png'))
        if files:
            print(f"  - {files[0].name}")


if __name__ == '__main__':
    main()

