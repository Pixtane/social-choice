"""
Verification script for research paper data.

This script:
1. Loads and verifies all JSON result files
2. Computes disagreement decomposition where possible
3. Extracts verified findings
4. Generates statistical summaries
5. Verifies key claims
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Add heterogenity-simulator to path for analysis tools
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from heterogenity_simulator.analyze_results import ResultsAnalyzer
except ImportError:
    # Fallback if import fails
    ResultsAnalyzer = None

RESULTS_DIR = Path("heterogenity-simulator/results")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def verify_utility_function():
    """Verify that experiments used linear utility function."""
    print("=" * 80)
    print("VERIFYING UTILITY FUNCTION")
    print("=" * 80)

    # Check research_suite.py
    research_suite_path = Path("heterogenity-simulator/research_suite.py")
    if research_suite_path.exists():
        with open(research_suite_path, 'r') as f:
            content = f.read()
            linear_count = content.count("function='linear'")
            gaussian_count = content.count("function='gaussian'")

            print(f"[OK] Found {linear_count} instances of function='linear'")
            print(f"[OK] Found {gaussian_count} instances of function='gaussian'")

            if linear_count > 0 and gaussian_count == 0:
                print("[VERIFIED] All experiments use linear utility function")
                return True
            else:
                print("[WARNING] Unexpected utility function usage")
                return False
    else:
        print("[WARNING] Could not find research_suite.py")
        return False


def compute_disagreement_decomposition(
    het_winners: np.ndarray,
    center_winners: np.ndarray,
    extreme_winners: np.ndarray
) -> Dict[str, float]:
    """
    Compute strong and extreme-aligned disagreement per METHODOLOGY.md.

    Args:
        het_winners: Heterogeneous winners (n_profiles,)
        center_winners: Center baseline winners (n_profiles,)
        extreme_winners: Extreme baseline winners (n_profiles,)

    Returns:
        Dictionary with:
        - strong_disagreement: Percentage where het ≠ center AND het ≠ extreme
        - extreme_aligned_disagreement: Percentage where het = extreme AND het ≠ center
        - total_disagreement: strong + extreme_aligned
    """
    het_winners = np.asarray(het_winners)
    center_winners = np.asarray(center_winners)
    extreme_winners = np.asarray(extreme_winners)

    n = len(het_winners)

    # Strong disagreement: het ≠ center AND het ≠ extreme
    strong_mask = (het_winners != center_winners) & (het_winners != extreme_winners)
    strong_disagreement = np.sum(strong_mask) / n * 100.0

    # Extreme-aligned disagreement: het = extreme AND het ≠ center
    extreme_aligned_mask = (het_winners == extreme_winners) & (het_winners != center_winners)
    extreme_aligned_disagreement = np.sum(extreme_aligned_mask) / n * 100.0

    # Total (should equal simple disagreement vs center)
    total_disagreement = strong_disagreement + extreme_aligned_disagreement

    return {
        'strong_disagreement': float(strong_disagreement),
        'extreme_aligned_disagreement': float(extreme_aligned_disagreement),
        'total_disagreement': float(total_disagreement)
    }


def analyze_dimensional_scaling(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze dimensional scaling results."""
    print("\n" + "=" * 80)
    print("ANALYZING DIMENSIONAL SCALING")
    print("=" * 80)

    dimensions = data['dimensions']
    results = {}

    for rule in ['plurality', 'borda', 'irv']:
        disagreements = []
        vse_values = []

        for dim in dimensions:
            dim_key = str(dim)
            if dim_key in data['data'] and rule in data['data'][dim_key]:
                rule_data = data['data'][dim_key][rule]
                disagreements.append(rule_data.get('disagreement_rate', 0.0))
                vse_values.append(rule_data.get('vse_het', 0.0))

        if len(disagreements) > 0:
            disagreements = np.array(disagreements)
            dims = np.array(dimensions[:len(disagreements)])

            # Find peak
            peak_idx = np.argmax(disagreements)
            peak_dim = dims[peak_idx]
            peak_value = disagreements[peak_idx]

            # Power law fit for pre-peak
            pre_peak_mask = dims <= peak_dim
            if np.sum(pre_peak_mask) >= 2:
                pre_peak_dims = dims[pre_peak_mask]
                pre_peak_disagreements = disagreements[pre_peak_mask]

                # Log-log fit
                log_dims = np.log(pre_peak_dims + 1e-10)
                log_disagreements = np.log(pre_peak_disagreements + 1e-10)

                coeffs = np.polyfit(log_dims, log_disagreements, 1)
                alpha = coeffs[0]

                # R-squared
                log_pred = np.polyval(coeffs, log_dims)
                ss_res = np.sum((log_disagreements - log_pred)**2)
                ss_tot = np.sum((log_disagreements - np.mean(log_disagreements))**2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            else:
                alpha = None
                r_squared = None

            results[rule] = {
                'dimensions': dims.tolist(),
                'disagreements': disagreements.tolist(),
                'vse_values': vse_values,
                'peak_dimension': int(peak_dim),
                'peak_disagreement': float(peak_value),
                'scaling_exponent': float(alpha) if alpha is not None else None,
                'r_squared': float(r_squared) if r_squared is not None else None,
                'min_disagreement': float(disagreements[0]),
                'max_disagreement': float(disagreements[-1])
            }

            print(f"\n{rule.upper()}:")
            print(f"  Peak dimension: {peak_dim}, Peak disagreement: {peak_value:.2f}%")
            print(f"  Range: {disagreements[0]:.2f}% to {disagreements[-1]:.2f}%")
            if alpha is not None:
                print(f"  Power law exponent: {alpha:.3f}, R²: {r_squared:.3f}")

    return results


def analyze_metric_pairs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze metric pair interactions."""
    print("\n" + "=" * 80)
    print("ANALYZING METRIC PAIRS")
    print("=" * 80)

    pairs = data.get('pairs', {})
    results = {}

    for rule in ['plurality', 'borda', 'irv']:
        pair_results = {}

        for pair_name, pair_data in pairs.items():
            if rule not in pair_data:
                continue

            rule_data = pair_data[rule]
            disagreement_ab = rule_data.get('disagreement_ab', 0.0)
            disagreement_ba = rule_data.get('disagreement_ba', 0.0)
            asymmetry = rule_data.get('asymmetry', 0.0)

            # Note: These are simple disagreements (vs center baseline)
            # We cannot compute decomposition without per-profile data
            pair_results[pair_name] = {
                'disagreement_ab': disagreement_ab,
                'disagreement_ba': disagreement_ba,
                'asymmetry': asymmetry,
                'avg_disagreement': (disagreement_ab + disagreement_ba) / 2,
                'cycle_rate_delta_ab': rule_data.get('cycle_rate_delta_ab', 0.0),
                'condorcet_efficiency_delta_ab': rule_data.get('condorcet_efficiency_delta_ab', 0.0)
            }

        # Sort by average disagreement
        sorted_pairs = sorted(
            pair_results.items(),
            key=lambda x: x[1]['avg_disagreement'],
            reverse=True
        )

        results[rule] = {
            'pairs': pair_results,
            'sorted_by_strength': sorted_pairs
        }

        print(f"\n{rule.upper()} - Top 5 strongest pairs:")
        for pair_name, pair_data in sorted_pairs[:5]:
            print(f"  {pair_name}: {pair_data['avg_disagreement']:.2f}% "
                  f"(asymmetry: {pair_data['asymmetry']:.2f}%)")

    return results


def analyze_centrality_concentration(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze centrality concentration results."""
    print("\n" + "=" * 80)
    print("ANALYZING CENTRALITY CONCENTRATION")
    print("=" * 80)

    by_dim = data.get('by_dimension', {})
    theory_value = data.get('theory', {}).get('sqrt_one_third', np.sqrt(1/3))

    results = {}

    for dim_str, dim_data in by_dim.items():
        dim = int(dim_str)
        centrality_stats = dim_data.get('centrality', {}).get('stats', {})

        mean_centrality = centrality_stats.get('mean', 0.0)
        std_centrality = centrality_stats.get('std', 0.0)

        results[dim] = {
            'mean': mean_centrality,
            'std': std_centrality,
            'theory': theory_value,
            'deviation': abs(mean_centrality - theory_value)
        }

        print(f"Dimension {dim}: mean={mean_centrality:.4f}, "
              f"theory={theory_value:.4f}, deviation={abs(mean_centrality - theory_value):.4f}")

    # Check convergence
    high_dims = [d for d in results.keys() if d >= 10]
    if high_dims:
        high_dim_means = [results[d]['mean'] for d in high_dims]
        avg_high_dim = np.mean(high_dim_means)
        print(f"\nAverage centrality for d>=10: {avg_high_dim:.4f} (theory: {theory_value:.4f})")
        print(f"Convergence: {abs(avg_high_dim - theory_value):.4f} deviation")

    return results


def main():
    """Main verification function."""
    print("=" * 80)
    print("RESEARCH PAPER DATA VERIFICATION")
    print("=" * 80)

    # Step 1: Verify utility function
    utility_verified = verify_utility_function()

    # Step 2: Load result files
    print("\n" + "=" * 80)
    print("LOADING RESULT FILES")
    print("=" * 80)

    files_to_load = {
        'dimensional_scaling': RESULTS_DIR / 'dimensional_scaling_l2_cosine_v100.json',
        'metric_pairs': RESULTS_DIR / 'metric_pairs_d2_v100.json',
        'threshold_sweep': RESULTS_DIR / 'threshold_sweep_l2_cosine_d2_v100.json',
        'voter_scaling': RESULTS_DIR / 'voter_scaling_l2_cosine_d2.json',
        'centrality_concentration': RESULTS_DIR / 'centrality_concentration_report.json'
    }

    loaded_data = {}
    for name, filepath in files_to_load.items():
        if filepath.exists():
            print(f"[OK] Loading {name}: {filepath.name}")
            loaded_data[name] = load_json(filepath)
        else:
            print(f"[WARNING] Missing: {filepath.name}")

    # Step 3: Analyze dimensional scaling
    if 'dimensional_scaling' in loaded_data:
        dim_analysis = analyze_dimensional_scaling(loaded_data['dimensional_scaling'])

    # Step 4: Analyze metric pairs
    if 'metric_pairs' in loaded_data:
        pair_analysis = analyze_metric_pairs(loaded_data['metric_pairs'])

    # Step 5: Analyze centrality concentration
    if 'centrality_concentration' in loaded_data:
        centrality_analysis = analyze_centrality_concentration(loaded_data['centrality_concentration'])

    # Step 6: Generate summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Utility function verified: {'YES' if utility_verified else 'NO'}")
    print(f"Files loaded: {len(loaded_data)}/{len(files_to_load)}")

    # Save findings
    findings = {
        'utility_function_verified': utility_verified,
        'dimensional_scaling': dim_analysis if 'dimensional_scaling' in loaded_data else None,
        'metric_pairs': pair_analysis if 'metric_pairs' in loaded_data else None,
        'centrality_concentration': centrality_analysis if 'centrality_concentration' in loaded_data else None
    }

    output_file = Path('verified_findings.json')
    with open(output_file, 'w') as f:
        json.dump(findings, f, indent=2)

    print(f"\n[OK] Findings saved to: {output_file}")
    print("\nNOTE: Disagreement decomposition (strong vs extreme-aligned) requires")
    print("per-profile winner data, which is not available in JSON files.")
    print("Current 'disagreement_rate' values are simple disagreements vs center baseline.")


if __name__ == "__main__":
    main()
