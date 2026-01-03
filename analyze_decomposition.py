"""
Analyze disagreement decomposition from new experimental results.

This script analyzes the strong and extreme-aligned disagreement components
to understand heterogeneity mechanisms.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_dimensional_scaling(filepath):
    """Load dimensional scaling results."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_decomposition(data):
    """Analyze disagreement decomposition patterns."""

    dimensions = data['dimensions']

    print("=" * 80)
    print("DISAGREEMENT DECOMPOSITION ANALYSIS")
    print("=" * 80)

    for rule in ['plurality', 'borda', 'irv']:
        print(f"\n{rule.upper()}:")
        print("-" * 40)

        strong = []
        extreme_aligned = []
        total = []
        simple = []

        for dim in dimensions:
            dim_data = data['data'][str(dim)][rule]
            strong.append(dim_data['strong_disagreement'])
            extreme_aligned.append(dim_data['extreme_aligned_disagreement'])
            total.append(dim_data['total_disagreement'])
            simple.append(dim_data['disagreement_rate'])

            # Validation
            if not dim_data.get('decomposition_valid', True):
                print(f"  WARNING: Dimension {dim} decomposition validation failed!")

        print("\nDimension | Strong | Ext-Align | Total | Simple (validation)")
        print("-" * 70)
        for i, dim in enumerate(dimensions):
            print(f"    {dim:2d}    | {strong[i]:5.1f}% | {extreme_aligned[i]:5.1f}%   | {total[i]:5.1f}% | {simple[i]:5.1f}%")

        # Check if they're actually different
        if max(strong) > 0.1:
            ratio = np.mean(extreme_aligned) / (np.mean(strong) + 0.001)
            print(f"\nMean ratio (extreme-aligned / strong): {ratio:.2f}")

            if ratio > 2:
                print("  → Amplification dominant (extreme-aligned >> strong)")
            elif ratio > 0.5:
                print("  → Mixed mechanism (both contribute)")
            else:
                print("  → Creation dominant (strong >> extreme-aligned)")
        else:
            print("\n  → No disagreement detected")


def analyze_metric_pairs(filepath):
    """Analyze metric pair decomposition."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("METRIC PAIR DECOMPOSITION")
    print("=" * 80)

    pairs = data.get('pairs', {})

    for rule in ['plurality', 'borda', 'irv']:
        print(f"\n{rule.upper()}:")
        print("-" * 40)
        print("Pair              | Strong | Ext-Align | Total")
        print("-" * 60)

        for pair_name, pair_data in sorted(pairs.items()):
            if rule not in pair_data:
                continue

            rule_data = pair_data[rule]

            if 'strong_disagreement_ab' in rule_data:
                strong = rule_data['strong_disagreement_ab']
                ext_align = rule_data['extreme_aligned_disagreement_ab']
                total = rule_data.get('total_disagreement_ab', strong + ext_align)
                print(f"{pair_name:17} | {strong:5.1f}% | {ext_align:5.1f}%   | {total:5.1f}%")


def main():
    """Main analysis function."""
    results_dir = Path("heterogenity-simulator/results")

    # Analyze dimensional scaling
    dim_file = results_dir / "dimensional_scaling_l2_cosine_v100.json"
    if dim_file.exists():
        data = load_dimensional_scaling(dim_file)
        analyze_decomposition(data)
    else:
        print(f"Dimensional scaling file not found: {dim_file}")

    # Analyze metric pairs
    pairs_file = results_dir / "metric_pairs_d2_v100.json"
    if pairs_file.exists():
        analyze_metric_pairs(pairs_file)
    else:
        print(f"Metric pairs file not found: {pairs_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
