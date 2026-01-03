#!/usr/bin/env python3
"""
Analyze disagreement decomposition patterns from research results.

This script examines the relationship between:
- Strong disagreement (het disagrees with both baselines)
- Extreme-aligned disagreement (het matches extreme, disagrees with center)

Key questions:
1. What is the typical ratio of strong vs extreme-aligned disagreement?
2. How does this vary by dimension?
3. How does this vary by voting rule?
4. Are there interesting patterns in metric pair asymmetries?
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def analyze_dimensional_scaling(data: dict) -> None:
    """Analyze how disagreement decomposition varies with dimension."""
    print("\n" + "="*80)
    print("DIMENSIONAL SCALING ANALYSIS")
    print("="*80)

    rules = ['plurality', 'borda', 'irv']
    dimensions = data['dimensions']

    # For each rule, track strong/extreme-aligned/total across dimensions
    for rule in rules:
        print(f"\n{rule.upper()}")
        print("-" * 40)
        print(f"{'Dim':<6}{'Strong':<12}{'Extreme-Aln':<15}{'Total':<12}{'Strong%':<12}{'Ext-Aln%':<12}")
        print("-" * 40)

        for dim in dimensions:
            dim_data = data['data'][str(dim)][rule]
            strong = dim_data.get('strong_disagreement', 0)
            extreme = dim_data.get('extreme_aligned_disagreement', 0)
            total = dim_data.get('total_disagreement', 0)

            if total > 0:
                strong_pct = 100 * strong / total
                extreme_pct = 100 * extreme / total
            else:
                strong_pct = 0
                extreme_pct = 0

            print(f"{dim:<6}{strong:<12.1f}{extreme:<15.1f}{total:<12.1f}{strong_pct:<12.1f}{extreme_pct:<12.1f}")

def analyze_metric_pairs(data: dict) -> None:
    """Analyze disagreement patterns for different metric combinations."""
    print("\n" + "="*80)
    print("METRIC PAIR ANALYSIS")
    print("="*80)

    pairs = data['pairs']
    rules = ['plurality', 'borda', 'irv']

    print("\nKey Metric Pairs:")
    print("-" * 80)

    for pair_name in ['l2_cosine', 'l1_cosine', 'cosine_l2']:
        if pair_name not in pairs:
            continue

        print(f"\n{pair_name.upper()} (A->B)")
        print("-" * 40)
        print(f"{'Rule':<15}{'Strong_AB':<15}{'ExtAln_AB':<15}{'Total_AB':<15}{'Strong%':<12}")
        print("-" * 40)

        for rule in rules:
            rule_data = pairs[pair_name][rule]
            strong_ab = rule_data.get('strong_disagreement_ab', 0)
            extreme_ab = rule_data.get('extreme_aligned_disagreement_ab', 0)
            total_ab = rule_data.get('total_disagreement_ab', 0)

            if total_ab > 0:
                strong_pct = 100 * strong_ab / total_ab
            else:
                strong_pct = 0

            print(f"{rule:<15}{strong_ab:<15.1f}{extreme_ab:<15.1f}{total_ab:<15.1f}{strong_pct:<12.1f}")

        # Check asymmetry
        print(f"\nAsymmetry check for {pair_name}:")
        for rule in rules:
            rule_data = pairs[pair_name][rule]
            total_ab = rule_data.get('total_disagreement_ab', 0)
            total_ba = rule_data.get('total_disagreement_ba', 0)
            asymm = abs(total_ab - total_ba)
            print(f"  {rule}: AB={total_ab:.1f}, BA={total_ba:.1f}, |diff|={asymm:.1f}")

def analyze_voter_scaling(data: dict) -> None:
    """Analyze how disagreement changes with voter count."""
    print("\n" + "="*80)
    print("VOTER SCALING ANALYSIS")
    print("="*80)

    voter_counts = data['voter_counts']
    rules = ['plurality', 'borda', 'irv']

    for rule in rules:
        print(f"\n{rule.upper()}")
        print("-" * 40)
        print(f"{'Voters':<10}{'Strong':<12}{'Ext-Aln':<15}{'Total':<12}{'Strong%':<12}")
        print("-" * 40)

        for n_voters in voter_counts:
            voter_data = data['data'][str(n_voters)][rule]
            strong = voter_data.get('strong_disagreement', 0)
            extreme = voter_data.get('extreme_aligned_disagreement', 0)
            total = voter_data.get('total_disagreement', 0)

            if total > 0:
                strong_pct = 100 * strong / total
            else:
                strong_pct = 0

            print(f"{n_voters:<10}{strong:<12.1f}{extreme:<15.1f}{total:<12.1f}{strong_pct:<12.1f}")

def main():
    results_dir = Path('heterogenity-simulator/results')

    # Dimensional scaling
    dim_file = results_dir / 'dimensional_scaling_l2_cosine_v100.json'
    if dim_file.exists():
        dim_data = load_json(dim_file)
        analyze_dimensional_scaling(dim_data)
    else:
        print(f"WARNING: {dim_file} not found")

    # Metric pairs
    pairs_file = results_dir / 'metric_pairs_d2_v100.json'
    if pairs_file.exists():
        pairs_data = load_json(pairs_file)
        analyze_metric_pairs(pairs_data)
    else:
        print(f"WARNING: {pairs_file} not found")

    # Voter scaling
    voter_file = results_dir / 'voter_scaling_l2_cosine_d2.json'
    if voter_file.exists():
        voter_data = load_json(voter_file)
        analyze_voter_scaling(voter_data)
    else:
        print(f"WARNING: {voter_file} not found")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
