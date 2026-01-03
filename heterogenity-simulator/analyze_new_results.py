"""Quick script to extract key findings from new results."""

import json
from pathlib import Path

# Load results
results_dir = Path("results")

# Dimensional scaling
with open(results_dir / "dimensional_scaling_l2_cosine_v100.json") as f:
    dim_data = json.load(f)

print("=" * 80)
print("DIMENSIONAL SCALING (L2->Cosine, threshold=0.5)")
print("=" * 80)
print(f"{'Dim':<6} {'Rule':<12} {'Strong':<8} {'Ext-Align':<10} {'Total':<8}")
print("-" * 80)
for dim in dim_data["dimensions"]:
    for rule in ["plurality", "borda", "irv"]:
        data = dim_data["data"][str(dim)][rule]
        print(f"{dim:<6} {rule:<12} {data['strong_disagreement']:<8.1f} "
              f"{data['extreme_aligned_disagreement']:<10.1f} {data['total_disagreement']:<8.1f}")

# Metric pairs
with open(results_dir / "metric_pairs_d2_v100.json") as f:
    pairs_data = json.load(f)

print("\n" + "=" * 80)
print("METRIC PAIR INTERACTIONS (Dimension 2, threshold=0.5)")
print("=" * 80)
print(f"{'Pair':<20} {'Rule':<12} {'A->B Total':<12} {'B->A Total':<12} {'Asymmetry':<10}")
print("-" * 80)

for pair_name in sorted(pairs_data["pairs"].keys()):
    pair_data = pairs_data["pairs"][pair_name]
    for rule in ["plurality", "borda", "irv"]:
        rule_data = pair_data[rule]
        ab = rule_data["total_disagreement_ab"]
        ba = rule_data["total_disagreement_ba"]
        asym = abs(ab - ba)
        print(f"{pair_name:<20} {rule:<12} {ab:<12.1f} {ba:<12.1f} {asym:<10.1f}")

# Check for Borda-Cosine extreme asymmetry
print("\n" + "=" * 80)
print("BORDA-COSINE INTERACTION DETAILS")
print("=" * 80)
cosine_l2 = pairs_data["pairs"]["cosine_l2"]["borda"]
l2_cosine = pairs_data["pairs"]["l2_cosine"]["borda"]
print(f"Cosine->L2: Strong={cosine_l2['strong_disagreement_ab']:.1f}%, "
      f"Ext-Align={cosine_l2['extreme_aligned_disagreement_ab']:.1f}%, "
      f"Total={cosine_l2['total_disagreement_ab']:.1f}%")
print(f"L2->Cosine: Strong={l2_cosine['strong_disagreement_ab']:.1f}%, "
      f"Ext-Align={l2_cosine['extreme_aligned_disagreement_ab']:.1f}%, "
      f"Total={l2_cosine['total_disagreement_ab']:.1f}%")
print(f"Asymmetry: {abs(cosine_l2['total_disagreement_ab'] - l2_cosine['total_disagreement_ab']):.1f}pp")
