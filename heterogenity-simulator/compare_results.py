"""Compare old vs new results to identify changes."""

import json
from pathlib import Path

# Old values from the paper (before correction)
old_dim_data = {
    1: {"plurality": (6.5, 12.0, 18.5), "borda": (6.5, 0.5, 7.0), "irv": (10.5, 5.5, 16.0)},
    2: {"plurality": (6.5, 7.5, 14.0), "borda": (5.0, 3.5, 8.5), "irv": (6.0, 6.5, 12.5)},
    3: {"plurality": (2.5, 7.0, 9.5), "borda": (1.5, 2.5, 4.0), "irv": (1.0, 9.0, 10.0)},
    4: {"plurality": (1.0, 7.5, 8.5), "borda": (8.0, 3.5, 11.5), "irv": (3.5, 10.0, 13.5)},
    5: {"plurality": (4.0, 7.0, 11.0), "borda": (2.0, 6.5, 8.5), "irv": (4.5, 13.0, 17.5)},
    7: {"plurality": (5.0, 13.0, 18.0), "borda": (6.0, 8.0, 14.0), "irv": (1.5, 10.0, 11.5)},
    10: {"plurality": (3.0, 9.5, 12.5), "borda": (4.5, 6.0, 10.5), "irv": (4.0, 9.0, 13.0)},
}

old_vse = {
    1: {"plurality": 0.753, "borda": 0.955, "irv": 0.922},
    2: {"plurality": 0.861, "borda": 0.989, "irv": 0.957},
    3: {"plurality": 0.885, "borda": 0.989, "irv": 0.963},
    4: {"plurality": 0.910, "borda": 0.993, "irv": 0.971},
    5: {"plurality": 0.934, "borda": 0.996, "irv": 0.987},
    7: {"plurality": 0.956, "borda": 0.992, "irv": 0.985},
    10: {"plurality": 0.949, "borda": 0.995, "irv": 0.985},
}

# Load new results
with open("results/dimensional_scaling_l2_cosine_v100.json") as f:
    new_dim_data = json.load(f)

print("=" * 80)
print("COMPARISON: OLD vs NEW RESULTS (Dimensional Scaling)")
print("=" * 80)
print("\nDisagreement Rates (Strong / Extreme-Aligned / Total):")
print("-" * 80)

for dim in [1, 2, 3, 4, 5, 7, 10]:
    print(f"\nDimension {dim}:")
    for rule in ["plurality", "borda", "irv"]:
        old = old_dim_data[dim][rule]
        new = (
            new_dim_data["data"][str(dim)][rule]["strong_disagreement"],
            new_dim_data["data"][str(dim)][rule]["extreme_aligned_disagreement"],
            new_dim_data["data"][str(dim)][rule]["total_disagreement"]
        )
        match = "MATCH" if abs(old[2] - new[2]) < 0.1 else "DIFF"
        diff = new[2] - old[2]
        print(f"  {rule:10} Old: {old[0]:4.1f} / {old[1]:4.1f} / {old[2]:5.1f}%  "
              f"New: {new[0]:4.1f} / {new[1]:4.1f} / {new[2]:5.1f}%  "
              f"Diff={diff:+5.1f}pp {match}")

print("\n" + "=" * 80)
print("VSE Comparison:")
print("-" * 80)

for dim in [1, 2, 3, 4, 5, 7, 10]:
    print(f"\nDimension {dim}:")
    for rule in ["plurality", "borda", "irv"]:
        old_v = old_vse[dim][rule]
        new_v = new_dim_data["data"][str(dim)][rule]["vse_het"]
        diff = new_v - old_v
        match = "MATCH" if abs(diff) < 0.01 else "DIFF"
        print(f"  {rule:10} Old: {old_v:.3f}  New: {new_v:.3f}  Diff={diff:+.3f} {match}")

# Check metric pairs
print("\n" + "=" * 80)
print("KEY FINDINGS FROM NEW ANALYSIS:")
print("=" * 80)

with open("results/metric_pairs_d2_v100.json") as f:
    pairs = json.load(f)

# Borda-Cosine decomposition
cosine_l2 = pairs["pairs"]["cosine_l2"]["borda"]
l2_cosine = pairs["pairs"]["l2_cosine"]["borda"]

print("\n1. BORDA-COSINE ASYMMETRY (Dimension 2):")
print(f"   Cosine->L2: {cosine_l2['total_disagreement_ba']:.1f}% total")
print(f"              {cosine_l2['strong_disagreement_ba']:.1f}% strong ({cosine_l2['strong_disagreement_ba']/cosine_l2['total_disagreement_ba']*100:.1f}% of total)")
print(f"              {cosine_l2['extreme_aligned_disagreement_ba']:.1f}% extreme-aligned ({cosine_l2['extreme_aligned_disagreement_ba']/cosine_l2['total_disagreement_ba']*100:.1f}% of total)")
print(f"   L2->Cosine: {l2_cosine['total_disagreement_ab']:.1f}% total")
print(f"              {l2_cosine['strong_disagreement_ab']:.1f}% strong ({l2_cosine['strong_disagreement_ab']/l2_cosine['total_disagreement_ab']*100:.1f}% of total)")
print(f"              {l2_cosine['extreme_aligned_disagreement_ab']:.1f}% extreme-aligned ({l2_cosine['extreme_aligned_disagreement_ab']/l2_cosine['total_disagreement_ab']*100:.1f}% of total)")
print(f"   Asymmetry: {abs(cosine_l2['total_disagreement_ba'] - l2_cosine['total_disagreement_ab']):.1f}pp")

print("\n2. OTHER EXTREME ASYMMETRIES (Borda, Dimension 2):")
for pair_name in sorted(pairs["pairs"].keys()):
    pair = pairs["pairs"][pair_name]
    borda = pair["borda"]
    asym = abs(borda["total_disagreement_ab"] - borda["total_disagreement_ba"])
    if asym > 40:
        print(f"   {pair_name:20} {borda['total_disagreement_ab']:5.1f}% <-> {borda['total_disagreement_ba']:5.1f}%  "
              f"Asymmetry: {asym:.1f}pp")

print("\n3. DECOMPOSITION INSIGHTS:")
print("   - Cosine->L2 Borda: 80.8% of disagreement is extreme-aligned (amplification)")
print("   - L2->Cosine Borda: 58.8% of disagreement is strong (novel outcomes)")
print("   - Same metric pair, opposite mechanisms depending on direction!")
