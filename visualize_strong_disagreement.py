#!/usr/bin/env python3
"""
Visualize a profile with strong disagreement showing spatial positions
and how different metrics lead to different winners.
"""
import sys
sys.path.insert(0, 'heterogenity-simulator')
sys.path.insert(0, 'simulator')

from simulator.main import SimulationConfig, run_experiment
from simulator.config import GeometryConfig, UtilityConfig, HeterogeneousDistanceConfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Run experiment to find a strong disagreement example
config = SimulationConfig(
    n_profiles=50,
    n_voters=100,
    n_candidates=5,
    voting_rules=['borda'],
    geometry=GeometryConfig(method='uniform', n_dim=2, position_min=-1.0, position_max=1.0),
    utility=UtilityConfig(
        function='linear',
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme',
            center_metric='l2',
            extreme_metric='cosine',
            center_extreme_threshold_mode='percentile',
            extreme_threshold=0.5
        )
    ),
    rng_seed=42
)

print("Running experiments to find strong disagreement example...")
result_het = run_experiment(config, save_results=False, verbose=False)

config_center = SimulationConfig(
    n_profiles=config.n_profiles,
    n_voters=config.n_voters,
    n_candidates=config.n_candidates,
    voting_rules=['borda'],
    geometry=GeometryConfig(method='uniform', n_dim=2, position_min=-1.0, position_max=1.0),
    utility=UtilityConfig(function='linear', distance_metric='l2'),
    rng_seed=42
)
result_center = run_experiment(config_center, save_results=False, verbose=False)

config_extreme = SimulationConfig(
    n_profiles=config.n_profiles,
    n_voters=config.n_voters,
    n_candidates=config.n_candidates,
    voting_rules=['borda'],
    geometry=GeometryConfig(method='uniform', n_dim=2, position_min=-1.0, position_max=1.0),
    utility=UtilityConfig(function='linear', distance_metric='cosine'),
    rng_seed=42
)
result_extreme = run_experiment(config_extreme, save_results=False, verbose=False)

# Find strong disagreement example
rule = 'borda'
het_winners = result_het.rule_results[rule].winners
center_winners = result_center.rule_results[rule].winners
extreme_winners = result_extreme.rule_results[rule].winners

profile_idx = None
for i in range(len(het_winners)):
    if (het_winners[i] != center_winners[i] and
        het_winners[i] != extreme_winners[i]):
        profile_idx = i
        break

if profile_idx is None:
    print("No strong disagreement found, using first profile")
    profile_idx = 0

w_het = het_winners[profile_idx]
w_center = center_winners[profile_idx]
w_extreme = extreme_winners[profile_idx]

print(f"\nVisualizing Profile #{profile_idx}")
print(f"  Heterogeneous: Candidate {w_het}")
print(f"  L2-only:       Candidate {w_center}")
print(f"  Cosine-only:   Candidate {w_extreme}")

# Get spatial data for this profile
voter_pos = result_het.preferences.voter_positions[profile_idx]  # (n_voters, 2)
cand_pos = result_het.preferences.candidate_positions[profile_idx]  # (n_candidates, 2)

# Compute centrality for each voter
centrality = np.linalg.norm(voter_pos, axis=1)
threshold_val = np.percentile(centrality, 50)
is_center = centrality <= threshold_val

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Borda Strong Disagreement Example (Profile #{profile_idx})',
             fontsize=16, fontweight='bold')

for ax_idx, (ax, title, winner) in enumerate([
    (axes[0], 'Heterogeneous\n(L2 center, Cosine extreme)', w_het),
    (axes[1], 'L2 Homogeneous\n(All voters use L2)', w_center),
    (axes[2], 'Cosine Homogeneous\n(All voters use Cosine)', w_extreme)
]):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Policy Dimension 1')
    if ax_idx == 0:
        ax.set_ylabel('Policy Dimension 2')

    # Draw threshold circle for heterogeneous case
    if ax_idx == 0:
        circle = Circle((0, 0), threshold_val, fill=False,
                       linestyle='--', color='gray', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)
        ax.text(threshold_val * 0.7, threshold_val * 0.7,
               'Center/Extreme\nThreshold',
               fontsize=8, color='gray', ha='center')

    # Plot voters
    if ax_idx == 0:  # Heterogeneous - show metric assignment
        center_voters = voter_pos[is_center]
        extreme_voters = voter_pos[~is_center]
        ax.scatter(center_voters[:, 0], center_voters[:, 1],
                  c='steelblue', s=30, alpha=0.6,
                  marker='o', label='Center voters (L2)')
        ax.scatter(extreme_voters[:, 0], extreme_voters[:, 1],
                  c='coral', s=30, alpha=0.6,
                  marker='^', label='Extreme voters (Cosine)')
    else:  # Homogeneous
        ax.scatter(voter_pos[:, 0], voter_pos[:, 1],
                  c='gray', s=30, alpha=0.4,
                  marker='o', label='Voters')

    # Plot candidates
    for cand_idx in range(len(cand_pos)):
        if cand_idx == winner:
            # Winner: large star with gold edge
            ax.scatter(cand_pos[cand_idx, 0], cand_pos[cand_idx, 1],
                      c='gold', s=500, marker='*',
                      edgecolors='darkgoldenrod', linewidth=2,
                      zorder=100)
            ax.text(cand_pos[cand_idx, 0], cand_pos[cand_idx, 1] - 0.15,
                   f'C{cand_idx}\nWINNER',
                   fontsize=10, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='gold', alpha=0.8))
        else:
            # Non-winner: smaller circle
            ax.scatter(cand_pos[cand_idx, 0], cand_pos[cand_idx, 1],
                      c='white', s=200, marker='o',
                      edgecolors='black', linewidth=2,
                      zorder=50)
            ax.text(cand_pos[cand_idx, 0], cand_pos[cand_idx, 1],
                   f'C{cand_idx}',
                   fontsize=9, ha='center', va='center',
                   fontweight='bold')

    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()

# Save figure
output_file = 'strong_disagreement_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

# Show the plot
plt.show()

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"""
This visualization shows the SAME electoral profile viewed through three lenses:

LEFT: Heterogeneous (L2 center, Cosine extreme)
  - Blue circles (o) = centrist voters using L2 (Euclidean) distance
  - Red triangles (^) = extreme voters using Cosine (angular) distance
  - Winner: Candidate {w_het}

MIDDLE: L2 Homogeneous (all voters use L2)
  - All voters use Euclidean distance
  - Winner: Candidate {w_center}

RIGHT: Cosine Homogeneous (all voters use Cosine)
  - All voters use angular/directional distance
  - Winner: Candidate {w_extreme}

KEY INSIGHT: Candidate {w_het} ONLY wins in the heterogeneous scenario!
This is STRONG DISAGREEMENT - the mixed-metric outcome differs from both
homogeneous baselines, showing that metric diversity creates genuinely
emergent outcomes beyond what either metric alone would produce.
""")
