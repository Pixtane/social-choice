#!/usr/bin/env python3
"""
Find and display a concrete example of Borda strong disagreement.

Strong disagreement = heterogeneous winner differs from BOTH baselines.
"""
import sys
sys.path.insert(0, 'heterogenity-simulator')
sys.path.insert(0, 'simulator')

from simulator.main import SimulationConfig, run_experiment
from simulator.config import GeometryConfig, UtilityConfig, HeterogeneousDistanceConfig
import numpy as np

# Run a small experiment to find examples
config = SimulationConfig(
    n_profiles=50,  # Small number for quick execution
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

print("="*80)
print("SEARCHING FOR BORDA STRONG DISAGREEMENT EXAMPLES")
print("="*80)
print(f"\nRunning {config.n_profiles} profiles with L2->Cosine heterogeneity...")
print("Dimension: 2, Voters: 100, Rule: Borda\n")

# Run heterogeneous
result_het = run_experiment(config, save_results=False, verbose=False)

# Run center baseline (L2 homogeneous)
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

# Run extreme baseline (Cosine homogeneous)
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

print("Experiments complete. Analyzing winners...\n")

# Find examples of each type
rule = 'borda'
het_winners = result_het.rule_results[rule].winners
center_winners = result_center.rule_results[rule].winners
extreme_winners = result_extreme.rule_results[rule].winners

examples_found = {
    'strong': [],
    'extreme_aligned': [],
    'agreement': []
}

for i in range(len(het_winners)):
    w_het = het_winners[i]
    w_center = center_winners[i]
    w_extreme = extreme_winners[i]

    if w_het != w_center and w_het != w_extreme:
        # Strong disagreement
        examples_found['strong'].append((i, w_het, w_center, w_extreme))
    elif w_het == w_extreme and w_het != w_center:
        # Extreme-aligned disagreement
        examples_found['extreme_aligned'].append((i, w_het, w_center, w_extreme))
    elif w_het == w_center and w_het == w_extreme:
        # Agreement
        examples_found['agreement'].append((i, w_het, w_center, w_extreme))

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Strong Disagreement:        {len(examples_found['strong'])} profiles ({100*len(examples_found['strong'])/config.n_profiles:.1f}%)")
print(f"Extreme-Aligned Disagreement: {len(examples_found['extreme_aligned'])} profiles ({100*len(examples_found['extreme_aligned'])/config.n_profiles:.1f}%)")
print(f"Agreement (all same):        {len(examples_found['agreement'])} profiles ({100*len(examples_found['agreement'])/config.n_profiles:.1f}%)")

# Show first example of strong disagreement
if examples_found['strong']:
    print("\n" + "="*80)
    print("EXAMPLE OF STRONG DISAGREEMENT")
    print("="*80)

    profile_idx, w_het, w_center, w_extreme = examples_found['strong'][0]

    print(f"\nProfile #{profile_idx}:")
    print(f"  Heterogeneous winner (L2->Cosine):     Candidate {w_het}")
    print(f"  Center baseline winner (L2 only):     Candidate {w_center}")
    print(f"  Extreme baseline winner (Cosine only): Candidate {w_extreme}")
    print(f"\n  -> Heterogeneous winner differs from BOTH baselines!")
    print(f"  -> This is STRONG DISAGREEMENT")

    # Show a few more examples if available
    if len(examples_found['strong']) > 1:
        print(f"\n  Additional strong disagreement profiles: ", end="")
        for i in range(1, min(6, len(examples_found['strong']))):
            profile_idx, w_het, w_center, w_extreme = examples_found['strong'][i]
            print(f"#{profile_idx} (Het:{w_het} vs Center:{w_center} vs Extreme:{w_extreme})", end="; ")
        print()

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"""
This profile demonstrates STRONG DISAGREEMENT in Borda count under metric heterogeneity:

* When voters use mixed metrics (L2 for centrists, Cosine for extremists),
  Candidate {w_het} wins with the highest Borda score.

* When ALL voters use L2 distance (center baseline),
  Candidate {w_center} wins instead.

* When ALL voters use Cosine distance (extreme baseline),
  Candidate {w_extreme} wins instead.

The heterogeneous outcome is NOT explained by simply favoring the extreme-metric
baseline. Instead, the interaction between center and extreme voters using
different metrics creates a GENUINELY NOVEL outcome that neither homogeneous
baseline would produce.

This is the signature of strong disagreement: heterogeneity creates emergent
outcomes beyond what either metric alone would produce.
""")

else:
    print("\nNo strong disagreement examples found in this sample.")
    print("Try increasing n_profiles or changing the random seed.")

if examples_found['extreme_aligned']:
    print("\n" + "="*80)
    print("CONTRAST: EXAMPLE OF EXTREME-ALIGNED DISAGREEMENT")
    print("="*80)

    profile_idx, w_het, w_center, w_extreme = examples_found['extreme_aligned'][0]

    print(f"\nProfile #{profile_idx}:")
    print(f"  Heterogeneous winner (L2->Cosine):     Candidate {w_het}")
    print(f"  Center baseline winner (L2 only):     Candidate {w_center}")
    print(f"  Extreme baseline winner (Cosine only): Candidate {w_extreme}")
    print(f"\n  -> Heterogeneous winner MATCHES extreme baseline")
    print(f"  -> This is EXTREME-ALIGNED DISAGREEMENT (not strong)")
    print(f"\n  Interpretation: Heterogeneity shifts outcome toward the extreme metric,")
    print(f"  but doesn't create a genuinely novel winner.")
