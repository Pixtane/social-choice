# Visualization Script for Research Findings

This script generates publication-quality visualizations for all 5 novel phenomena described in `FINDINGS.md`.

## Usage

Run the visualization script:

```bash
python heterogeneity-research/visualize_findings.py
```

## What It Does

The script runs simulations at research scale (10,000 voters per profile) and generates 5 visualizations:

1. **Asymmetric Metric Interaction** (`1_asymmetric_metric_interaction.png`)
   - Shows how the order of metric assignment (A→B vs B→A) creates different disagreement rates
   - Tests L1↔Cosine, L1↔Chebyshev, and L1↔L2 pairs

2. **Dimensional Scaling Laws** (`2_dimensional_scaling.png`)
   - Shows how heterogeneity effects scale with dimensionality
   - Demonstrates peak effects at 2-3 dimensions
   - Tests dimensions: 1, 2, 3, 4, 5, 7, 10

3. **Threshold Phase Transitions** (`3_threshold_phase_transitions.png`)
   - Shows sigmoidal response curves with inflection points
   - Tests thresholds from 0.05 to 0.95 in 0.05 increments
   - Includes sigmoid fits (requires scipy, optional)

4. **Preference Structure Destabilization Paradox** (`4_preference_destabilization.png`)
   - Shows the paradox: increased cycle rates but improved Condorcet efficiency
   - Compares homogeneous vs heterogeneous for Plurality, Borda, and IRV

5. **Metric Interaction Strength Hierarchy** (`5_metric_interaction_hierarchy.png`)
   - Heatmap showing interaction strength for all metric pairs
   - Focuses on Plurality rule as in FINDINGS.md

## Parameters

- **Voters per profile**: 10,000 (research scale)
- **Profiles**: 200 (for statistical significance)
- **Candidates**: 5
- **Dimensions**: Varies by visualization (typically 2D)

## Output

All images are saved to `heterogeneity-research/visualizations/` directory as PNG files with 300 DPI resolution.

## Runtime

This script will take a significant amount of time to run (potentially hours) due to:
- Large number of voters (10,000 per profile)
- Multiple experiments across different configurations
- 200 profiles per configuration for statistical significance

## Dependencies

- numpy
- matplotlib
- scipy (optional, for sigmoid fitting in visualization 3)

## Notes

- The script uses a fixed random seed for reproducibility (via SimulationConfig defaults)
- All visualizations use consistent color schemes for metrics and voting rules
- Images are saved with publication-quality resolution (300 DPI)






