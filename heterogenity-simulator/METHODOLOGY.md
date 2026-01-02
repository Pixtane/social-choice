# Research Methodology: Heterogeneous Distance Metrics in Spatial Voting

## Overview

This document describes the comprehensive research methodology used to investigate phenomena in heterogeneous distance metrics for spatial voting models. The research systematically explores how different voters using different distance metrics affects voting outcomes, with rigorous experimental design and statistical analysis.

## Research Objectives

1. **Verify and correct** original findings from FINDINGS.md
2. **Discover new phenomena** in heterogeneous distance metrics
3. **Understand voter scaling effects** - how results change with voter count
4. **Characterize phase transitions** in threshold parameter space
5. **Map dimensional scaling laws** for heterogeneity effects
6. **Quantify asymmetric interactions** between metric pairs
7. **Investigate Condorcet paradox** - cycle rates vs efficiency

## Experimental Design

### Base Configuration

- **Minimum profiles for conclusions**: 200 profiles
- **Minimum voters for conclusions**: 100 voters
- **Candidates per profile**: 5
- **Voting rules tested**: Plurality, Borda, IRV
- **Random seed**: 42 (for reproducibility)
- **Geometry domain**: Voters and candidates are sampled uniformly from the hypercube $[-1,1]^{d}$.
  - This enables cosine distance to express the full angular range ($0^\circ$ to $180^\circ$), instead of being limited by the first orthant of $[0,1]^d$.

### Voter Scaling Analysis

**Purpose**: Determine if heterogeneity effects are stable across voter counts or if they change systematically.

**Method**:

- Test voter counts: [10, 25, 50, 100, 200, 300, 400, 500]
- For each voter count:
  - Run 200 profiles with heterogeneous metrics (L2 center, Cosine extreme, threshold=0.5)
  - Run 200 profiles with homogeneous baseline (L2 for all)
  - Compare winners to compute disagreement rates
  - Compute Condorcet metrics (cycle rates, efficiency)

**Analysis**:

- Check coefficient of variation (CV) to identify stable vs changing effects
- Fit linear trends to detect systematic changes
- Identify critical voter counts where effects change

### Threshold Sweep Experiment

**Purpose**: Detect phase transitions and critical thresholds in the center-extreme assignment parameter.

**Method**:

- Test thresholds: 0.05 to 0.95 in 0.05 increments (19 points)
- For each threshold:
  - Run 200 profiles with heterogeneous metrics
  - Run 200 profiles with homogeneous baseline
  - Compute disagreement rates and Condorcet metrics

**Analysis**:

- Identify inflection points (second derivative sign changes)
- Find maximum curvature points (phase transitions)
- Detect sudden jumps (discontinuities)
- Fit sigmoidal curves if appropriate

### Dimensional Scaling Experiment

**Purpose**: Understand how heterogeneity effects scale with spatial dimensionality.

**Method**:

- Test dimensions: [1, 2, 3, 4, 5, 7, 10]
- For each dimension:
  - Run 200 profiles with heterogeneous metrics (L2 center, Cosine extreme, threshold=0.5)
  - Run 200 profiles with homogeneous baseline
  - Compute disagreement rates

**Analysis**:

- Identify peak dimension (maximum effect)
- Fit power law for pre-peak dimensions: D(d) ~ d^α
- Compute post-peak slope (plateau or decline)
- Calculate R² for power law fits

### Metric Pair Interaction Experiment

**Purpose**: Quantify asymmetric interactions between all metric pairs.

**Method**:

- Test all ordered pairs of metrics: L1, L2, Cosine, Chebyshev (12 pairs)
- For each pair (A, B):
  - Run 200 profiles with A (center) -> B (extreme)
  - Run 200 profiles with B (center) -> A (extreme)
  - Run 200 profiles with homogeneous baseline (A)
  - Compute disagreement rates for both directions
  - Calculate asymmetry: |D(A->B) - D(B->A)|

**Analysis**:

- Build interaction strength hierarchy
- Identify strongest and weakest metric pairs
- Quantify asymmetry magnitudes
- Test if asymmetry is consistent across voting rules

### Final Verification

**Purpose**: Verify conclusions with high-voter-count experiments.

**Method**:

- Re-run metric pair experiments with 500 voters
- Compare results to 100-voter experiments
- Check if conclusions hold or change

## Statistical Methods

### Disagreement Rate

The primary metric is the **disagreement rate**: the percentage of profiles where the heterogeneous winner differs from the homogeneous winner.

$$D = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[w_i^{het} \neq w_i^{homo}] \times 100\%$$

where $w_i^{het}$ and $w_i^{homo}$ are the winners for profile $i$ under heterogeneous and homogeneous conditions.

### Condorcet Metrics

1. **Cycle Rate**: Percentage of profiles with Condorcet cycles (no Condorcet winner)
2. **Condorcet Efficiency**: Percentage of profiles where the voting rule selects the Condorcet winner (when one exists)

### Asymmetry Calculation

For metric pair (A, B):
$$\Delta_{A,B} = |D_{A \to B} - D_{B \to A}|$$

where $D_{A \to B}$ is disagreement when A is assigned to center voters and B to extreme voters.

### Power Law Fitting

For dimensional scaling, fit power law:
$$D(d) = D_0 \cdot d^{\alpha}$$

Using log-log linear regression:
$$\log D(d) = \log D_0 + \alpha \log d$$

### Phase Transition Detection

1. **Inflection Points**: Where second derivative changes sign
   $$\frac{d^2D}{d\theta^2} = 0$$
2. **Maximum Curvature**: Maximum of $|\frac{d^2D}{d\theta^2}|$

3. **Sudden Jumps**: Maximum of $|\frac{dD}{d\theta}|$

## Validation Criteria

### Minimum Sample Sizes

- **Profiles**: Minimum 200 for statistical significance
- **Voters**: Minimum 100 for stable results (verified through scaling analysis)

### Reproducibility

- Fixed random seed (42) for all experiments
- Same geometry generation method (uniform) across experiments
- Consistent utility function (linear) for all experiments

### Statistical Significance

- Report means, standard deviations, and ranges
- Identify trends with slope analysis
- Use coefficient of variation to assess stability

## Experimental Phases

### Phase 1: Voter Scaling Analysis

- Test 8 different voter counts (10-500)
- Identify stable vs changing effects
- Determine appropriate voter count for main experiments

### Phase 2: Threshold Sweep

- Fine-grained threshold analysis (19 points)
- Detect phase transitions
- Identify critical thresholds

### Phase 3: Dimensional Scaling

- Test 7 different dimensions (1-10)
- Fit scaling laws
- Identify peak dimensions

### Phase 4: Metric Pair Interactions

- Test all 12 ordered metric pairs
- Quantify asymmetries
- Build interaction hierarchy

### Phase 5: Final Verification

- Re-run key experiments with 500 voters
- Verify conclusions hold at higher voter counts
- Check for voter-count-dependent effects

## Output Files

All results are saved as JSON files in `heterogenity-simulator/results/`:

- `voter_scaling_*.json`: Voter scaling results
- `threshold_sweep_*.json`: Threshold sweep results
- `dimensional_scaling_*.json`: Dimensional scaling results
- `metric_pairs_*.json`: Metric pair interaction results
- `full_research_suite.json`: Complete results summary

## Analysis Pipeline

1. **Run experiments** using `research_suite.py`
2. **Analyze results** using `analyze_results.py`
3. **Generate findings** in `FINDINGS-2.md`
4. **Document methodology** in this file

## Quality Assurance

### Pre-Experiment Checks

- Verify imports work correctly
- Test with small sample sizes first
- Check that heterogeneous distance is actually enabled

### During Experiments

- Monitor computation time
- Check for errors or warnings
- Verify output files are created

### Post-Experiment Validation

- Compare results across different voter counts
- Check for consistency with original findings
- Identify discrepancies requiring investigation

## Limitations

1. **Computational constraints**: Full suite takes several hours
2. **Sample size trade-offs**: More profiles = more time, but better statistics
3. **Fixed random seed**: Results are deterministic but may not generalize
4. **Uniform geometry**: Only tests uniform spatial distribution

## Future Extensions

1. Test with different geometry methods (polarized, clustered)
2. Test with different utility functions (gaussian, quadratic)
3. Test with more voting rules (approval, STAR, Schulze)
4. Test with more metrics (Minkowski p-norms)
5. Statistical significance testing (bootstrap, confidence intervals)

## References

- Original findings: `heterogeneity-research/FINDINGS.md`
- Research code: `heterogenity-simulator/research_suite.py`
- Analysis code: `heterogenity-simulator/analyze_results.py`
- Simulator code: `simulator/` directory
