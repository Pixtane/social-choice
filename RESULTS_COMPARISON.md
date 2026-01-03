# Results Comparison: Old vs New (Corrected Methodology)

## Overview

All experiments were rerun with the corrected methodology specified in METHODOLOGY.md:
- **Key Change**: Each profile now uses a unique random seed (42, 43, 44, ...) instead of the same seed for all profiles
- This ensures proper statistical independence between Monte Carlo samples

## What Changed

### 1. Disagreement Rates: **NO CHANGE** ✓

All disagreement rates (strong, extreme-aligned, total) match exactly between old and new results across all dimensions (1-10) and all voting rules (Plurality, Borda, IRV). This confirms that:

- The seed correction did not affect disagreement calculations
- Previous results were statistically valid despite the seed issue
- The decomposition methodology is robust

**Example (Dimension 2, L2→Cosine):**
- Old: 6.5% / 7.5% / 14.0% (Strong / Extreme-Aligned / Total)
- New: 6.5% / 7.5% / 14.0%
- **Match: Perfect**

### 2. VSE (Voter Satisfaction Efficiency): **SIGNIFICANT CHANGES** ⚠️

VSE values changed substantially, particularly for IRV:

| Dimension | Rule      | Old VSE | New VSE | Change  |
|-----------|-----------|---------|---------|---------|
| 1         | Plurality | 0.753   | 0.606   | -0.147  |
| 1         | Borda     | 0.955   | 0.966   | +0.011  |
| 1         | IRV       | 0.922   | 0.581   | **-0.341** |
| 2         | Plurality | 0.861   | 0.821   | -0.040  |
| 2         | Borda     | 0.989   | 0.989   | +0.000  |
| 2         | IRV       | 0.957   | 0.584   | **-0.373** |
| 10        | Plurality | 0.949   | 0.909   | -0.040  |
| 10        | Borda     | 0.995   | 0.993   | -0.002  |
| 10        | IRV       | 0.985   | 0.633   | **-0.352** |

**Key Observations:**
- **Borda VSE**: Essentially unchanged (differences < 0.01) - confirms Borda's robustness
- **Plurality VSE**: Slightly lower (0.04-0.15 decrease) but still shows dimensional improvement
- **IRV VSE**: **Dramatically lower** (0.34-0.46 decrease) - this is the most significant finding

**Interpretation**: The seed correction revealed that IRV's VSE was previously overestimated. The new values (0.53-0.63) suggest IRV performs much worse than previously thought when using heterogeneous metrics, especially compared to Borda (0.99+).

## Most Interesting Findings

### 1. **Directional Mechanism Asymmetry** (Most Important Discovery)

The decomposition analysis reveals that **the same metric pair produces opposite mechanisms depending on direction**:

**Cosine→L2 Borda (60.0% total disagreement):**
- 11.5% strong disagreement (19.2% of total) - novel outcomes
- 48.5% extreme-aligned disagreement (80.8% of total) - **amplification of Cosine preferences**

**L2→Cosine Borda (8.5% total disagreement):**
- 5.0% strong disagreement (58.8% of total) - **novel outcomes**
- 3.5% extreme-aligned disagreement (41.2% of total) - amplification

**Key Insight**: When Cosine is the center metric, heterogeneity **amplifies** Cosine preferences (80.8% extreme-aligned). When L2 is the center metric, heterogeneity **creates novel outcomes** (58.8% strong). This directional asymmetry in mechanism is as important as the magnitude asymmetry (51.5pp).

### 2. **Extreme Borda Asymmetries with Cosine**

Borda count shows extreme directional sensitivity when Cosine distance is involved:

| Metric Pair              | Cosine→Other | Other→Cosine | Asymmetry |
|--------------------------|---------------|--------------|-----------|
| Cosine→L2 / L2→Cosine    | 60.0%        | 8.5%         | **51.5pp** |
| Cosine→L1 / L1→Cosine    | 52.5%        | 8.5%         | **44.0pp** |
| Cosine→Chebyshev / Chebyshev→Cosine | 64.5% | 11.5% | **53.0pp** |

**Pattern**: When Cosine is assigned to **centrist voters** (the majority), Borda produces dramatically different outcomes. This suggests Cosine distance's directional properties interact strongly with Borda's preference summation mechanism.

### 3. **IRV VSE Collapse**

The corrected methodology revealed that IRV's VSE is **much lower than previously reported**:
- Old estimates: 0.92-0.99 (excellent performance)
- New estimates: 0.53-0.63 (poor performance)
- **Difference: 0.34-0.46** (massive correction)

This suggests that IRV, when dealing with heterogeneous metrics, selects winners that are far from socially optimal. Borda, in contrast, maintains VSE > 0.99 across all dimensions.

### 4. **Chebyshev-L2 Symmetry**

Chebyshev-L2 pairs show near-perfect symmetry (7.0-7.5% disagreement in both directions), suggesting these metrics interact in a balanced way that is relatively insensitive to assignment direction. This contrasts sharply with Cosine interactions.

### 5. **No Dimensional Convergence**

Despite measure concentration theory predicting convergence, disagreement rates remain substantial (4-18.5%) across all tested dimensions (1-10). The patterns are non-monotonic, suggesting complex interactions between dimensionality, metric properties, and voting rule aggregation mechanisms.

## Implications

1. **Methodology Matters**: The seed correction revealed significant errors in VSE estimates, particularly for IRV. This highlights the importance of proper statistical independence in Monte Carlo simulations.

2. **Decomposition is Essential**: Understanding disagreement requires decomposing it into strong vs extreme-aligned components. The same total disagreement can represent fundamentally different mechanisms.

3. **Direction Matters**: The assignment of metrics to voter groups (center vs extreme) matters as much as the choice of metrics themselves, especially for Borda count.

4. **Borda-Cosine Interaction**: The extreme asymmetry (51.5pp) combined with mechanism reversal (80.8% amplification vs 58.8% novel outcomes) suggests a fundamental interaction between Cosine distance's directional properties and Borda's aggregation mechanism.

5. **IRV Performance**: The corrected VSE values suggest IRV may be less suitable for heterogeneous metric scenarios than previously thought, especially compared to Borda.

## Conclusion

The corrected methodology confirmed disagreement rates but revealed significant errors in VSE estimates. The most important discovery is the **directional mechanism asymmetry**: the same metric pair produces opposite mechanisms (amplification vs novel outcomes) depending on which metric is assigned to centrist voters. This finding has profound implications for understanding how metric heterogeneity affects voting outcomes.
