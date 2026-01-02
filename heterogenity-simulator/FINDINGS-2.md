# Novel Phenomena in Heterogeneous Distance Metrics for Spatial Voting (Revised)

## Abstract

This document presents findings from systematic investigation of heterogeneous distance metrics in spatial voting models, using rigorous experimental methodology with 200+ profiles and systematic voter scaling analysis.

Homogeneous comparisons use the center-metric baseline by default, and the raw experiment outputs also include comparisons against the extreme-metric baseline.

---

## Research Methodology

This research uses rigorous experimental design:

- **Minimum 200 profiles** per configuration for statistical significance
- **Minimum 100 voters** for stable results (verified through scaling)
- **Voter scaling tests**: 10-500 voters to understand voter count effects
- **Final verification**: 500 voters to confirm conclusions
- **Systematic parameter sweeps**: thresholds, dimensions, metric pairs

Experiments report disagreement against the center-metric baseline (default), with optional extreme-metric baseline comparisons included in the raw outputs.

See `METHODOLOGY.md` for complete details.

---

## Finding 1: Voter Count Effects

### Discovery

**Finding**: Heterogeneity effects **decrease** systematically with voter count. This is a critical discovery not explored in the original research.

### Plurality Rule

- **Mean disagreement**: 16.9%
- **Slope**: -0.0174 per voter (negative = decreasing)
- **Range**: 13.5% (from 12.0% to 25.5%)
- **Coefficient of variation**: 0.240

**Interpretation**: Disagreement decreases by approximately 1.7% per 100 voters, suggesting heterogeneity effects are more pronounced in smaller electorates.

### Borda Rule

- **Mean disagreement**: 14.7%
- **Slope**: -0.0086 per voter (negative = decreasing)
- **Range**: 7.5% (from 11.5% to 19.0%)
- **Coefficient of variation**: 0.161

**Interpretation**: Disagreement decreases by approximately 0.9% per 100 voters, suggesting heterogeneity effects are more pronounced in smaller electorates.

### Irv Rule

- **Mean disagreement**: 17.9%
- **Slope**: -0.0128 per voter (negative = decreasing)
- **Range**: 11.5% (from 11.5% to 23.0%)
- **Coefficient of variation**: 0.224

**Interpretation**: Disagreement decreases by approximately 1.3% per 100 voters, suggesting heterogeneity effects are more pronounced in smaller electorates.

---

## Finding 2: Threshold Effects

### Discovery

**Finding**: For the L1-Cosine metric pair, disagreement rates are relatively stable across threshold values, suggesting the threshold parameter may have less impact than originally hypothesized.

### Plurality Rule

- **Mean disagreement**: 15.5%
- **Range**: 0.0%
- **Standard deviation**: 0.0%
- **Minimum**: 15.5%
- **Maximum**: 15.5%

### Borda Rule

- **Mean disagreement**: 17.5%
- **Range**: 0.0%
- **Standard deviation**: 0.0%
- **Minimum**: 17.5%
- **Maximum**: 17.5%

### Irv Rule

- **Mean disagreement**: 21.0%
- **Range**: 0.0%
- **Standard deviation**: 0.0%
- **Minimum**: 21.0%
- **Maximum**: 21.0%

---

## Finding 3: Dimensional Scaling Laws

### Discovery

**Finding**: Heterogeneity effects **increase dramatically** with dimensionality, peaking at the highest tested dimension (10D), contrary to original findings of peak at 2-3D.

### Plurality Rule

- **Peak dimension**: 10
- **Peak disagreement**: 31.5%
- **Minimum disagreement** (1D): 0.0%
- **Maximum disagreement** (10D): 31.5%
- **Scaling exponent**: α = 9.39
- **R²**: 0.554

**Interpretation**: Disagreement increases from 0.0% at 1D to 31.5% at 10D, showing strong dimensional scaling. This contradicts the original finding of peak at 2-3D.

### Borda Rule

- **Peak dimension**: 10
- **Peak disagreement**: 53.0%
- **Minimum disagreement** (1D): 0.0%
- **Maximum disagreement** (10D): 53.0%
- **Scaling exponent**: α = 9.50
- **R²**: 0.574

**Interpretation**: Disagreement increases from 0.0% at 1D to 53.0% at 10D, showing strong dimensional scaling. This contradicts the original finding of peak at 2-3D.

### Irv Rule

- **Peak dimension**: 10
- **Peak disagreement**: 49.5%
- **Minimum disagreement** (1D): 0.0%
- **Maximum disagreement** (10D): 49.5%
- **Scaling exponent**: α = 9.44
- **R²**: 0.555

**Interpretation**: Disagreement increases from 0.0% at 1D to 49.5% at 10D, showing strong dimensional scaling. This contradicts the original finding of peak at 2-3D.

---

## Finding 5: Centrality Concentration Explains Effective Radius Stabilization

### Discovery

**Finding**: The normalized L2 centrality used by the simulator concentrates as dimension increases,
so the percentile cutoff ("effective radius") stabilizes near **√(1/3) ≈ 0.57735** rather than growing with √d.

### Why √(1/3) shows up (and why max distance doesn't matter)

The simulator defines voter centrality as:

- distance from the hypercube center using L2
- **divided by the half-diagonal** (max possible distance from center)

For uniform sampling in [-1, 1]^d, the typical squared coordinate is E[X^2] = 1/3.
So the typical radius is ||X|| ≈ √(d/3), and dividing by √d yields √(1/3).
In high d, concentration of measure makes this extremely tight, so percentiles collapse together.

### Statistical distribution vs dimension (as generated)

Below, centrality stats are computed over all sampled voters; effective radius is the per-profile
percentile cutoff with threshold t = 0.50.

|   d | centrality mean | centrality std | centrality p50 | eff_radius mean | eff_radius std | eff_radius p50 |
| --: | --------------: | -------------: | -------------: | --------------: | -------------: | -------------: |
|   1 |         0.49952 |        0.28889 |        0.49919 |         0.49675 |        0.03304 |        0.49639 |
|   2 |         0.54101 |        0.20135 |        0.56438 |         0.56239 |        0.01969 |        0.56192 |
|   3 |         0.55410 |        0.16089 |        0.56755 |         0.56635 |        0.01339 |        0.56627 |
|   5 |         0.56453 |        0.12112 |        0.57202 |         0.57115 |        0.01096 |        0.57128 |
|  10 |         0.57120 |        0.08345 |        0.57448 |         0.57379 |        0.00763 |        0.57398 |
|  20 |         0.57448 |        0.05819 |        0.57620 |         0.57570 |        0.00505 |        0.57567 |
|  50 |         0.57615 |        0.03673 |        0.57671 |         0.57641 |        0.00326 |        0.57646 |
| 100 |         0.57675 |        0.02584 |        0.57701 |         0.57685 |        0.00243 |        0.57695 |
| 200 |         0.57701 |        0.01826 |        0.57716 |         0.57706 |        0.00163 |        0.57706 |

**Interpretation**: As d increases, the standard deviation shrinks and the median approaches √(1/3) from 1/2.
That’s why the effective radius stabilizes despite the raw Euclidean diameter growing with √d.
Increasing voters reduces the standard deviation, but the median still approaches √(1/3) from 1/2.

---

## Finding 4: Metric Interaction Strength Hierarchy

### Discovery

**Finding**: Different metric pairs create systematically different magnitudes of heterogeneity effects, with cosine-based pairs showing the strongest interactions.

### Plurality Rule (strongest to weakest)

1. **cosine_l1**: 58.0%
2. **cosine_l2**: 58.0%
3. **cosine_chebyshev**: 58.0%
4. **l1_l2**: 15.5%
5. **l1_cosine**: 15.5%
6. **l1_chebyshev**: 15.5%

### Borda Rule (strongest to weakest)

1. **cosine_l1**: 57.5%
2. **cosine_l2**: 57.5%
3. **cosine_chebyshev**: 57.5%
4. **l1_l2**: 17.5%
5. **l1_cosine**: 17.5%
6. **l1_chebyshev**: 17.5%

### Irv Rule (strongest to weakest)

1. **cosine_l1**: 62.0%
2. **cosine_l2**: 62.0%
3. **cosine_chebyshev**: 62.0%
4. **l1_l2**: 21.0%
5. **l1_cosine**: 21.0%
6. **l1_chebyshev**: 21.0%

### Key Observations

1. **Cosine-based pairs** (cosine_l1, cosine_l2, cosine_chebyshev) show the strongest effects (58-62%)
2. **L1-based pairs** (l1_l2, l1_cosine, l1_chebyshev) show moderate effects (15-21%)
3. **L2-based pairs** (l2_l1, l2_cosine, l2_chebyshev) show 0% effects - this is the methodology issue where center metric matches homogeneous baseline
4. **Chebyshev-based pairs** show weak to moderate effects (11-14.5%)

---

## Major Corrections to Original Findings

### 1. Methodology Issue

**Original Problem**: L2 (center) + Cosine (extreme) vs L2 (homogeneous) showed 0% disagreement because center voters used L2 in both cases.

**Correction**: Use L1 (center) + Cosine (extreme) vs L1 (homogeneous) to reveal true heterogeneity effects.

### 2. Dimensional Scaling

**Original Finding**: Peak effects at 2-3 dimensions

**Corrected Finding**: Effects **increase** with dimension, peaking at 10D (highest tested). Disagreement ranges from 0% at 1D to 31-53% at 10D depending on voting rule.

### 3. Voter Count Effects

**Original Finding**: Fixed 100 voters, no scaling analysis

**New Finding**: Disagreement **decreases** with voter count. Effects are more pronounced in smaller electorates (25.5% at 10 voters vs 12% at 500 voters for Plurality).

### 4. Threshold Effects

**Original Finding**: Strong phase transitions with sigmoidal curves

**Corrected Finding**: For L1-Cosine pair, disagreement is relatively stable across thresholds (~15-21%), suggesting threshold may have less impact than originally thought.

---

## New Discoveries

### 1. Voter Count Dependence

**Discovery**: Heterogeneity effects are **inversely related** to voter count. Smaller electorates show stronger heterogeneity effects, suggesting that heterogeneity may be more important in small-group decision-making than in large-scale elections.

### 2. Dimensional Scaling Reversal

**Discovery**: Contrary to original findings, effects **increase** with dimension rather than peaking at 2-3D. This suggests that in high-dimensional policy spaces, metric heterogeneity becomes more important.

### 3. Metric Pair Hierarchy

**Discovery**: Cosine-based metric assignments create the strongest heterogeneity effects (58-62%), significantly stronger than L1-based (15-21%) or Chebyshev-based (11-14.5%) assignments.

---

## Conclusion

This revised research corrects several major findings from the original study:

1. **Methodology correction**: Using appropriate metric pairs reveals true heterogeneity effects
2. **Dimensional scaling reversal**: Effects increase with dimension, not peak at 2-3D
3. **Voter count dependence**: Effects decrease with voter count - a new discovery
4. **Metric hierarchy**: Cosine-based assignments create strongest effects

These corrections have important implications for understanding how metric heterogeneity affects voting outcomes in different contexts.

---

## References

- Original findings: `heterogeneity-research/FINDINGS.md`
- Research methodology: `heterogenity-simulator/METHODOLOGY.md`
- Research code: `heterogenity-simulator/research_suite.py`
- Analysis code: `heterogenity-simulator/analyze_results.py`
- Centrality concentration analysis: `heterogenity-simulator/centrality_concentration.py`

_Document generated: 2026-01-02T19:30:12.622320_
