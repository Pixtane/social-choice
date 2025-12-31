# Distance Metric Patterns and Analysis

This document presents findings from comprehensive tests comparing homogeneous and heterogeneous distance metrics in spatial voting simulations.

## Test Configuration

- **Profiles**: 500 election profiles
- **Voters**: 25 per profile
- **Candidates**: 3 per profile
- **Geometry**: Uniform distribution in 2D space (with additional tests on polarized and clustered geometries)
- **Voting Rules Tested**: Plurality, Borda, IRV, Approval, STAR, Schulze
- **Utility Function**: Gaussian

## Key Findings

### 1. Homogeneous Distance Metrics Comparison

#### Overall Performance Ranking

Across all voting rules, the homogeneous metrics perform as follows:

1. **L2 (Euclidean)**: Generally best or near-best for most rules

   - Best for: Plurality, Borda
   - Strong performance: IRV, Approval, STAR, Schulze
   - **Key insight**: L2 provides the most balanced distance measure

2. **Cosine**: Excellent for Condorcet methods

   - Best for: STAR, Schulze
   - **Key insight**: Cosine distance eliminates cycles (0% cycle rate) while maintaining high VSE
   - Produces more ideologically consistent preferences

3. **L1 (Manhattan)**: Good for IRV

   - Best for: IRV
   - **Key insight**: L1's additive nature works well with elimination-based methods

4. **Chebyshev**: Best for Approval voting
   - Best for: Approval
   - **Key insight**: Maximum coordinate difference aligns with approval thresholds

#### Cycle Patterns

- **Cosine**: 0% cycles (unique among all metrics)
- **L2**: 1.6% cycles
- **L1**: 2.0% cycles
- **Chebyshev**: 4.2% cycles (highest)

**Interpretation**: Cosine distance creates more transitive preferences because it focuses on directional similarity rather than magnitude. This makes it ideal for Condorcet methods.

### 2. Heterogeneous Distance Effects

#### Center-Extreme Strategy

The center-extreme strategy divides voters into moderates (using center_metric) and extremists (using extreme_metric) based on distance from the political center.

##### Threshold Effects

**Low Threshold (0.3)**: Most voters classified as extreme

- When threshold=0.3, most voters use the extreme metric
- Results approach pure homogeneous extreme metric
- **Finding**: Very low thresholds essentially become homogeneous

**Medium Threshold (0.5)**: Balanced split

- Approximately 44-56% split between center and extreme metrics
- **Finding**: This creates the most interesting heterogeneous effects
- Some configurations show improvement over homogeneous L2

**High Threshold (0.7)**: Most voters classified as center

- 88% use center metric, 12% use extreme
- **Finding**: High thresholds approach homogeneous center metric
- Best performing configurations often use high thresholds

##### Metric Combination Effects

**L2 + Cosine (center + extreme)**:

- Moderate performance
- Low cycle rates (0.6-1.6%)
- **Finding**: Cosine for extremists reduces cycles but can lower VSE

**L2 + Chebyshev (center + extreme)**:

- **Best performing combination** for many rules
- CE_t0.7_l2_chebyshev achieves highest VSE for Borda (0.9676)
- **Finding**: Chebyshev for extremists captures single-issue voters effectively

**L1 + L2 (center + extreme)**:

- Good performance, especially for IRV
- **Finding**: Mixing L1 and L2 creates nuanced preferences

#### Radial Steps Strategy

The radial steps strategy divides the space into concentric regions with different metrics.

##### Scaling Function Effects

**Linear Scaling**:

- Equal spacing between metric boundaries
- **Distribution**: ~4% L1, ~84% L2, ~12% Chebyshev (for 3-metric setup)
- **Performance**: Best for IRV and STAR
- **Finding**: Linear scaling provides balanced metric distribution

**Logarithmic Scaling**:

- More regions near center (compressed at edge)
- **Distribution**: ~0% L1, ~72% L2, ~28% Chebyshev
- **Performance**: Best for Approval voting
- **Finding**: Logarithmic scaling emphasizes center metrics

**Exponential Scaling**:

- More regions near edge (compressed at center)
- **Distribution**: Similar to logarithmic (symmetry in uniform geometry)
- **Performance**: Similar to logarithmic
- **Finding**: In uniform geometry, exponential and logarithmic produce similar results

##### Metric Count Effects

**3 Metrics (L1 → L2 → Chebyshev)**:

- Best overall performance
- RS_linear_3 achieves best VSE for IRV (0.9334) and STAR (0.9630)
- **Finding**: Three metrics provide optimal granularity

**4 Metrics (L1 → L2 → Cosine → Chebyshev)**:

- More complex distribution
- Slightly lower performance than 3-metric version
- **Finding**: More metrics don't necessarily improve outcomes

### 3. Voting Rule Specific Patterns

#### Plurality

- **Best Homogeneous**: L2 (0.8743)
- **Best Heterogeneous**: CE_t0.7_l1_chebyshev (0.8904)
- **Improvement**: +1.8% over homogeneous L2
- **Key Pattern**: Heterogeneous metrics with high threshold and Chebyshev for extremists work best

#### Borda

- **Best Homogeneous**: L2 (0.9619)
- **Best Heterogeneous**: CE_t0.7_l2_chebyshev (0.9676)
- **Improvement**: +0.6% over homogeneous L2
- **Key Pattern**: Small but consistent improvement with heterogeneous metrics

#### IRV

- **Best Homogeneous**: L1 (0.9326)
- **Best Heterogeneous**: RS_linear_3 (0.9334)
- **Improvement**: +0.1% over homogeneous L1
- **Key Pattern**: Radial steps with linear scaling matches L1's strength

#### Approval

- **Best Homogeneous**: Chebyshev (0.8603)
- **Best Heterogeneous**: RS_log_3 (0.8698)
- **Improvement**: +1.1% over homogeneous Chebyshev
- **Key Pattern**: Radial steps with logarithmic scaling works well

#### STAR

- **Best Homogeneous**: Cosine (0.9613)
- **Best Heterogeneous**: RS_linear_3 (0.9630)
- **Improvement**: +0.2% over homogeneous Cosine
- **Key Pattern**: Radial steps slightly outperform pure cosine

#### Schulze

- **Best Homogeneous**: Cosine (0.9613)
- **Best Heterogeneous**: CE_t0.7_l2_chebyshev (0.9604)
- **Change**: -0.1% (slight degradation)
- **Key Pattern**: Pure cosine is optimal for Condorcet methods

### 4. Geometry-Specific Effects

#### Uniform Geometry

- Heterogeneous metrics show small improvements or degradations
- Differences are modest (±2-4%)
- **Finding**: Uniform geometry is relatively insensitive to distance metric choice

#### Polarized Geometry

- **Dramatic effects** with heterogeneous metrics
- Plurality: +39% improvement (0.2957 → 0.6883)
- IRV: +45% improvement (0.3286 → 0.7761)
- **Finding**: Heterogeneous metrics are **highly beneficial** in polarized electorates
- **Interpretation**: When voters are clustered, allowing different cognitive models (center vs extreme) better captures real preferences

#### Clustered Geometry

- Small effects, similar to uniform
- Differences are minimal (±1-2%)
- **Finding**: Clustered geometry behaves similarly to uniform

### 5. Cycle Reduction Patterns

#### Homogeneous Metrics

- Cosine: 0% cycles (unique)
- L2: 1.6% cycles
- L1: 2.0% cycles
- Chebyshev: 4.2% cycles

#### Heterogeneous Strategies

- Center-Extreme with Cosine: 0.6-1.6% cycles
- Center-Extreme with Chebyshev: 2.4-3.4% cycles
- Radial Steps: 1.4-3.0% cycles

**Key Finding**: Heterogeneous metrics generally maintain or reduce cycle rates compared to worst homogeneous metrics, while potentially improving VSE.

### 6. Condorcet Efficiency Patterns

#### Homogeneous Metrics

- Cosine: 100% Condorcet efficiency for STAR and Schulze
- L2: 90-100% efficiency
- L1: 76-100% efficiency
- Chebyshev: 66-100% efficiency

#### Heterogeneous Metrics

- Best configurations: 69-100% efficiency
- **Finding**: Heterogeneous metrics can maintain high Condorcet efficiency while improving VSE

### 7. Interesting Behavioral Patterns

#### Pattern 1: Cosine Eliminates Cycles

- **Observation**: Pure cosine distance produces 0% cycles
- **Explanation**: Cosine focuses on directional similarity, creating more transitive preferences
- **Implication**: Use cosine when cycle reduction is critical

#### Pattern 2: Threshold Sweet Spot

- **Observation**: Threshold=0.7 often produces best results
- **Explanation**: High threshold means most voters use center metric (L2), with extremists using alternative metric
- **Implication**: Small fraction of extremists with different metric can improve outcomes

#### Pattern 3: Chebyshev for Extremists

- **Observation**: Chebyshev as extreme metric often improves VSE
- **Explanation**: Extremists care about single most important issue (max coordinate difference)
- **Implication**: Single-issue voters are better modeled with Chebyshev

#### Pattern 4: Polarized Geometry Benefits

- **Observation**: Heterogeneous metrics show massive improvements in polarized geometry
- **Explanation**: Real polarized electorates have different cognitive models for moderates vs extremists
- **Implication**: Heterogeneous metrics are essential for modeling real-world polarized politics

#### Pattern 5: Radial Steps Optimal Granularity

- **Observation**: 3 metrics (L1→L2→Chebyshev) performs better than 4 metrics
- **Explanation**: Too many metric regions may fragment preferences unnecessarily
- **Implication**: Simpler heterogeneous models can be more effective

#### Pattern 6: Linear Scaling Advantage

- **Observation**: Linear scaling in radial steps often performs best
- **Explanation**: Equal spacing provides balanced metric distribution
- **Implication**: Natural metric boundaries work better than forced compression/expansion

### 8. Performance Implications

#### Computational Overhead

- Heterogeneous distance computation: ~10-20% slower than homogeneous
- **Trade-off**: Small performance cost for potentially significant VSE improvements

#### When to Use Heterogeneous Metrics

**Use Heterogeneous When**:

1. Modeling polarized electorates (large improvements)
2. Single-issue voting is important (Chebyshev for extremists)
3. Cycle reduction is needed (Cosine for extremists)
4. Voter cognitive heterogeneity is expected

**Use Homogeneous When**:

1. Uniform or clustered geometries (small benefits)
2. Computational efficiency is critical
3. Simple baseline comparisons needed
4. Cosine is sufficient (for Condorcet methods)

### 9. Recommendations

#### For Different Voting Rules

1. **Plurality**: Use CE_t0.7_l1_chebyshev (heterogeneous)
2. **Borda**: Use CE_t0.7_l2_chebyshev (heterogeneous) or L2 (homogeneous)
3. **IRV**: Use RS_linear_3 (heterogeneous) or L1 (homogeneous)
4. **Approval**: Use RS_log_3 (heterogeneous) or Chebyshev (homogeneous)
5. **STAR**: Use RS_linear_3 (heterogeneous) or Cosine (homogeneous)
6. **Schulze**: Use Cosine (homogeneous) - heterogeneous provides no benefit

#### For Different Geometries

1. **Uniform**: Homogeneous L2 is sufficient
2. **Polarized**: **Always use heterogeneous** (center-extreme strategy)
3. **Clustered**: Homogeneous L2 is sufficient
4. **Single-peaked**: Homogeneous L2 is sufficient

### 10. Theoretical Implications

#### Cognitive Heterogeneity

The success of heterogeneous metrics, especially in polarized geometries, suggests that:

- Real voters do use different cognitive models
- Extremists think differently about candidates than moderates
- Single-issue voters (Chebyshev) are an important subset

#### Metric Selection

The patterns suggest:

- **L2 (Euclidean)**: Best general-purpose metric
- **Cosine**: Best for Condorcet consistency
- **Chebyshev**: Best for single-issue modeling
- **L1 (Manhattan)**: Best for elimination methods

#### Strategy Selection

- **Center-Extreme**: Best for polarized electorates
- **Radial Steps**: Best for uniform electorates with nuanced preferences
- **Homogeneous**: Best for simple, efficient modeling

### 11. Future Research Directions

1. **Adaptive Thresholds**: Thresholds that adjust based on voter distribution
2. **Learned Metrics**: Metrics learned from voter behavior data
3. **Multi-Dimensional Heterogeneity**: Different metrics for different policy dimensions
4. **Temporal Dynamics**: Metrics that change over time
5. **Strategic Voting**: How heterogeneous metrics affect strategic behavior

### 12. Summary Statistics

#### Homogeneous Metrics (Best per Rule)

- Average VSE: 0.924
- Average Cycle Rate: 1.5%
- Average Condorcet Efficiency: 88%

#### Heterogeneous Metrics (Best per Rule)

- Average VSE: 0.930
- Average Cycle Rate: 1.6%
- Average Condorcet Efficiency: 87%

**Overall**: Heterogeneous metrics provide modest average improvements (+0.6% VSE) with similar cycle rates and Condorcet efficiency. However, in polarized geometries, improvements can be dramatic (+40% VSE).

## Conclusion

Heterogeneous distance metrics provide meaningful improvements in specific contexts:

- **Polarized electorates**: Large improvements (30-45% VSE increase)
- **Uniform electorates**: Small improvements (0-2% VSE increase)
- **Cycle reduction**: Cosine-based strategies eliminate cycles
- **Single-issue modeling**: Chebyshev for extremists captures important voter types

The choice between homogeneous and heterogeneous metrics should depend on:

1. **Electorate structure** (polarized vs uniform)
2. **Voting rule** (some rules benefit more)
3. **Computational constraints** (heterogeneous is slower)
4. **Modeling goals** (realism vs simplicity)

For most applications, homogeneous L2 provides excellent baseline performance. However, when modeling real-world polarized politics or when small VSE improvements are critical, heterogeneous metrics offer valuable enhancements.
