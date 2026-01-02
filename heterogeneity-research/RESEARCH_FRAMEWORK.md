# Research Framework: Heterogeneity in Distance Functions

## Core Research Questions

### 1. **When Does Heterogeneity Matter?**
- **Question**: Under what conditions does heterogeneous distance significantly affect voting outcomes?
- **Hypotheses**:
  - Heterogeneity matters more when voter distribution is polarized
  - Effect increases with number of candidates
  - Effect depends on the specific metric combinations
- **Tests**: Compare homogeneous vs heterogeneous across different geometries and candidate counts

### 2. **Which Metric Combinations Are Most Impactful?**
- **Question**: Which pairs/combinations of distance metrics create the largest differences in outcomes?
- **Hypotheses**:
  - L2 + Cosine (center-extreme) creates ideological vs pragmatic split
  - L1 + Chebyshev creates nuanced vs single-issue split
  - Radial steps with many metrics creates complex preference structures
- **Tests**: Systematic sweep of all metric pairs in center-extreme strategy

### 3. **How Does Threshold Affect Outcomes?**
- **Question**: How does the center-extreme threshold parameter affect voting results?
- **Hypotheses**:
  - Low thresholds (most voters extreme) → outcomes dominated by extreme metric
  - Medium thresholds (balanced) → maximum heterogeneity effects
  - High thresholds (most voters center) → outcomes dominated by center metric
- **Tests**: Sweep threshold from 0.1 to 0.9 in 0.1 increments

### 4. **Which Voting Rules Are Most Sensitive?**
- **Question**: Which voting rules are most affected by distance heterogeneity?
- **Hypotheses**:
  - Plurality: High sensitivity (first-choice effects)
  - Condorcet methods: Moderate sensitivity (pairwise comparisons)
  - Borda: Low sensitivity (averaging effect)
- **Tests**: Compare rule disagreement rates across homogeneous/heterogeneous

### 5. **Spatial Patterns of Metric Assignment**
- **Question**: How does the spatial distribution of metrics affect outcomes?
- **Hypotheses**:
  - Clustered metric regions create local preference structures
  - Random metric assignment creates noise
  - Radial patterns create systematic biases
- **Tests**: Compare different strategies (center-extreme, radial-steps, random)

### 6. **Dimensionality Effects**
- **Question**: How does heterogeneity effect change with spatial dimensions?
- **Hypotheses**:
  - Higher dimensions → larger heterogeneity effects (more room for differences)
  - Cosine distance becomes more important in high dimensions
  - L1/L2 differences matter less in high dimensions
- **Tests**: Sweep dimensions 1-10 with fixed heterogeneity configuration

## Key Metrics to Track

### Outcome Metrics
1. **Rule Disagreement**: % of profiles where heterogeneous ≠ homogeneous winner
2. **VSE Difference**: Change in Voter Satisfaction Efficiency
3. **Winner Extremism Change**: How winner position changes
4. **Condorcet Consistency**: Whether Condorcet winner selection changes

### Distribution Metrics
1. **Metric Distribution**: Fraction of voters using each metric
2. **Spatial Clustering**: How metrics cluster in space
3. **Winner Support by Metric**: Which metric groups support the winner

### Preference Structure Metrics
1. **Cycle Rate**: How often Condorcet cycles occur
2. **Preference Transitivity**: Measure of preference consistency
3. **Utility Variance**: Spread of voter utilities

## Research Design

### Phase 1: Baseline Characterization
**Goal**: Understand homogeneous distance metric behavior

1. Run all 4 metrics (L1, L2, Cosine, Chebyshev) separately
2. Compare across all voting rules
3. Identify baseline patterns and differences

**Output**: Baseline performance matrix (metric × rule)

### Phase 2: Systematic Heterogeneity Exploration
**Goal**: Map the heterogeneity parameter space

1. **Center-Extreme Sweep**:
   - All metric pairs (6 combinations)
   - Threshold sweep: 0.1, 0.2, ..., 0.9
   - Fixed: 2D, 5 candidates, 100 voters, 100 profiles

2. **Radial Steps Sweep**:
   - Different metric sequences (e.g., [L1, L2, Chebyshev])
   - Scaling functions (linear, logarithmic, exponential)
   - Boundary parameter variations

**Output**: Parameter sensitivity maps

### Phase 3: Context-Dependent Effects
**Goal**: Understand when heterogeneity matters most

1. **Geometry Variations**:
   - Uniform, polarized, clustered, single-peaked
   - Same heterogeneity config across all

2. **Candidate Count Sweep**:
   - 2, 3, 5, 10 candidates
   - Fixed heterogeneity config

3. **Dimensionality Sweep**:
   - 1, 2, 3, 5, 10 dimensions
   - Fixed heterogeneity config

**Output**: Context-dependent effect sizes

### Phase 4: Voting Rule Sensitivity
**Goal**: Identify which rules are most affected

1. Run all rules with same heterogeneity config
2. Compare disagreement rates
3. Analyze which rules show largest VSE changes

**Output**: Rule sensitivity ranking

## Visualization Strategy

### 1. **Spatial Visualizations** (2D only)
- Voters colored by assigned metric
- Candidates with winner highlighted
- Voting lines showing preferences
- Metric region boundaries (for radial steps)

### 2. **Parameter Sweep Visualizations**
- Heatmaps: threshold × metric pair → outcome metric
- Line plots: threshold → outcome (for each metric pair)
- 3D surfaces: threshold × candidate_count → outcome

### 3. **Comparison Visualizations**
- Side-by-side: homogeneous vs heterogeneous outcomes
- Difference plots: heterogeneous - homogeneous
- Distribution overlays: winner position distributions

### 4. **Statistical Visualizations**
- Box plots: outcome distributions across profiles
- Violin plots: full distribution shapes
- Scatter plots: correlation between metrics

## Interpretation Guidelines

### What to Look For

1. **Large Rule Disagreement (>50%)**
   - Indicates heterogeneity fundamentally changes outcomes
   - Check if disagreement is systematic or random

2. **VSE Improvements**
   - Heterogeneous > homogeneous suggests better representation
   - Check if improvement is consistent or rare

3. **Spatial Patterns**
   - Do metric regions align with voter clusters?
   - Are winners supported by specific metric groups?

4. **Threshold Effects**
   - Is there a "sweet spot" threshold?
   - Do effects plateau or change continuously?

5. **Rule-Specific Patterns**
   - Which rules benefit from heterogeneity?
   - Which rules are destabilized?

### Red Flags

- **Inconsistent Results**: High variance suggests unstable effects
- **No Effect**: If heterogeneity ≈ homogeneous, may not be worth complexity
- **Extreme Outliers**: Check for numerical issues or edge cases

## Expected Findings (Hypotheses)

1. **Heterogeneity matters most when**:
   - Voters are polarized
   - Many candidates (5+)
   - Using center-extreme with medium threshold (0.4-0.6)

2. **Best metric combinations**:
   - L2 + Cosine: Ideological vs pragmatic
   - L2 + Chebyshev: Balanced vs single-issue
   - L1 + L2: Nuanced vs simple

3. **Most sensitive rules**:
   - Plurality (highest)
   - IRV (high)
   - Borda (moderate)
   - Condorcet methods (lowest)

4. **Dimensionality effects**:
   - Effects increase up to ~5 dimensions
   - Plateau or decrease beyond 5 dimensions

## Next Steps After Initial Findings

1. **Refine Hypotheses**: Based on initial results
2. **Targeted Experiments**: Focus on promising parameter regions
3. **Theoretical Analysis**: Explain observed patterns
4. **Real-World Validation**: Compare to empirical voting data





