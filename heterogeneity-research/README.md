# Heterogeneity in Distance Functions - Research Tools

This directory contains comprehensive tools for researching heterogeneity in distance functions for spatial voting simulations.

## Overview

This research framework helps you:

1. **Understand** when and how distance metric heterogeneity affects voting outcomes
2. **Systematically test** different heterogeneity configurations
3. **Visualize** spatial patterns and parameter effects
4. **Analyze** results and generate insights

## Directory Structure

```
heterogeneity-research/
├── README.md                    # This file
├── RESEARCH_FRAMEWORK.md        # Research questions and hypotheses
├── visualization_tools.py       # Enhanced visualization functions
├── systematic_tests.py          # Automated test suite
├── analysis_tools.py            # Result analysis and interpretation
├── results/                     # Test results (generated)
└── figures/                     # Generated visualizations (generated)
```

## Quick Start

### 1. Run Baseline Characterization

Understand how homogeneous distance metrics perform:

```bash
python heterogeneity-research/systematic_tests.py --phase baseline
```

This tests all 4 metrics (L1, L2, Cosine, Chebyshev) separately and creates a baseline for comparison.

### 2. Run Threshold Sweep

Test how the center-extreme threshold parameter affects outcomes:

```bash
python heterogeneity-research/systematic_tests.py --phase threshold
```

This sweeps threshold from 0.1 to 0.9 and compares heterogeneous vs homogeneous.

### 3. Run Full Test Suite

Run all tests systematically:

```bash
python heterogeneity-research/systematic_tests.py --phase all
```

For faster testing (reduced parameters):

```bash
python heterogeneity-research/systematic_tests.py --phase all --quick
```

### 4. Analyze Results

Generate summary reports and insights:

```python
from heterogeneity_research.analysis_tools import HeterogeneityAnalyzer

analyzer = HeterogeneityAnalyzer()
report = analyzer.generate_summary_report("heterogeneity-research/summary_report.md")
```

### 5. Create Visualizations

Visualize your results:

```python
from heterogeneity_research.visualization_tools import HeterogeneityVisualizer
import numpy as np

viz = HeterogeneityVisualizer()

# Load results
import json
with open('heterogeneity-research/results/threshold_sweep_l2_cosine.json') as f:
    results = json.load(f)

# Plot threshold sweep
thresholds = np.array(results['thresholds'])
rule_data = results['results']['0.5']['plurality']  # Example

viz.plot_threshold_sweep(
    thresholds=thresholds,
    results={'rule_disagreement': [results['results'][f'{t:.1f}']['plurality']['disagreement_rate']
                                   for t in thresholds]},
    metric_pair="L2 + Cosine",
    save_path="heterogeneity-research/figures/threshold_sweep.png"
)
```

## Research Framework

See `RESEARCH_FRAMEWORK.md` for:

- Core research questions
- Hypotheses to test
- Key metrics to track
- Research design phases
- Interpretation guidelines

## Key Research Questions

1. **When does heterogeneity matter?**

   - Test across different geometries, candidate counts, dimensions

2. **Which metric combinations are most impactful?**

   - Systematic testing of all metric pairs

3. **How does threshold affect outcomes?**

   - Sweep threshold parameter to find optimal values

4. **Which voting rules are most sensitive?**

   - Compare rule disagreement rates

5. **What are the spatial patterns?**
   - Visualize metric distribution in 2D space

## Test Phases

### Phase 1: Baseline Characterization

- Test all homogeneous metrics separately
- Establish baseline performance
- Identify best metrics for each rule

### Phase 2: Systematic Exploration

- Threshold sweep for center-extreme strategy
- Test all metric pairs
- Radial steps parameter exploration

### Phase 3: Context-Dependent Effects

- Geometry variations (uniform, polarized, clustered)
- Candidate count sweep
- Dimensionality sweep

### Phase 4: Voting Rule Sensitivity

- Compare all rules with same heterogeneity config
- Identify most/least sensitive rules

## Visualization Types

1. **Spatial Visualizations** (2D)

   - Voters colored by metric
   - Candidates with winners
   - Metric region boundaries

2. **Parameter Sweep Visualizations**

   - Heatmaps: threshold × metric pair → outcome
   - Line plots: threshold → outcome
   - 3D surfaces for multi-parameter exploration

3. **Comparison Visualizations**

   - Side-by-side homogeneous vs heterogeneous
   - Difference plots
   - Distribution overlays

4. **Statistical Visualizations**
   - Box plots, violin plots
   - Scatter plots for correlations

## Interpretation Guidelines

### What to Look For

1. **High Rule Disagreement (>50%)**

   - Indicates heterogeneity fundamentally changes outcomes
   - Check if systematic or random

2. **VSE Improvements**

   - Heterogeneous > homogeneous suggests better representation
   - Check consistency

3. **Spatial Patterns**

   - Do metric regions align with voter clusters?
   - Are winners supported by specific metric groups?

4. **Threshold Effects**
   - Is there a "sweet spot" threshold?
   - Do effects plateau or change continuously?

### Red Flags

- **Inconsistent Results**: High variance suggests unstable effects
- **No Effect**: If heterogeneity ≈ homogeneous, may not be worth complexity
- **Extreme Outliers**: Check for numerical issues

## Expected Findings

Based on initial hypotheses:

1. **Heterogeneity matters most when**:

   - Voters are polarized
   - Many candidates (5+)
   - Medium threshold (0.4-0.6)

2. **Best metric combinations**:

   - L2 + Cosine: Ideological vs pragmatic
   - L2 + Chebyshev: Balanced vs single-issue
   - L1 + L2: Nuanced vs simple

3. **Most sensitive rules**:
   - Plurality (highest)
   - IRV (high)
   - Borda (moderate)
   - Condorcet methods (lowest)

## Example Workflow

```python
# 1. Run tests
from heterogeneity_research.systematic_tests import HeterogeneityTestSuite

suite = HeterogeneityTestSuite()
results = suite.run_full_suite(quick_mode=False)

# 2. Analyze
from heterogeneity_research.analysis_tools import HeterogeneityAnalyzer

analyzer = HeterogeneityAnalyzer()
insights = analyzer.identify_research_insights(results)

# 3. Visualize
from heterogeneity_research.visualization_tools import HeterogeneityVisualizer

viz = HeterogeneityVisualizer()
# ... create visualizations

# 4. Generate report
report = analyzer.generate_summary_report("report.md")
```

## Tips for Research

1. **Start with quick mode** to explore parameter space
2. **Focus on high-disagreement regions** for detailed analysis
3. **Use visualizations** to understand spatial patterns
4. **Compare multiple rules** to see rule-specific effects
5. **Document findings** as you go

## Next Steps

After initial findings:

1. **Refine hypotheses** based on results
2. **Targeted experiments** in promising regions
3. **Theoretical analysis** to explain patterns
4. **Real-world validation** with empirical data

## Contributing

When adding new tests or visualizations:

1. Follow the research framework structure
2. Document hypotheses and expected findings
3. Include interpretation guidelines
4. Add to the test suite systematically

## References

- See `DISTANCE_PATTERNS.md` for previous findings
- See `HETEROGENITY_ANALYSIS.md` for analysis of past experiments
- See `HETEROGENEOUS_DISTANCE.md` for implementation details




