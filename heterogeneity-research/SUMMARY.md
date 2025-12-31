# Heterogeneity Research - Summary

## What You Now Have

A comprehensive research framework for studying heterogeneity in distance functions for spatial voting, including:

### 1. Research Framework (`RESEARCH_FRAMEWORK.md`)
- **6 core research questions** with hypotheses
- **4-phase research design** (baseline → exploration → context → sensitivity)
- **Interpretation guidelines** for what to look for
- **Expected findings** based on theory

### 2. Visualization Tools (`visualization_tools.py`)
- **Spatial metric distribution** plots (2D voters colored by metric)
- **Threshold sweep** analysis plots
- **Metric pair heatmaps** for comparing combinations
- **Comparison distributions** (homogeneous vs heterogeneous)
- **Rule sensitivity** visualizations
- **Dimensionality effects** plots

### 3. Systematic Test Suite (`systematic_tests.py`)
- **Baseline characterization**: Test all homogeneous metrics
- **Threshold sweep**: Test center-extreme threshold parameter
- **All metric pairs**: Systematic testing of combinations
- **Geometry effects**: Test across different voter distributions
- **Dimensionality effects**: Test across spatial dimensions
- **Full suite**: Run everything systematically

### 4. Analysis Tools (`analysis_tools.py`)
- **Baseline analysis**: Rank metrics by performance
- **Threshold analysis**: Find optimal thresholds
- **Comparison tools**: Compare homogeneous vs heterogeneous
- **Summary reports**: Generate comprehensive reports
- **Insight generation**: Identify key research insights

### 5. Documentation
- **README.md**: Full documentation
- **QUICKSTART.md**: 5-minute getting started guide
- **RESEARCH_FRAMEWORK.md**: Research questions and design

## How to Use

### Quick Start (5 minutes)
```bash
# 1. Run baseline test
python heterogeneity-research/systematic_tests.py --phase baseline --quick

# 2. Analyze results
python -c "from heterogeneity-research.analysis_tools import HeterogeneityAnalyzer; a = HeterogeneityAnalyzer(); a.generate_summary_report()"
```

### Full Research Workflow

1. **Phase 1: Baseline** - Understand homogeneous metrics
   ```bash
   python heterogeneity-research/systematic_tests.py --phase baseline
   ```

2. **Phase 2: Threshold Sweep** - Find optimal parameters
   ```bash
   python heterogeneity-research/systematic_tests.py --phase threshold
   ```

3. **Phase 3: Context Effects** - Test when heterogeneity matters
   ```bash
   python heterogeneity-research/systematic_tests.py --phase geometry
   python heterogeneity-research/systematic_tests.py --phase dimensions
   ```

4. **Visualize Results**
   ```python
   from heterogeneity-research.visualization_tools import HeterogeneityVisualizer
   # ... create visualizations
   ```

5. **Analyze and Report**
   ```python
   from heterogeneity-research.analysis_tools import HeterogeneityAnalyzer
   analyzer = HeterogeneityAnalyzer()
   report = analyzer.generate_summary_report("my_report.md")
   ```

## Key Research Questions Answered

### 1. When does heterogeneity matter?
- **Test**: Compare across geometries, candidate counts, dimensions
- **Look for**: High disagreement rates (>50%)

### 2. Which metric combinations are most impactful?
- **Test**: All metric pairs with threshold sweep
- **Look for**: Large VSE differences or disagreement rates

### 3. How does threshold affect outcomes?
- **Test**: Sweep threshold 0.1 to 0.9
- **Look for**: Optimal threshold (usually 0.4-0.6)

### 4. Which voting rules are most sensitive?
- **Test**: All rules with same heterogeneity config
- **Look for**: Rule-specific disagreement rates

### 5. What are the spatial patterns?
- **Test**: Visualize 2D distributions
- **Look for**: Metric clustering, winner support patterns

## What to Look For

### Strong Signals
- ✅ **Disagreement > 50%**: Heterogeneity fundamentally changes outcomes
- ✅ **VSE improvements**: Heterogeneity helps representation
- ✅ **Spatial clustering**: Metrics align with voter clusters
- ✅ **Threshold sweet spots**: Clear optimal parameter values

### Weak Signals
- ⚠️ **Low disagreement (<20%)**: Heterogeneity may not matter much
- ⚠️ **Inconsistent results**: High variance suggests instability
- ⚠️ **No spatial patterns**: Random metric assignment may not help

## Expected Findings (Hypotheses)

Based on theory and initial experiments:

1. **Heterogeneity matters most when**:
   - Voters are polarized (not uniform)
   - Many candidates (5+)
   - Medium threshold (0.4-0.6)

2. **Best metric combinations**:
   - L2 + Cosine: Ideological vs pragmatic split
   - L2 + Chebyshev: Balanced vs single-issue
   - L1 + L2: Nuanced vs simple

3. **Most sensitive rules**:
   - Plurality (highest sensitivity)
   - IRV (high)
   - Borda (moderate)
   - Condorcet methods (lowest)

4. **Dimensionality effects**:
   - Effects increase up to ~5 dimensions
   - Plateau or decrease beyond 5D

## Next Steps

1. **Run initial tests** to validate hypotheses
2. **Focus on high-disagreement regions** for detailed analysis
3. **Create visualizations** to understand patterns
4. **Generate reports** to document findings
5. **Formulate new hypotheses** based on results

## Tips

- Start with `--quick` mode to explore parameter space
- Use visualizations to understand spatial patterns
- Compare multiple rules to see rule-specific effects
- Document findings as you go
- Focus on statistically significant differences

## File Locations

- **Results**: `heterogeneity-research/results/`
- **Figures**: `heterogeneity-research/figures/` (create if needed)
- **Reports**: Save anywhere (e.g., `heterogeneity-research/summary_report.md`)

## Integration with Existing Tools

These tools integrate with your existing simulator:
- Uses `simulator.config` for configurations
- Uses `simulator.main.run_experiment` for simulations
- Uses `simulator.metrics` for metric computation
- Compatible with existing GUI and CLI tools

## Support

- See `QUICKSTART.md` for getting started
- See `RESEARCH_FRAMEWORK.md` for research questions
- See `README.md` for full documentation
- Check existing results in `heterogenity-testing/` for examples

