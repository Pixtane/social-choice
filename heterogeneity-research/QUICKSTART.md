# Quick Start Guide: Heterogeneity Research

Get started with heterogeneity research in 5 minutes!

## Step 1: Run Your First Test

```bash
# Quick test (fast, ~5 minutes)
python heterogeneity-research/systematic_tests.py --phase baseline --quick

# Full test (slower, ~30 minutes)
python heterogeneity-research/systematic_tests.py --phase baseline
```

This creates `heterogeneity-research/results/baseline_characterization.json`

## Step 2: Visualize Results

```python
# Quick visualization script
import json
import numpy as np
import matplotlib.pyplot as plt
from heterogeneity_research.visualization_tools import HeterogeneityVisualizer

# Load results
with open('heterogeneity-research/results/baseline_characterization.json') as f:
    baseline = json.load(f)

# Create comparison
viz = HeterogeneityVisualizer()

metrics = ['l1', 'l2', 'cosine', 'chebyshev']
rules = ['plurality', 'borda', 'irv']

# Extract VSE values
vse_data = {}
for rule in rules:
    vse_data[rule] = [baseline[m][rule]['vse_mean'] for m in metrics]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.25

for i, rule in enumerate(rules):
    offset = (i - 1) * width
    ax.bar(x + offset, vse_data[rule], width, label=rule)

ax.set_xlabel('Distance Metric')
ax.set_ylabel('VSE')
ax.set_title('Baseline: VSE by Metric and Rule')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heterogeneity-research/figures/baseline_comparison.png', dpi=300)
print("Saved to heterogeneity-research/figures/baseline_comparison.png")
```

## Step 3: Test Heterogeneity

```bash
# Test threshold sweep (L2 + Cosine)
python heterogeneity-research/systematic_tests.py --phase threshold
```

This tests how the center-extreme threshold affects outcomes.

## Step 4: Compare Homogeneous vs Heterogeneous

```python
from simulator import SimulationConfig, GeometryConfig, UtilityConfig, HeterogeneousDistanceConfig
from simulator.main import run_experiment

# Homogeneous (L2)
config_homo = SimulationConfig(
    n_profiles=50,
    n_voters=100,
    n_candidates=5,
    voting_rules=['plurality', 'borda'],
    geometry=GeometryConfig(method='uniform', n_dim=2),
    utility=UtilityConfig(
        function='linear',
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
    )
)

# Heterogeneous (L2 center + Cosine extreme)
config_het = SimulationConfig(
    n_profiles=50,
    n_voters=100,
    n_candidates=5,
    voting_rules=['plurality', 'borda'],
    geometry=GeometryConfig(method='uniform', n_dim=2),
    utility=UtilityConfig(
        function='linear',
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme',
            center_metric='l2',
            extreme_metric='cosine',
            extreme_threshold=0.5
        )
    )
)

# Run both
result_homo = run_experiment(config_homo, save_results=False, verbose=True)
result_het = run_experiment(config_het, save_results=False, verbose=True)

# Compare
for rule in ['plurality', 'borda']:
    homo_vse = result_homo.rule_results[rule].aggregate_metrics.vse_mean
    het_vse = result_het.rule_results[rule].aggregate_metrics.vse_mean
    print(f"{rule}: Homo={homo_vse:.4f}, Het={het_vse:.4f}, Δ={het_vse-homo_vse:+.4f}")
```

## Step 5: Visualize Spatial Distribution

```python
from simulator.geometry import GeometryGenerator
from simulator.utility import UtilityComputer
from simulator.config import GeometryConfig, UtilityConfig, HeterogeneousDistanceConfig
import numpy as np

# Generate a single profile
rng = np.random.default_rng(42)
geometry = GeometryGenerator(GeometryConfig(method='uniform', n_dim=2), rng)
spatial = geometry.generate(1, 100, 5)

# Create heterogeneous utility computer
het_config = HeterogeneousDistanceConfig(
    enabled=True,
    strategy='center_extreme',
    center_metric='l2',
    extreme_metric='cosine',
    extreme_threshold=0.5
)
utility_config = UtilityConfig(
    function='linear',
    distance_metric='l2',
    heterogeneous_distance=het_config
)
utility_computer = UtilityComputer(utility_config)

# Get voter metrics
voter_pos = spatial.voter_positions[0]
candidate_pos = spatial.candidate_positions[0]
voter_metrics = utility_computer.get_voter_metrics(voter_pos)

# Visualize
from heterogeneity_research.visualization_tools import HeterogeneityVisualizer

viz = HeterogeneityVisualizer()
viz.plot_spatial_metric_distribution(
    voter_positions=voter_pos,
    candidate_positions=candidate_pos,
    voter_metrics=voter_metrics,
    title="Example: Spatial Distribution of Metrics",
    save_path="heterogeneity-research/figures/spatial_example.png"
)
```

## Step 6: Analyze Your Results

```python
from heterogeneity_research.analysis_tools import HeterogeneityAnalyzer

analyzer = HeterogeneityAnalyzer()

# Generate summary report
report = analyzer.generate_summary_report("heterogeneity-research/my_report.md")
print(report)
```

## Common Questions

### Q: Which test should I run first?

**A**: Start with `--phase baseline` to understand homogeneous metrics, then `--phase threshold` to see heterogeneity effects.

### Q: How long do tests take?

**A**:

- Quick mode: 5-15 minutes per phase
- Full mode: 30-60 minutes per phase
- Full suite: 2-4 hours

### Q: What should I look for?

**A**:

1. **High disagreement rates** (>50%) = heterogeneity matters
2. **VSE improvements** = heterogeneity helps
3. **Spatial patterns** = understand why effects occur

### Q: How do I interpret threshold sweep results?

**A**:

- Low threshold (0.1-0.3): Most voters extreme → dominated by extreme metric
- Medium threshold (0.4-0.6): Balanced → maximum heterogeneity effects
- High threshold (0.7-0.9): Most voters center → dominated by center metric

### Q: What if I see no effect?

**A**:

- Try different geometries (polarized, clustered)
- Try different metric pairs
- Check if you're using enough profiles (need 50+ for stable results)

## Next Steps

1. Read `RESEARCH_FRAMEWORK.md` for research questions
2. Run systematic tests: `python heterogeneity-research/systematic_tests.py --phase all`
3. Create visualizations for your results
4. Generate analysis reports
5. Formulate new hypotheses based on findings

## Getting Help

- Check `RESEARCH_FRAMEWORK.md` for research questions
- See `README.md` for full documentation
- Look at existing results in `heterogenity-testing/` for examples

