# Heterogeneous Distance Metrics

A new feature that allows different voters to use different distance metrics based on their position in space.

## Overview

In the standard spatial voting model, all voters use the same distance metric (e.g., Euclidean/L2) to compute their preference for candidates. However, real voters may perceive political "distance" differently:

- **Moderates** might care about overall similarity (Euclidean distance)
- **Extremists** might care more about ideological direction (Cosine distance)
- **Single-issue voters** might care about specific policy dimensions (Manhattan/Chebyshev)

This feature enables **heterogeneous cognitive models** where different voters use different distance functions.

## Strategies

### 1. Center-Extreme Strategy

Voters are classified based on their distance from the center of the political space:

- **Center voters**: Use one metric (default: L2/Euclidean)
- **Extreme voters**: Use another metric (default: Cosine)

The classification is controlled by a threshold parameter.

**Configuration:**
```python
HeterogeneousDistanceConfig(
    enabled=True,
    strategy='center_extreme',
    center_metric='l2',      # Metric for moderates
    extreme_metric='cosine', # Metric for extremists
    extreme_threshold=0.5    # 0.0=all extreme, 1.0=all center
)
```

**Interpretation:**
- `extreme_threshold=0.5`: Voters in the inner 50% (by distance from center) use the center metric; outer 50% use the extreme metric
- Useful for modeling political polarization where extremists think differently about candidates

### 2. Radial Steps Strategy

The political space is divided into concentric regions, with each region using a different distance metric:

- **Inner region**: L1 (Manhattan) - care about specific policy differences
- **Middle region**: L2 (Euclidean) - balanced view
- **Outer region**: Chebyshev (L∞) - only care about the biggest difference

The boundaries between regions are controlled by a scaling function.

**Configuration:**
```python
HeterogeneousDistanceConfig(
    enabled=True,
    strategy='radial_steps',
    radial_metrics=['l1', 'l2', 'chebyshev'],  # Center to edge
    radial_scaling='linear',     # How boundaries are spaced
    scaling_parameter=2.0        # Controls non-linear scaling
)
```

**Scaling Options:**
- `linear`: Equal spacing between boundaries
- `logarithmic`: More regions near center (compressed at edge)
- `exponential`: More regions near edge (compressed at center)

## Available Distance Metrics

| Metric | Name | Formula | Description |
|--------|------|---------|-------------|
| `l2` | Euclidean | √Σ(xᵢ - yᵢ)² | Straight-line distance |
| `l1` | Manhattan | Σ\|xᵢ - yᵢ\| | City-block distance |
| `cosine` | Cosine | 1 - cos(θ) | Directional similarity |
| `chebyshev` | Chebyshev | max\|xᵢ - yᵢ\| | Maximum coordinate difference |

## Usage

### In Python

```python
from simulator import (
    SimulationConfig, UtilityConfig, 
    HeterogeneousDistanceConfig
)

# Create heterogeneous distance config
het_config = HeterogeneousDistanceConfig(
    enabled=True,
    strategy='center_extreme',
    center_metric='l2',
    extreme_metric='cosine',
    extreme_threshold=0.5
)

# Create simulation config
config = SimulationConfig(
    n_profiles=1000,
    n_voters=50,
    n_candidates=3,
    voting_rules=['plurality', 'borda', 'irv'],
    utility=UtilityConfig(
        function='gaussian',
        distance_metric='l2',  # Fallback (not used when het is enabled)
        heterogeneous_distance=het_config
    )
)

# Run simulation
from simulator import run_experiment
result = run_experiment(config, save_results=True, verbose=True)
```

### In GUI

1. Launch the GUI: `python run_gui.py`
2. Select "New Simulation"
3. Scroll down in the right panel to "Heterogeneous Distance"
4. Check "Enable heterogeneous distance metrics"
5. Configure:
   - **Strategy**: center_extreme or radial_steps
   - **Center/Extreme Settings**: For center_extreme strategy
   - **Radial Steps Settings**: For radial_steps strategy
6. Run simulation

## Analysis

You can analyze the distribution of metrics across voters:

```python
from simulator.utility import UtilityComputer
from simulator.config import UtilityConfig, HeterogeneousDistanceConfig

config = UtilityConfig(
    heterogeneous_distance=HeterogeneousDistanceConfig(
        enabled=True,
        strategy='center_extreme',
        extreme_threshold=0.4
    )
)

computer = UtilityComputer(config)

# Get per-voter metrics
voter_metrics = computer.get_voter_metrics(voter_positions)
print(voter_metrics)  # ['l2', 'l2', 'cosine', 'cosine', ...]

# Get distribution
distribution = computer.get_metric_distribution(voter_positions)
print(distribution)  # {'l2': 0.4, 'cosine': 0.6}
```

## Examples

### Example 1: Polarized Electorate with Cognitive Heterogeneity

```python
from simulator import *

config = SimulationConfig(
    n_profiles=2000,
    n_voters=51,
    n_candidates=3,
    voting_rules=['plurality', 'approval', 'star', 'schulze'],
    geometry=GeometryConfig(method='polarized', n_dim=2),
    utility=UtilityConfig(
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme',
            center_metric='l2',
            extreme_metric='cosine',
            extreme_threshold=0.3  # More voters using cosine
        )
    )
)

result = run_experiment(config, verbose=True)
```

### Example 2: Radial Complexity Model

```python
config = SimulationConfig(
    n_profiles=1000,
    n_voters=100,
    n_candidates=4,
    voting_rules=['borda', 'star', 'schulze'],
    geometry=GeometryConfig(method='uniform', n_dim=2),
    utility=UtilityConfig(
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='radial_steps',
            radial_metrics=['l1', 'l2', 'cosine', 'chebyshev'],
            radial_scaling='logarithmic',
            scaling_parameter=3.0
        )
    )
)
```

### Example 3: Comparing Homogeneous vs Heterogeneous

```python
# Run homogeneous
config_homo = SimulationConfig(
    n_profiles=1000,
    n_voters=50,
    n_candidates=3,
    voting_rules=['plurality', 'borda'],
    utility=UtilityConfig(
        distance_metric='l2',
        heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
    )
)

# Run heterogeneous
config_het = SimulationConfig(
    n_profiles=1000,
    n_voters=50,
    n_candidates=3,
    voting_rules=['plurality', 'borda'],
    utility=UtilityConfig(
        heterogeneous_distance=HeterogeneousDistanceConfig(
            enabled=True,
            strategy='center_extreme'
        )
    )
)

result_homo = run_experiment(config_homo)
result_het = run_experiment(config_het)

# Compare VSE
for rule in ['plurality', 'borda']:
    vse_homo = result_homo.rule_results[rule].aggregate_metrics.vse_mean
    vse_het = result_het.rule_results[rule].aggregate_metrics.vse_mean
    print(f"{rule}: Homo={vse_homo:.3f}, Het={vse_het:.3f}")
```

## Theory

### Why Heterogeneous Distance?

Standard spatial voting models assume all voters use the same distance function. This is a strong assumption that may not hold in practice:

1. **Cognitive heterogeneity**: People process information differently
2. **Issue salience**: Different voters care about different issues
3. **Ideological reasoning**: Moderates vs extremists may think differently
4. **Information availability**: Some voters may have more nuanced views

### Center-Extreme Model

Based on research suggesting that political extremists process information differently:

- **Moderates**: Consider overall platform similarity (L2)
- **Extremists**: Focus on ideological direction/purity (Cosine)

The cosine distance cares about the *angle* between positions, not the magnitude. An extremist at position (0.9, 0.9) would see a candidate at (0.8, 0.8) as very similar (same direction), even though they're spatially separated.

### Radial Steps Model

Models increasing cognitive complexity as voters move from center:

- **Center voters (L1)**: Simple, additive reasoning about policies
- **Middle voters (L2)**: Standard Euclidean reasoning
- **Edge voters (Chebyshev)**: Focus on the single most important issue

This creates a natural "gradient" of cognitive models across the electorate.

## Implementation Details

### File Structure

```
simulator/
├── config.py                    # HeterogeneousDistanceConfig
├── heterogeneous_distance.py    # Strategy classes
├── utility.py                   # Integration with UtilityComputer
└── gui.py                       # GUI controls
```

### Key Classes

- `HeterogeneousDistanceConfig`: Configuration dataclass
- `HeterogeneousDistanceStrategy`: Abstract base class
- `CenterExtremeStrategy`: Implementation of center-extreme
- `RadialStepsStrategy`: Implementation of radial steps
- `UtilityComputer`: Integrates heterogeneous distance

### Performance

Heterogeneous distance computation is slightly slower than homogeneous because:
1. Voters must be classified by position
2. Distances are computed separately for each metric group

However, the overhead is minimal (~10-20%) and the feature provides significant modeling flexibility.

## Testing

Run the test suite:

```bash
python test_heterogeneous.py
```

Tests cover:
- Center-extreme strategy
- Radial steps strategy (all scaling functions)
- Integration with UtilityComputer
- Full simulation runs
- Configuration serialization

## Configuration Reference

### HeterogeneousDistanceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable heterogeneous distance |
| `strategy` | str | 'center_extreme' | Strategy type |
| `center_metric` | str | 'l2' | Metric for center voters |
| `extreme_metric` | str | 'cosine' | Metric for extreme voters |
| `extreme_threshold` | float | 0.5 | Threshold for extreme classification |
| `radial_metrics` | List[str] | ['l1','l2','chebyshev'] | Metrics from center outward |
| `radial_scaling` | str | 'linear' | Boundary scaling function |
| `scaling_parameter` | float | 2.0 | Parameter for non-linear scaling |

### Strategy Parameters

**center_extreme:**
- `center_metric`: Metric used by voters within threshold
- `extreme_metric`: Metric used by voters beyond threshold
- `extreme_threshold`: 0.0 = all use extreme, 1.0 = all use center

**radial_steps:**
- `radial_metrics`: List of metrics from center to edge
- `radial_scaling`: 'linear', 'logarithmic', or 'exponential'
- `scaling_parameter`: Base for log/exp scaling (higher = more compression)

## Future Extensions

Potential additions:
1. Custom strategy plugins
2. Per-voter metric assignment
3. Dynamic metrics (change during simulation)
4. Metric mixing (weighted combination)
5. Learned metric preferences

## See Also

- `VOTING_PATTERNS.md` - Voting theory background
- `GUI_README.md` - GUI usage guide
- `test_heterogeneous.py` - Test examples



