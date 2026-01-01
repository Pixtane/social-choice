# Voter Scaling Verification Script

## Overview

This script (`verify_voter_scaling.py`) systematically verifies all 5 novel phenomena described in `FINDINGS.md` while testing how different numbers of voters affect each phenomenon.

## Purpose

The original findings in `FINDINGS.md` were based on experiments with 100 voters. This script:

1. **Verifies** that all 5 phenomena actually occur in simulations
2. **Tests** how voter count (50, 100, 200, 500, 1000) affects each phenomenon
3. **Analyzes** whether effects are stable across different voter counts
4. **Compares** results to expected values from the original findings

## Phenomena Tested

### 1. Asymmetric Metric Interaction
- **What it tests**: Whether the order of metric assignment matters (D(A→B) ≠ D(B→A))
- **Expected**: Asymmetry of ~6-8% for tested metric pairs
- **Voter scaling question**: Does asymmetry magnitude change with voter count?

### 2. Dimensional Scaling Laws
- **What it tests**: Whether effects peak at 2-3 dimensions
- **Expected**: Peak at dimension 2 (plurality/borda) or 3 (IRV)
- **Voter scaling question**: Does peak dimension remain stable?

### 3. Threshold Phase Transitions
- **What it tests**: Whether threshold response follows sigmoidal curves
- **Expected**: Maximum disagreement at specific thresholds (θ ≈ 0.6-0.8)
- **Voter scaling question**: Do critical thresholds shift with voter count?

### 4. Preference Destabilization Paradox
- **What it tests**: Whether heterogeneity increases cycles but improves efficiency
- **Expected**: Cycle increase ~3.5%, efficiency increase 0.8-7.0% depending on rule
- **Voter scaling question**: Does the paradox persist across voter counts?

### 5. Metric Interaction Strength Hierarchy
- **What it tests**: Whether different metric pairs have systematically different strengths
- **Expected**: Specific hierarchy (e.g., L2↔Chebyshev strongest)
- **Voter scaling question**: Does hierarchy order remain consistent?

## Usage

### Basic Usage

```bash
python heterogeneity-research/verify_voter_scaling.py
```

This runs with default parameters:
- 200 profiles per configuration
- 5 candidates
- Voter counts: [50, 100, 200, 500, 1000]

### Custom Parameters

```bash
python heterogeneity-research/verify_voter_scaling.py \
    --n-profiles 300 \
    --n-candidates 7 \
    --voter-counts 25 50 100 250 500 1000 \
    --output-dir custom_results
```

### Arguments

- `--n-profiles`: Number of election profiles per configuration (default: 200)
- `--n-candidates`: Number of candidates per profile (default: 5)
- `--voter-counts`: Space-separated list of voter counts to test (default: 50 100 200 500 1000)
- `--output-dir`: Directory to save results (default: `heterogeneity-research/voter_scaling_results`)

## Output

The script generates a JSON file with comprehensive results:

```json
{
  "timestamp": "20250101_120000",
  "parameters": {
    "n_profiles": 200,
    "n_candidates": 5,
    "voter_counts": [50, 100, 200, 500, 1000]
  },
  "results": {
    "phenomenon_1": {
      "phenomenon_name": "Asymmetric Metric Interaction",
      "voter_counts": [50, 100, 200, 500, 1000],
      "summary": {...},
      "verified": true,
      "scaling_results": [
        {
          "n_voters": 50,
          "results": {
            "test_pairs": {
              "asymmetric_l1_cosine_plurality": {
                "disagreement_ab": 82.5,
                "disagreement_ba": 88.1,
                "asymmetry": 5.6,
                ...
              }
            }
          }
        },
        ...
      ]
    },
    ...
  }
}
```

## Key Metrics Computed

For each phenomenon and voter count, the script computes:

### Asymmetric Interaction
- Disagreement rate for A→B assignment
- Disagreement rate for B→A assignment
- Asymmetry magnitude (|D(A→B) - D(B→A)|)
- Comparison to expected values

### Dimensional Scaling
- Disagreement rate for each dimension (1, 2, 3, 4, 5, 7, 10)
- Peak dimension identification
- Peak disagreement value
- Stability analysis across voter counts

### Threshold Phase Transitions
- Disagreement rate for each threshold (0.05 to 0.95, 19 points)
- Maximum disagreement threshold
- Range of disagreement values
- Sigmoid characteristics

### Destabilization Paradox
- Cycle rate (heterogeneous vs homogeneous)
- Condorcet efficiency (heterogeneous vs homogeneous)
- Cycle increase magnitude
- Efficiency increase magnitude
- Paradox verification (both increases present)

### Metric Hierarchy
- Interaction strength for each metric pair
- Hierarchy ordering (strongest to weakest)
- Stability of hierarchy across voter counts

## Analysis Features

The script includes several analysis functions:

1. **Trend Analysis**: Computes whether metrics increase, decrease, or remain stable across voter counts
2. **Peak Stability**: Checks if peak dimensions remain consistent
3. **Sigmoid Stability**: Analyzes variance in critical thresholds
4. **Paradox Stability**: Verifies if paradox persists across all voter counts
5. **Hierarchy Stability**: Checks if metric pair ordering remains consistent

## Expected Runtime

Approximate runtime per voter count:
- 50 voters: ~2-3 minutes
- 100 voters: ~3-5 minutes
- 200 voters: ~5-8 minutes
- 500 voters: ~10-15 minutes
- 1000 voters: ~20-30 minutes

**Total runtime**: ~1-2 hours for all 5 voter counts and 5 phenomena

## Interpretation

### What to Look For

1. **Stability**: Do effects remain consistent across voter counts?
2. **Scaling**: Do effects increase, decrease, or remain constant with more voters?
3. **Verification**: Do observed values match expected values from FINDINGS.md?
4. **Robustness**: Are phenomena robust to changes in electorate size?

### Key Questions

- **Asymmetry**: Does asymmetry magnitude change with voter count? (Expected: stable)
- **Peak Dimension**: Does peak dimension shift? (Expected: stable at 2-3D)
- **Critical Threshold**: Do sigmoid inflection points shift? (Expected: stable)
- **Paradox**: Does paradox persist? (Expected: yes, always)
- **Hierarchy**: Does metric pair ordering change? (Expected: stable)

## Comparison to Original Findings

The script compares results to expected values from `FINDINGS.md`:

| Phenomenon | Expected Value | Comparison |
|------------|---------------|------------|
| L1↔Cosine asymmetry (Plurality) | Δ = 6.0% | Computed for each voter count |
| Dimensional peak (Plurality) | d = 2 | Checked for stability |
| Threshold max (Plurality) | D = 84.0% | Compared across voter counts |
| Cycle increase | Δ = +3.5% | Verified for each voter count |
| Efficiency increase (Plurality) | Δ = +7.0% | Verified for each voter count |

## Notes

- All experiments use fixed random seed (42) for reproducibility
- Same spatial profiles are used across voter counts (when possible)
- Results are saved with timestamps for tracking
- The script prints progress indicators during execution

## Dependencies

- numpy
- simulator package (from parent directory)
- Standard library: json, pathlib, typing, dataclasses, datetime

## Example Output Interpretation

If asymmetry is stable across voter counts:
```
Asymmetry trend: stable (slope ≈ 0)
Mean asymmetry: 6.2% ± 0.3%
```

If peak dimension shifts:
```
Peak stability: NOT STABLE
Peaks observed: [2, 2, 3, 3, 3]  # Shifts at higher voter counts
```

If paradox persists:
```
Paradox always present: TRUE
Cycle increase: 3.5% ± 0.2% (stable)
Efficiency increase: 7.0% ± 0.5% (stable)
```

## Future Extensions

Potential enhancements:
- Test with different numbers of candidates
- Test with different spatial geometries
- Add statistical significance testing
- Generate visualization plots
- Compare to theoretical predictions


