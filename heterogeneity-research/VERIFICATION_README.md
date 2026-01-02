# Experimental Verification of Findings

This directory contains scripts to experimentally verify the findings documented in `FINDINGS.md`.

## Overview

The `verify_findings.py` script systematically tests each of the 5 novel phenomena discovered in heterogeneous distance metrics:

1. **Asymmetric Metric Interaction** - Order of metric assignment matters
2. **Dimensional Scaling Laws** - Effects peak at 2-3 dimensions
3. **Threshold Phase Transitions** - Sigmoidal response curves
4. **Preference Destabilization Paradox** - More cycles but better efficiency
5. **Metric Interaction Strength Hierarchy** - Systematic strength differences

## Usage

### Full Verification (Recommended)

Runs comprehensive tests with 200 profiles per configuration:

```bash
python verify_findings.py
```

This will take approximately 1-2 hours depending on your system.

### Quick Verification

For faster testing with reduced sample sizes, modify the script:

```python
verifier = FindingsVerifier(n_profiles=50, n_voters=100, n_candidates=5)
```

## What Gets Tested

### Finding 1: Asymmetric Metric Interaction

Tests that `D(A→B) ≠ D(B→A)` for metric pairs:
- L1 ↔ Cosine (Plurality): Expected Δ = 6.0%
- L1 ↔ Chebyshev (Borda): Expected Δ = 7.3%
- L1 ↔ L2 (IRV): Expected Δ = 8.0%

**Verification Criteria**: Asymmetry exists and is within 3% of expected values.

### Finding 2: Dimensional Scaling Laws

Tests power-law scaling with peak at 2-3 dimensions:
- Plurality: Peak at d=2, α=0.263, D(2)=84.7%
- Borda: Peak at d=2, α=0.099, D(2)=83.3%
- IRV: Peak at d=3, α=0.008, D(3)=82.0%

**Verification Criteria**: Peak dimension matches and peak disagreement within 5% of expected.

### Finding 3: Threshold Phase Transitions

Tests sigmoidal response curves:
- Plurality: D_max=84.0% at θ=0.6, range=13.5%
- Borda: D_max=83.5% at θ=0.7, range=7.5%
- IRV: D_max=85.0% at θ=0.15, range=8.5%

**Verification Criteria**: Maximum disagreement and threshold within tolerance.

### Finding 4: Preference Destabilization Paradox

Tests that cycles increase while efficiency improves:
- Plurality: ΔC = +3.5%, ΔE = +7.0%
- Borda: ΔC = +3.5%, ΔE = +1.2%
- IRV: ΔC = +3.5%, ΔE = +0.8%

**Verification Criteria**: Both cycle increase and efficiency improvement occur.

### Finding 5: Metric Interaction Strength Hierarchy

Tests systematic ordering of metric pair strengths (Plurality):
1. Cosine ↔ L1: 87.3%
2. L2 ↔ Chebyshev: 86.0%
3. L1 ↔ L2: 84.0%
4. L2 ↔ Cosine: 82.7%
5. Cosine ↔ Chebyshev: 79.3%

**Verification Criteria**: Ordering matches and values within 5% tolerance.

## Output

The script generates:

1. **Console Output**: Real-time progress and verification status
2. **JSON Results**: `verification_results/verification_results.json` with detailed data

### Result Format

```json
{
  "finding_1": {
    "finding_name": "Asymmetric Metric Interaction",
    "verified": true,
    "expected_values": {...},
    "observed_values": {...},
    "notes": [...]
  },
  ...
}
```

## Interpretation

- **Verified**: Finding matches documented values within tolerance
- **Failed**: Finding does not match or phenomenon not observed

Note: Some variation is expected due to:
- Random sampling differences
- Finite sample sizes
- Numerical precision

Tolerances are set to account for reasonable experimental variation.

## Dependencies

- numpy
- simulator module (from parent directory)
- Standard Python libraries

## Notes

- Uses fixed random seed (42) for reproducibility
- Each test runs independent simulations
- Comparisons are made against homogeneous baselines
- All tests use 2D spatial geometry unless testing dimensionality







