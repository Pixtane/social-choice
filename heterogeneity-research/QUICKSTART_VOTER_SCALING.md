# Quick Start: Voter Scaling Verification

## Run the Script

```bash
# Basic run (uses defaults: 200 profiles, 5 candidates, voters [50,100,200,500,1000])
python heterogeneity-research/verify_voter_scaling.py

# Quick test with fewer voters (faster)
python heterogeneity-research/verify_voter_scaling.py --voter-counts 50 100 200

# More thorough test
python heterogeneity-research/verify_voter_scaling.py --n-profiles 300 --voter-counts 50 100 200 500 1000
```

## What It Does

The script tests all 5 phenomena from `FINDINGS.md` across different voter counts:

1. **Asymmetric Metric Interaction** - Does order matter? (A→B vs B→A)
2. **Dimensional Scaling** - Do effects peak at 2-3D?
3. **Threshold Phase Transitions** - Are there sigmoidal responses?
4. **Destabilization Paradox** - More cycles but better efficiency?
5. **Metric Hierarchy** - Do metric pairs have consistent ordering?

## Expected Output

The script will:
- Print progress for each phenomenon and voter count
- Save detailed JSON results to `heterogeneity-research/voter_scaling_results/`
- Save a human-readable summary text file
- Print a final verification summary

## Example Output

```
================================================================================
PHENOMENON 1: ASYMMETRIC METRIC INTERACTION (Voter Scaling)
================================================================================

Testing with 50 voters...
  l1↔cosine (plurality)... Δ=5.6%
  l1↔chebyshev (borda)... Δ=7.1%
  l1↔l2 (irv)... Δ=8.2%

Testing with 100 voters...
  l1↔cosine (plurality)... Δ=6.0%
  ...

Results saved to: heterogeneity-research/voter_scaling_results/voter_scaling_verification_20250101_120000.json
Summary saved to: heterogeneity-research/voter_scaling_results/voter_scaling_summary_20250101_120000.txt
```

## Runtime Estimates

- **50 voters**: ~2-3 min per phenomenon
- **100 voters**: ~3-5 min per phenomenon  
- **200 voters**: ~5-8 min per phenomenon
- **500 voters**: ~10-15 min per phenomenon
- **1000 voters**: ~20-30 min per phenomenon

**Total for all 5 phenomena × 5 voter counts**: ~1-2 hours

## Quick Test (Faster)

To test quickly with fewer configurations:

```bash
# Test only first 2 phenomena with 3 voter counts
python -c "
from heterogeneity_research.verify_voter_scaling import VoterScalingVerifier
v = VoterScalingVerifier(n_profiles=100, voter_counts=[50, 100, 200])
v.verify_phenomenon_1_asymmetric_interaction()
v.verify_phenomenon_2_dimensional_scaling()
"
```

## Interpreting Results

### Good Signs ✓
- Asymmetry remains stable across voter counts
- Peak dimensions don't shift
- Paradox persists (cycles increase AND efficiency increases)
- Hierarchy order remains consistent

### Things to Investigate
- Large changes in asymmetry with voter count
- Peak dimension shifts (e.g., 2D → 3D)
- Paradox disappears at certain voter counts
- Hierarchy order changes

## Output Files

1. **JSON file**: `voter_scaling_verification_TIMESTAMP.json`
   - Complete data for all phenomena and voter counts
   - Use for detailed analysis or plotting

2. **Summary file**: `voter_scaling_summary_TIMESTAMP.txt`
   - Human-readable overview
   - Quick reference for key metrics

## Next Steps

After running, you can:
1. Compare results to `FINDINGS.md` expected values
2. Analyze trends across voter counts
3. Generate visualizations from JSON data
4. Test additional voter counts if needed

## Troubleshooting

**Script is slow?**
- Reduce `--n-profiles` (e.g., 100 instead of 200)
- Test fewer voter counts
- Test fewer phenomena (modify script)

**Out of memory?**
- Reduce voter counts (skip 1000)
- Reduce number of profiles

**Results don't match FINDINGS.md?**
- Check random seed (should be 42)
- Verify same parameters (n_candidates=5, n_dim=2, etc.)
- Statistical variation is expected with fewer profiles


