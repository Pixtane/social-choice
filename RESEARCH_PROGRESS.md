# Research Progress: Proper Disagreement Decomposition Implementation

## What Was Wrong

The original code only computed **simple disagreement** (het ≠ center), which doesn't tell us:
- Whether heterogeneity **creates** new outcomes (strong disagreement)
- Whether heterogeneity **amplifies** extreme-metric outcomes (extreme-aligned disagreement)

## What We Fixed

Added `compute_disagreement_decomposition()` method that implements METHODOLOGY.md:

### Strong Disagreement (D_strong)
- Measures: het ≠ center AND het ≠ extreme
- Interpretation: **Outcome creation** - heterogeneity produces novel winner

### Extreme-Aligned Disagreement (D_ext-align)
- Measures: het = extreme AND het ≠ center
- Interpretation: **Outcome amplification** - heterogeneity shifts toward extreme metric

### Total Disagreement
- D_total = D_strong + D_ext-align
- Should equal simple disagreement vs center (validation check)

## Experiments Running

Full research suite with proper decomposition:
1. **Voter scaling** (10-500 voters)
2. **Threshold sweep** (0.05-0.95, 19 points)
3. **Dimensional scaling** (d=1,2,3,4,5,7,10)
4. **Metric pairs** (all 12 ordered pairs)

## Expected New Insights

With decomposition, we can now answer:

1. **Does L2-Cosine show zero disagreement because:**
   - No strong disagreement (no new outcomes)?
   - Or strong + extreme-aligned cancel out?

2. **For Cosine-L1 pairs (58% disagreement):**
   - How much is strong (new outcomes)?
   - How much is extreme-aligned (amplification)?

3. **How does decomposition change with dimension?**
   - Does strong disagreement decay faster than extreme-aligned?
   - At what dimension does outcome creation disappear?

4. **Are asymmetries in strong vs extreme-aligned different?**
   - Which metric pairs show asymmetric creation?
   - Which show asymmetric amplification?

## Next Steps

1. Wait for experiments to complete (~15-20 minutes)
2. Analyze decomposition patterns
3. Look for interesting phenomena:
   - Unexpected crossovers between strong/extreme-aligned
   - Dimension-dependent phase transitions
   - Metric-specific behaviors
4. Run targeted experiments to investigate discoveries
5. Update paper with verified decomposed findings
