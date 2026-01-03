# Impact of First Normalization (Opposite Corner) Analysis

## Summary

We tested whether the first normalization step (using opposite corner to compute `d_max_v`) actually matters in practice when using heterogeneous distance metrics.

## Key Findings

### 1. Utility Values DO Change Significantly

- **Maximum difference**: Up to 0.49-0.67 in utility values
- **Mean difference**: ~0.18-0.20 across typical profiles
- The first normalization produces meaningfully different utility values than just best-worst scaling

### 2. Rankings (Preference Orders) Do NOT Change

- Across all tested scenarios, **rankings remain identical**
- This means for **ordinal voting rules** (which only use rankings), the first normalization doesn't matter

### 3. Voting Outcomes: Mixed Results

**Score Voting**:
- **0% difference** in winners across 100 trials
- Even though utility values differ, the relative ordering and score distributions remain similar enough that winners don't change

**Approval Voting**:
- **18% difference** in winners across 100 trials
- This is because approval voting uses a threshold (e.g., 0.5), and the different utility values can push candidates above/below the threshold differently

## Why This Happens

### Without First Normalization:
- Each voter's utilities are normalized to [0,1] based on their actual best/worst candidates in the profile
- Utilities are relative to the specific candidate set

### With First Normalization:
- Each voter's utilities are normalized to [0,1] based on the theoretical maximum distance (opposite corner)
- Utilities are relative to the entire possible space, not just the candidates present
- This accounts for each voter's position and metric type

### Example:
Consider a voter using L∞ metric at position (0.5, 0.5):
- **Without normalization**: If candidates are all close, distances might be [0.1, 0.2, 0.3] → utilities [1.0, 0.5, 0.0]
- **With normalization**: Opposite corner distance is 0.5, so same distances [0.1, 0.2, 0.3] → utilities [0.8, 0.6, 0.4]

The rankings are the same, but the absolute utility values differ.

## Implications

### For Ordinal Voting Rules (Plurality, Borda, Copeland, etc.):
- **First normalization is unnecessary** - rankings don't change
- You could skip it and just use best-worst scaling

### For Cardinal Voting Rules:
- **First normalization matters** for rules that use utility thresholds (like Approval)
- **First normalization may not matter** for rules that use relative utilities (like Score Voting in our tests)
- Depends on how the rule uses the utility values

### For Heterogeneous Metrics:
- The first normalization ensures utilities are on a comparable scale across different metrics
- Without it, a distance of 0.6 under L1 means something different than 0.6 under L∞
- This is important for **interpersonal utility comparison** and **social welfare calculations**

## Conclusion

The first normalization:
1. **Changes utility values significantly** (up to 0.5-0.7 difference)
2. **Does NOT change rankings** (preference orders stay the same)
3. **CAN change voting outcomes** for threshold-based rules like Approval (18% of cases)
4. **Does NOT change outcomes** for score-based rules in our tests (0% of cases)

**Recommendation**: Keep the first normalization if you:
- Use heterogeneous distance metrics
- Need interpersonal utility comparison
- Use threshold-based voting rules (like Approval)
- Want utilities to be comparable across voters with different metrics

You could potentially skip it if you:
- Only use ordinal voting rules
- Don't need interpersonal utility comparison
- Want simpler code
