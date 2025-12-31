# Plurality vs Random Winner Test

## Overview
This test compares plurality voting to random winner selection across different spatial dimensions and candidate counts.

## Test Configuration
- **Monte Carlo Runs**: 1000 per configuration
- **Voters per Run**: 100
- **Distance Metric**: L2 (Euclidean)
- **Dimensions**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- **Candidate Counts**: 2, 3, 4, 5, 10
- **Total Configurations**: 50 (10 dimensions × 5 candidate counts)

## Metrics Tracked

1. **Average Distance to Ideal**: Mean distance from voters to their selected winner
2. **Winner Extremism**: Distance of winner from the center of the space
3. **Worst-off Distance**: Maximum distance from any voter to the winner
4. **Rule Disagreement Frequency**: How often Plurality winner ≠ Random winner
5. **Condorcet Consistency**: Fraction of runs where the rule picks the Condorcet winner
6. **Variance in Voter Satisfaction**: Variance of distances/utilities across voters

## Key Findings

### Average Distance to Ideal
- **Plurality consistently outperforms Random** across all dimensions and candidate counts
- The advantage increases with dimension count (e.g., at D=10, C=5: Plurality=1.167, Random=1.270)
- Both methods show increasing distances as dimensionality grows (curse of dimensionality)

### Winner Extremism
- **Plurality selects more moderate candidates** (closer to center) than Random
- The gap widens significantly with more candidates (e.g., at D=10, C=10: Plurality=0.720, Random=0.896)
- Extremism increases with dimension for both methods

### Disagreement Frequency
- For 2 candidates: ~50% disagreement (expected for binary choice)
- For 3 candidates: ~66% disagreement
- For 4 candidates: ~75% disagreement
- For 5 candidates: ~80% disagreement
- For 10 candidates: ~90% disagreement
- Relatively consistent across dimensions

### Condorcet Consistency
- **Plurality**: 100% for 2 candidates (Condorcet=Plurality), decreases with more candidates
  - At C=3: 63-91% depending on dimension
  - At C=10: 21-70% (increases with dimension)
- **Random**: Much lower at ~50% for 2 candidates, drops to ~10% for 10 candidates
- Plurality is 2-7× more likely to select the Condorcet winner

### Variance in Voter Satisfaction
- **Plurality shows lower variance** (more equitable outcomes) than Random
- For 2 candidates: Plurality≈0.037-0.049, Random≈0.049-0.054
- Gap narrows slightly as dimensions increase
- Both methods show relatively stable variance across dimensions

## Worst-off Voter Distance
- Plurality provides better outcomes for the worst-off voter across all configurations
- Gap increases with dimension (e.g., at D=10, C=5: Plurality=1.664, Random=1.793)

## Conclusions

1. **Plurality is systematically better than Random** across all quality metrics
2. **Strategic voting matters**: Even simple plurality voting captures voter preferences far better than random selection
3. **Condorcet compliance**: Plurality has much higher Condorcet consistency than random selection
4. **Equity**: Plurality provides more equitable outcomes (lower variance, better worst-off distances)
5. **Dimensionality effects**: Both methods degrade with dimension, but plurality degrades more gracefully

## Files Generated
- `test_plur_vs_random.py`: Test implementation
- `results_plur-rand.json`: Complete numerical results
- Console output includes formatted grids for all metrics

## Running the Test
```powershell
cd c:\Programming\fun\social-choice\heterogenity-testing
python test_plur_vs_random.py
```

Expected runtime: ~6-7 minutes for all 50 configurations (1000 runs each)

