# Runtime Estimate for Visualization Script

Based on your observation that one direction (l1 → cosine) took ~3 minutes, here's the breakdown:

## Experiment Count Breakdown

Each "experiment run" includes both heterogeneous and homogeneous simulations.

### 1. Asymmetric Metric Interaction

- **3 pairs** × **2 directions** = **6 experiment runs**
- **Estimated time**: 6 × 3 min = **~18 minutes**

### 2. Dimensional Scaling Laws

- **7 dimensions** × **2 configs** (het + homo) = **14 experiment runs**
- **Estimated time**: 14 × 3 min = **~42 minutes**

### 3. Threshold Phase Transitions

- **19 thresholds** × **2 configs** (het + homo) = **38 experiment runs**
- **Estimated time**: 38 × 3 min = **~114 minutes** = **~1.9 hours**

### 4. Preference Destabilization Paradox

- **2 configs** (het + homo) = **2 experiment runs**
- **Estimated time**: 2 × 3 min = **~6 minutes**

### 5. Metric Interaction Strength Hierarchy

- **12 pairs** (4×4 - 4 diagonal) × **2 configs** (het + homo) = **24 experiment runs**
- **Estimated time**: 24 × 3 min = **~72 minutes** = **~1.2 hours**

## Total Estimated Runtime

**~252 minutes = ~4.2 hours**

## Breakdown by Visualization

| Visualization                 | Experiment Runs | Estimated Time    |
| ----------------------------- | --------------- | ----------------- |
| 1. Asymmetric Interaction     | 6               | ~18 min           |
| 2. Dimensional Scaling        | 14              | ~42 min           |
| 3. Threshold Transitions      | 38              | **~1.9 hours** ⚠️ |
| 4. Preference Destabilization | 2               | ~6 min            |
| 5. Interaction Hierarchy      | 24              | **~1.2 hours**    |
| **TOTAL**                     | **84**          | **~4.2 hours**    |

## Notes

- The longest single visualization is **#3 (Threshold Phase Transitions)** at ~1.9 hours
- Visualization **#5 (Interaction Hierarchy)** is second longest at ~1.2 hours
- These estimates assume consistent performance (3 min per experiment run)
- Actual time may vary based on system load and CPU performance

## Optimization Suggestions

If you want to speed things up, you could:

1. Reduce `RESEARCH_N_PROFILES` from 200 to 100 (halves time, slightly less statistical power)
2. Reduce `RESEARCH_N_VOTERS` from 10,000 to 5,000 (halves time, but less "research scale")
3. Run visualizations separately (comment out others in `main()`)
4. Reduce threshold count in #3 from 19 to 10 points (saves ~1 hour)





