# MVP Variable Requirements for Heterogeneity Research

This document lists the **minimum viable set of variables** (from `VARIABLES.md`) required to simulate and reproduce the findings in `FINDINGS.md`.

---

## LAYER 0 - Structural Size Parameters

**REQUIRED:**
- **Number of voters**: 100 (fixed per profile)
- **Number of candidates**: 5 (fixed per profile)
- **Issue space dimension**: Variable (1, 2, 3, 4, 5, 7, 10)
- **Number of elections**: 150-200 Monte Carlo runs per configuration

**Notes**: These are the core scale parameters. Dimension must be variable to study scaling laws (Phenomenon 2).

---

## LAYER 1 - Issue Space Geometry

**REQUIRED:**
- **Space type**: Euclidean (standard spatial voting)
- **Axis meaning**: Independent issues
- **Axis scaling**: Uniform

**Notes**: Standard Euclidean space is sufficient. No need for manifolds or graph topologies.

---

## LAYER 2 - Voter Ideal-Point Generation

**REQUIRED:**
- **Distribution family**: Center-extreme strategy (threshold-based)
  - Center voters: Compact cluster near origin
  - Extreme voters: Distributed in periphery
- **Radial distribution**: Bimodal (center-heavy + edge-heavy)
- **Threshold parameter θ**: Controls assignment (0.05 to 0.95 in 0.05 increments)

**CRITICAL**: The threshold θ is the key parameter for:
- **Phenomenon 1** (Asymmetric Metric Interaction): Order of metric assignment matters
- **Phenomenon 3** (Threshold Phase Transitions): Sigmoidal response curves

**Implementation**: Voters with distance from origin < θ percentile → "center" voters; others → "extreme" voters.

---

## LAYER 3 - Candidate Position Generation

**REQUIRED:**
- **Candidate distribution**: Uniform random (or same as voters - not explicitly specified)
- **Candidate count regime**: Fixed (5 candidates)
- **Position constraints**: Must be distinct

**Notes**: Exact distribution not critical, but must be consistent across runs.

---

## LAYER 4 - Distance (Geometry) Functions

**REQUIRED:**
- **Metric family**: 
  - L1 (Manhattan)
  - L2 (Euclidean)
  - L∞ (Chebyshev)
  - Cosine similarity
- **Heterogeneity**: Per voter (center vs extreme assignment)
- **Regime switching**: Radius-based (using threshold θ)

**CRITICAL**: This is the core of all findings!

**Requirements**:
1. Must support **all 12 metric pairs**: (L1, L2, Cosine, Chebyshev) × 2 orders = 12 combinations
2. Must assign metrics based on **radial distance from origin** (threshold θ)
3. Must support **asymmetric assignment**: Metric A to center, Metric B to extreme (order matters!)
4. Must support **homogeneous baseline**: All voters use same metric (for comparison)

**Implementation**:
- For each voter, compute distance from origin: `r = ||voter_position||`
- If `r < θ_percentile`: assign metric A (center metric)
- Else: assign metric B (extreme metric)
- Must be able to swap: A→B vs B→A to test asymmetry

---

## LAYER 5 - Utility Functions

**REQUIRED:**
- **Utility mapping**: Linear or quadratic (standard spatial voting)
  - Typically: `u = -distance` or `u = -distance²`
- **Utility bounds**: Unbounded (standard)
- **Heterogeneity**: None needed (utility is derived from distance)

**Notes**: Standard utility mapping is sufficient. The heterogeneity effects come from distance metrics, not utility functions.

---

## LAYER 6 - Preference Formation

**REQUIRED:**
- **Strict vs weak preferences**: Strict (no ties)
- **Noise in evaluation**: None (deterministic preferences from utilities)

**Notes**: Preferences are deterministic rankings based on utility (distance). No strategic misreporting needed.

---

## LAYER 7 - Acceptability / Thresholds

**NOT REQUIRED**: This layer is not used in the heterogeneity research.

---

## LAYER 8 - Voting Rule

**REQUIRED:**
- **Voting system**: 
  - Plurality
  - Borda
  - IRV (Instant Runoff Voting)
- **Tie-breaking rules**: Any deterministic method (random, lexicographic, geometric)
- **Ballot type**: Ordinal (rankings)

**Notes**: These three rules are sufficient to demonstrate rule-dependent effects.

---

## LAYER 9 - Aggregation Mechanics

**REQUIRED:**
- **Standard implementations** for each voting rule
- **Pairwise margin definition**: Simple majority (for Condorcet analysis)

**Notes**: Standard aggregation is sufficient. Need pairwise comparisons for Condorcet cycle detection.

---

## LAYER 10 - Post-Processing / Outcome Interpretation

**REQUIRED:**
- **Single winner**: Yes (one winner per profile)

---

## LAYER 11 - Metrics (What You Record)

**REQUIRED:**
- **Disagreement rate**: Percentage of profiles where heterogeneous winner ≠ homogeneous winner
- **VSE difference**: Change in Voter Satisfaction Efficiency (heterogeneous - homogeneous)
- **Cycle rate**: Percentage of profiles with Condorcet cycles (intransitive preferences)
- **Condorcet efficiency**: Percentage of profiles where voting rule selects Condorcet winner (when one exists)

**CRITICAL METRICS**:
1. **Disagreement rate** (primary metric for all 5 phenomena)
2. **Cycle rate** (for Phenomenon 4: Preference Destabilization Paradox)
3. **Condorcet efficiency** (for Phenomenon 4: Preference Destabilization Paradox)
4. **VSE difference** (mentioned but not primary focus)

**Implementation**:
- For each profile, run **two simulations**:
  1. **Heterogeneous**: Voters use different metrics (center vs extreme)
  2. **Homogeneous**: All voters use same metric (baseline)
- Compare winners and compute metrics

---

## LAYER 12 - Temporal & Cultural Variation

**NOT REQUIRED**: This layer is not used in the heterogeneity research.

---

## Summary: MVP Variable Checklist

### Core Requirements (Must Have)

✅ **LAYER 0**: Variable dimensions (1-10D), 100 voters, 5 candidates, 150-200 runs  
✅ **LAYER 2**: Threshold-based center-extreme voter distribution (θ parameter)  
✅ **LAYER 4**: 4 metrics (L1, L2, Cosine, Chebyshev) with per-voter heterogeneity  
✅ **LAYER 4**: Radius-based regime switching (threshold θ controls assignment)  
✅ **LAYER 4**: Asymmetric metric assignment (order matters: A→B vs B→A)  
✅ **LAYER 8**: 3 voting rules (Plurality, Borda, IRV)  
✅ **LAYER 11**: 4 metrics (disagreement rate, cycle rate, Condorcet efficiency, VSE difference)

### Secondary Requirements (Should Have)

✅ **LAYER 1**: Euclidean space, uniform scaling  
✅ **LAYER 3**: Fixed candidate count, distinct positions  
✅ **LAYER 5**: Linear/quadratic utility mapping  
✅ **LAYER 6**: Strict preferences, no noise

### Not Required

❌ **LAYER 7**: Acceptability thresholds  
❌ **LAYER 12**: Temporal variation

---

## Experimental Configuration Space

To reproduce all findings, the simulator must support:

1. **Dimensions**: 1, 2, 3, 4, 5, 7, 10 (7 values)
2. **Thresholds**: 0.05 to 0.95 in 0.05 increments (19 values)
3. **Metric pairs**: 12 combinations (4 metrics × 2 orders)
4. **Voting rules**: 3 rules (Plurality, Borda, IRV)
5. **Baseline comparison**: Homogeneous vs heterogeneous (2 modes)

**Total configurations**: 7 × 19 × 12 × 3 × 2 = **9,576 configurations**

Each configuration needs 150-200 profiles, so approximately **1.4M - 1.9M total profiles**.

---

## Key Implementation Details

### Threshold-Based Metric Assignment

```python
# Pseudocode for threshold-based assignment
for voter in voters:
    r = distance_from_origin(voter.position)
    r_percentile = percentile_rank(r, all_voter_distances)
    
    if r_percentile < theta:
        voter.metric = center_metric  # e.g., L2
    else:
        voter.metric = extreme_metric  # e.g., Cosine
```

### Asymmetric Testing

For each metric pair (A, B), test **both orders**:
- **Order 1**: A → center, B → extreme
- **Order 2**: B → center, A → extreme

This reveals the asymmetry (Phenomenon 1).

### Disagreement Rate Calculation

```python
for profile in profiles:
    winner_homogeneous = run_election(profile, metric=all_same)
    winner_heterogeneous = run_election(profile, metric=threshold_based)
    
    if winner_homogeneous != winner_heterogeneous:
        disagreements += 1

disagreement_rate = disagreements / total_profiles
```

---

## Minimal "Serious" Subset for This Research

Following `VARIABLES.md`'s recommendation, the minimal subset is:

1. ✅ **Voter distribution**: Center-extreme (threshold-based)
2. ✅ **Distance family**: 4 metrics (L1, L2, Cosine, Chebyshev) + heterogeneity
3. ✅ **Utility mapping**: Linear/quadratic
4. ✅ **Voting rule**: 3 rules (Plurality, Borda, IRV)
5. ✅ **Metrics**: 4 metrics (disagreement, cycle rate, Condorcet efficiency, VSE)

**Plus**: Threshold parameter θ and variable dimensions (d).

---

## Notes

- The **threshold parameter θ** is the most critical novel parameter not in standard spatial voting models
- **Metric assignment order** (asymmetry) must be explicitly supported
- **Homogeneous baseline** comparison is essential for all metrics
- **Condorcet cycle detection** requires pairwise comparisons
- **Dimensional scaling** requires variable dimension support (1-10D)

---

_Generated from analysis of `FINDINGS.md` and `VARIABLES.md`_


