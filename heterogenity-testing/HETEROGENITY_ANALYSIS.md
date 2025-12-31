# Heterogeneity Testing Analysis

This document contains analysis of 15 heterogeneity experiments.

## Test 1: Vary L2/Cosine Fraction

### Configuration

- **l2_fraction**: 0.5

### Key Metrics

- **Rule Disagreement**: 55.00%

#### Average Distance to Ideal

- **plurality**: 0.4685 ± 0.0695
- **borda**: 0.4364 ± 0.0505
- **ranked_pairs**: 0.4359 ± 0.0540

#### Winner Extremism

- **plurality**: 0.2995 ± 0.1414
- **borda**: 0.2293 ± 0.1284
- **ranked_pairs**: 0.2253 ± 0.1275

#### Worst-Off Distance

- **plurality**: 0.9091 ± 0.1459
- **borda**: 0.8429 ± 0.1279
- **ranked_pairs**: 0.8425 ± 0.1295

#### Condorcet Consistency

- **plurality**: 58.00%
- **borda**: 87.00%
- **ranked_pairs**: 100.00%

---

## Test 10: Candidate Count Sweep

### Configuration

- **candidate_counts**: [3, 5, 10]

### Key Metrics

- **Rule Disagreement**: 69.00%

#### Average Distance to Ideal

- **plurality**: 0.4706 ± 0.0735
- **borda**: 0.4246 ± 0.0423
- **ranked_pairs**: 0.4287 ± 0.0466

#### Winner Extremism

- **plurality**: 0.3002 ± 0.1387
- **borda**: 0.2100 ± 0.1035
- **ranked_pairs**: 0.2182 ± 0.1139

#### Worst-Off Distance

- **plurality**: 0.9078 ± 0.1338
- **borda**: 0.8221 ± 0.1024
- **ranked_pairs**: 0.8307 ± 0.1091

#### Condorcet Consistency

- **plurality**: 46.00%
- **borda**: 86.00%
- **ranked_pairs**: 100.00%

---

## Test 11: Outlier Voters

### Configuration

- **experiment_id**: b9f26a72
- **created_at**: 2025-12-31T13:00:01.647378
- **n_profiles**: 100
- **n_voters**: 100
- **n_candidates**: 5
- **voting_rules**: ['plurality', 'borda', 'ranked_pairs']
- **geometry_method**: uniform
- **geometry_n_dim**: 2
- **geometry_phi**: 0.5
- **manipulation_enabled**: False
- **manipulation_fraction**: 0.2
- **manipulation_strategy**: compromise
- **utility_function**: linear
- **utility_distance_metric**: l2
- **utility_sigma_factor**: 0.5
- **rng_seed**: None
- **epsilon**: 1e-09
- **heterogeneous_distance_enabled**: False

### Key Metrics

- **Rule Disagreement**: 52.00%

#### Average Distance to Ideal

- **plurality**: 0.4522 ± 0.0677
- **borda**: 0.4262 ± 0.0411
- **ranked_pairs**: 0.4238 ± 0.0383

#### Winner Extremism

- **plurality**: 0.2721 ± 0.1269
- **borda**: 0.2214 ± 0.1030
- **ranked_pairs**: 0.2186 ± 0.1019

#### Worst-Off Distance

- **plurality**: 0.8783 ± 0.1301
- **borda**: 0.8271 ± 0.0986
- **ranked_pairs**: 0.8255 ± 0.0953

#### Condorcet Consistency

- **plurality**: 58.00%
- **borda**: 87.00%
- **ranked_pairs**: 100.00%

---

## Test 12: Noise in Voter Perception

### Configuration

- **noise_std**: 0.05

### Key Metrics

- **Rule Disagreement**: 57.00%

#### Average Distance to Ideal

- **plurality**: 0.4569 ± 0.0607
- **borda**: 0.4303 ± 0.0424
- **ranked_pairs**: 0.4293 ± 0.0426

#### Winner Extremism

- **plurality**: 0.2731 ± 0.1246
- **borda**: 0.2151 ± 0.1052
- **ranked_pairs**: 0.2126 ± 0.1066

#### Worst-Off Distance

- **plurality**: 0.8841 ± 0.1229
- **borda**: 0.8287 ± 0.1034
- **ranked_pairs**: 0.8274 ± 0.1073

#### Condorcet Consistency

- **plurality**: 55.00%
- **borda**: 89.00%
- **ranked_pairs**: 100.00%

---

## Test 13: Hybrid Distance Switching by Candidate Location

### Configuration

- **center_threshold**: 0.3

### Key Metrics

- **Rule Disagreement**: 40.00%

#### Average Distance to Ideal

- **plurality**: 0.5351 ± 0.0651
- **borda**: 0.5427 ± 0.0719
- **ranked_pairs**: 0.5442 ± 0.0705

#### Winner Extremism

- **plurality**: 0.4301 ± 0.0866
- **borda**: 0.4373 ± 0.0978
- **ranked_pairs**: 0.4380 ± 0.0951

#### Worst-Off Distance

- **plurality**: 1.0271 ± 0.0984
- **borda**: 1.0332 ± 0.1103
- **ranked_pairs**: 1.0358 ± 0.1064

#### Condorcet Consistency

- **plurality**: 71.00%
- **borda**: 88.00%
- **ranked_pairs**: 100.00%

---

## Test 14: Incremental Heterogeneity Sweep

### Configuration

- **non_l2_fractions**: [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0]

### Key Metrics

- **Rule Disagreement**: 57.00%

#### Average Distance to Ideal

- **plurality**: 0.4596 ± 0.0673
- **borda**: 0.4322 ± 0.0501
- **ranked_pairs**: 0.4306 ± 0.0498

#### Winner Extremism

- **plurality**: 0.2839 ± 0.1365
- **borda**: 0.2131 ± 0.1159
- **ranked_pairs**: 0.2115 ± 0.1155

#### Worst-Off Distance

- **plurality**: 0.8878 ± 0.1450
- **borda**: 0.8180 ± 0.1170
- **ranked_pairs**: 0.8176 ± 0.1136

#### Condorcet Consistency

- **plurality**: 53.00%
- **borda**: 89.00%
- **ranked_pairs**: 100.00%

---

## Test 15: Saturated Utility + Heterogeneous Distance

### Configuration

- **experiment_id**: 2e0bc6d7
- **created_at**: 2025-12-31T13:00:09.862165
- **n_profiles**: 100
- **n_voters**: 100
- **n_candidates**: 5
- **voting_rules**: ['plurality', 'borda', 'ranked_pairs']
- **geometry_method**: uniform
- **geometry_n_dim**: 2
- **geometry_phi**: 0.5
- **manipulation_enabled**: False
- **manipulation_fraction**: 0.2
- **manipulation_strategy**: compromise
- **utility_function**: saturated
- **utility_distance_metric**: l2
- **utility_sigma_factor**: 0.5
- **rng_seed**: None
- **epsilon**: 1e-09
- **heterogeneous_distance_enabled**: False

### Key Metrics

- **Rule Disagreement**: 56.00%

#### Average Distance to Ideal

- **plurality**: 0.4673 ± 0.0637
- **borda**: 0.4465 ± 0.0592
- **ranked_pairs**: 0.4501 ± 0.0610

#### Winner Extremism

- **plurality**: 0.2996 ± 0.1254
- **borda**: 0.2600 ± 0.1304
- **ranked_pairs**: 0.2664 ± 0.1341

#### Worst-Off Distance

- **plurality**: 0.9080 ± 0.1197
- **borda**: 0.8701 ± 0.1233
- **ranked_pairs**: 0.8786 ± 0.1268

#### Condorcet Consistency

- **plurality**: 62.00%
- **borda**: 91.00%
- **ranked_pairs**: 100.00%

---

## Test 3: Distance Rule Depends on Radius

### Configuration

- **experiment_id**: 7d9b6393
- **created_at**: 2025-12-31T12:59:52.912366
- **n_profiles**: 100
- **n_voters**: 100
- **n_candidates**: 5
- **voting_rules**: ['plurality', 'borda', 'ranked_pairs']
- **geometry_method**: uniform
- **geometry_n_dim**: 2
- **geometry_phi**: 0.5
- **manipulation_enabled**: False
- **manipulation_fraction**: 0.2
- **manipulation_strategy**: compromise
- **utility_function**: linear
- **utility_distance_metric**: l2
- **utility_sigma_factor**: 0.5
- **rng_seed**: None
- **epsilon**: 1e-09
- **heterogeneous_distance_enabled**: False

### Key Metrics

- **Rule Disagreement**: 52.00%

#### Average Distance to Ideal

- **plurality**: 0.4477 ± 0.0596
- **borda**: 0.4309 ± 0.0519
- **ranked_pairs**: 0.4297 ± 0.0493

#### Winner Extremism

- **plurality**: 0.2678 ± 0.1210
- **borda**: 0.2218 ± 0.1116
- **ranked_pairs**: 0.2269 ± 0.1166

#### Worst-Off Distance

- **plurality**: 0.8813 ± 0.1193
- **borda**: 0.8319 ± 0.1130
- **ranked_pairs**: 0.8347 ± 0.1136

#### Condorcet Consistency

- **plurality**: 65.00%
- **borda**: 87.00%
- **ranked_pairs**: 100.00%

---

## Test 4: Random Distance Function Per Voter

### Configuration

- **experiment_id**: 15de281d
- **created_at**: 2025-12-31T12:59:53.469097
- **n_profiles**: 100
- **n_voters**: 100
- **n_candidates**: 5
- **voting_rules**: ['plurality', 'borda', 'ranked_pairs']
- **geometry_method**: uniform
- **geometry_n_dim**: 2
- **geometry_phi**: 0.5
- **manipulation_enabled**: False
- **manipulation_fraction**: 0.2
- **manipulation_strategy**: compromise
- **utility_function**: linear
- **utility_distance_metric**: l2
- **utility_sigma_factor**: 0.5
- **rng_seed**: None
- **epsilon**: 1e-09
- **heterogeneous_distance_enabled**: False

### Key Metrics

- **Rule Disagreement**: 57.00%

#### Average Distance to Ideal

- **plurality**: 0.4591 ± 0.0625
- **borda**: 0.4321 ± 0.0497
- **ranked_pairs**: 0.4327 ± 0.0468

#### Winner Extremism

- **plurality**: 0.2858 ± 0.1348
- **borda**: 0.2215 ± 0.1124
- **ranked_pairs**: 0.2205 ± 0.1073

#### Worst-Off Distance

- **plurality**: 0.8898 ± 0.1354
- **borda**: 0.8268 ± 0.1125
- **ranked_pairs**: 0.8267 ± 0.1083

#### Condorcet Consistency

- **plurality**: 59.00%
- **borda**: 87.00%
- **ranked_pairs**: 100.00%

---

## Test 5: Utility Nonlinearity

### Configuration

- **utility_functions**: ['linear', 'quadratic', 'saturated']

### Key Metrics

- **Rule Disagreement**: 63.00%

#### Average Distance to Ideal

- **plurality**: 0.4496 ± 0.0549
- **borda**: 0.4282 ± 0.0397
- **ranked_pairs**: 0.4269 ± 0.0397

#### Winner Extremism

- **plurality**: 0.2687 ± 0.1163
- **borda**: 0.2177 ± 0.1001
- **ranked_pairs**: 0.2164 ± 0.0993

#### Worst-Off Distance

- **plurality**: 0.8750 ± 0.1236
- **borda**: 0.8267 ± 0.1009
- **ranked_pairs**: 0.8229 ± 0.0972

#### Condorcet Consistency

- **plurality**: 53.00%
- **borda**: 88.00%
- **ranked_pairs**: 100.00%

---

## Test 6: Strategic Misreporting

### Configuration

- **manipulation_fraction**: 0.2

### Key Metrics

- **Rule Disagreement**: 48.00%

#### Average Distance to Ideal

- **plurality**: 0.4501 ± 0.0637
- **borda**: 0.4252 ± 0.0470
- **ranked_pairs**: 0.4245 ± 0.0472

#### Winner Extremism

- **plurality**: 0.2694 ± 0.1269
- **borda**: 0.2119 ± 0.1068
- **ranked_pairs**: 0.2124 ± 0.1074

#### Worst-Off Distance

- **plurality**: 0.8758 ± 0.1223
- **borda**: 0.8230 ± 0.1057
- **ranked_pairs**: 0.8222 ± 0.1023

#### Condorcet Consistency

- **plurality**: 64.00%
- **borda**: 92.00%
- **ranked_pairs**: 100.00%

---

## Test 7: Candidate Clustering

### Configuration

- **candidate_cluster**: [0.7, 1.0]

### Key Metrics

- **Rule Disagreement**: 20.00%

#### Average Distance to Ideal

- **plurality**: 0.5154 ± 0.0453
- **borda**: 0.5152 ± 0.0452
- **ranked_pairs**: 0.5143 ± 0.0444

#### Winner Extremism

- **plurality**: 0.3925 ± 0.0625
- **borda**: 0.3917 ± 0.0612
- **ranked_pairs**: 0.3906 ± 0.0600

#### Worst-Off Distance

- **plurality**: 1.0011 ± 0.0699
- **borda**: 1.0007 ± 0.0703
- **ranked_pairs**: 0.9994 ± 0.0692

#### Condorcet Consistency

- **plurality**: 92.00%
- **borda**: 91.00%
- **ranked_pairs**: 100.00%

---

## Test 8: Thresholds

### Configuration

- **experiment_id**: 30776c97
- **created_at**: 2025-12-31T12:59:56.770402
- **n_profiles**: 100
- **n_voters**: 100
- **n_candidates**: 5
- **voting_rules**: ['approval']
- **geometry_method**: uniform
- **geometry_n_dim**: 2
- **geometry_phi**: 0.5
- **manipulation_enabled**: False
- **manipulation_fraction**: 0.2
- **manipulation_strategy**: compromise
- **utility_function**: linear
- **utility_distance_metric**: l2
- **utility_sigma_factor**: 0.5
- **rng_seed**: None
- **epsilon**: 1e-09
- **heterogeneous_distance_enabled**: False

### Key Metrics

- **Rule Disagreement**: 0.00%

#### Average Distance to Ideal

- **approval**: 0.4273 ± 0.0401

#### Winner Extremism

- **approval**: 0.2182 ± 0.0967

#### Worst-Off Distance

- **approval**: 0.8281 ± 0.0976

#### Condorcet Consistency

- **approval**: 72.00%

---

## Test 9: Dimensionality Sweep

### Configuration

- **dimensions**: [1, 2, 3, 5]

### Key Metrics

- **Rule Disagreement**: 59.00%

#### Average Distance to Ideal

- **plurality**: 0.4810 ± 0.0796
- **borda**: 0.4481 ± 0.0564
- **ranked_pairs**: 0.4528 ± 0.0625

#### Winner Extremism

- **plurality**: 0.3205 ± 0.1446
- **borda**: 0.2565 ± 0.1196
- **ranked_pairs**: 0.2629 ± 0.1262

#### Worst-Off Distance

- **plurality**: 0.9297 ± 0.1466
- **borda**: 0.8685 ± 0.1222
- **ranked_pairs**: 0.8742 ± 0.1298

#### Condorcet Consistency

- **plurality**: 57.00%
- **borda**: 89.00%
- **ranked_pairs**: 100.00%

---

## Summary

### Key Findings

1. **Heterogeneity Impact**: Different distance metrics lead to different outcomes
2. **Rule Sensitivity**: Voting rules respond differently to heterogeneity
3. **Extremism Effects**: Extreme voters using different metrics affect outcomes
