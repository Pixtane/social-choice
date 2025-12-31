# FULL VARIABLE SPACE - DIMENSIONAL VOTING SIMULATION

## LAYER 0 - Structural size parameters

These define the _scale_ of the election.

- **Number of voters**
- **Number of candidates**
- **Issue space dimension**
- **Number of elections** (Monte Carlo runs)

## LAYER 1 - Issue space geometry

Defines _what kind of world voters live in_.

- **Space type**

- Euclidean
- Bounded cube / sphere
- Manifold (torus, simplex)
- Graph / discrete topology

- **Axis meaning**

- independent issues
- correlated issues
- rotated axes

- **Axis scaling**

- uniform
- weighted per dimension
- anisotropic scaling

## LAYER 2 - Voter ideal-point generation

Defines _where voters are_.

- **Distribution family**

- uniform
- Gaussian
- mixture of Gaussians
- clustered
- ring / shell
- heavy-tailed

- **Correlation structure**

- independent dimensions
- correlated dimensions
- ideological line + noise

- **Radial distribution**

- center-heavy
- edge-heavy
- bimodal

- **Outliers**

- probability of extremists

## LAYER 3 - Candidate position generation

Defines _what options exist_.

- **Candidate distribution**

- same as voters
- strategic placement
- clustered elites
- uniform random

- **Candidate count regime**

- fixed
- Poisson
- endogenous (entry models)

- **Position constraints**

- must lie in convex hull
- must lie on axes
- must be distinct

## LAYER 4 - Distance (geometry) functions

Defines _how differences are measured_.

- **Metric family**

- L1
- L2
- L∞
- cosine
- weighted
- Mahalanobis
- graph distance

- **Heterogeneity**

- per voter
- per group
- stochastic assignment

- **Regime switching**

- radius-based
- candidate-based
- issue-based

- **Non-compensability**

- veto dimensions
- max-issue rules

## LAYER 5 - Utility functions

Defines _how distance feels_.

- **Utility mapping**

- linear
- quadratic
- exponential
- saturating
- thresholded
- sigmoid

- **Utility bounds**

- bounded / unbounded
- symmetric / asymmetric

- **Heterogeneity**

- per voter
- per distance regime

- **Salience scaling**

- amplify certain distances
- diminishing returns

## LAYER 6 - Preference formation

Defines _how utilities become rankings_.

- **Strict vs weak preferences**

- ties allowed or not

- **Noise in evaluation**

- random utility noise
- perception error

- **Heuristics**

- attractiveness override
- party labels
- incumbency bias

- **Strategic misreporting**

- truncation
- exaggeration
- bullet voting

## LAYER 7 - Acceptability / thresholds

Defines _lesser-evil logic_.

- **Acceptability threshold**

- fixed
- voter-dependent
- distance-dependent

- **Rejection behavior**

- abstention
- equal ranking
- protest vote

## LAYER 8 - Voting rule

Defines _how preferences are aggregated_.

- **Voting system**

- plurality
- IRV / STV
- approval
- score
- Borda
- Condorcet variants
- custom rules

- **Tie-breaking rules**

- random
- lexicographic
- geometric

- **Ballot type**

- ordinal
- cardinal
- approval
- mixed

## LAYER 9 - Aggregation mechanics

Defines _how rule is applied_.

- **Pairwise margin definition**

- simple majority
- weighted
- turnout-adjusted

- **Elimination order**

- fixed
- stochastic

- **Quota definitions** (STV)

## LAYER 10 - Post-processing / outcome interpretation

Defines _what "winning" means_.

- **Single winner vs set**
- **Multi-round selection**
- **Probabilistic outcomes**

## LAYER 11 - Metrics (what you record)

Defines _what you study_.

- **Efficiency**

- welfare
- regret
- average distance

- **Fairness**

- worst-off (distance / utility)
- inequality
- acceptability

- **Coherence**

- Condorcet consistency
- Smith inclusion
- cycle rate

- **Robustness**

- noise sensitivity
- volatility

- **Manipulability**

- local / global

## LAYER 12 - Temporal & cultural variation (optional)

Defines _change over time_.

- **Time evolution**

- voter drift
- issue salience drift

- **Cultural parameters**

- threshold norms
- polarization norms

## Minimal "serious" subset (recommended)

If you want _power without explosion_, keep:

- voter & candidate distributions
- distance family (+ heterogeneity)
- utility mapping
- voting rule
- 5 metrics

Everything else is secondary.

## One-sentence compression

A spatial voting simulation is a pipeline of choices about geometry, psychology, aggregation, and measurement - each layer introduces its own failure modes.
