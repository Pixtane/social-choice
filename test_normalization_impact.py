"""
Test whether the first normalization (opposite corner) matters in practice
when using heterogeneous distance metrics.

We'll compare:
1. With first normalization: u = 1 - d/d_max_v (where d_max_v uses opposite corner)
2. Without first normalization: just best-worst scaling on raw distances
"""

import numpy as np
from simulator.utility import UtilityComputer
from simulator.config import UtilityConfig, GeometryConfig, HeterogeneousDistanceConfig
from simulator.heterogeneous_distance import create_strategy


def compute_utilities_with_normalization(distances, voter_positions, voter_metrics, pmin, pmax):
    """Compute utilities WITH first normalization (opposite corner)."""
    from simulator.utility import _compute_opposite_corner, _pointwise_distance

    # Compute opposite corners
    opposite = _compute_opposite_corner(voter_positions, pmin, pmax)

    # Compute per-voter max distances using their own metrics
    d_max_v = np.empty((voter_positions.shape[0],), dtype=float)
    for metric in np.unique(voter_metrics):
        mask = voter_metrics == metric
        if np.any(mask):
            d_max_v[mask] = _pointwise_distance(voter_positions[mask], opposite[mask], metric)

    # Safe divide
    safe_dmax = np.where(d_max_v > 1e-12, d_max_v, 1.0)
    utilities = 1.0 - (distances / safe_dmax[:, np.newaxis])
    utilities = np.clip(utilities, 0.0, 1.0)

    return utilities


def compute_utilities_without_normalization(distances):
    """Compute utilities WITHOUT first normalization (just best-worst scaling)."""
    n_voters = distances.shape[0]
    utilities = np.zeros_like(distances)

    for v in range(n_voters):
        d_min = np.min(distances[v])
        d_max = np.max(distances[v])
        d_range = d_max - d_min

        if d_range > 1e-12:
            # Normalize: best (min distance) = 1, worst (max distance) = 0
            utilities[v] = 1.0 - (distances[v] - d_min) / d_range
        else:
            # All distances equal
            utilities[v] = 1.0

    return utilities


def compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics):
    """Compute distances using each voter's assigned metric."""
    n_voters = voter_positions.shape[0]
    n_candidates = candidate_positions.shape[0]
    distances = np.zeros((n_voters, n_candidates))

    for v in range(n_voters):
        metric = voter_metrics[v]
        for c in range(n_candidates):
            diff = voter_positions[v] - candidate_positions[c]

            if metric == 'l2':
                distances[v, c] = np.linalg.norm(diff, ord=2)
            elif metric == 'l1':
                distances[v, c] = np.linalg.norm(diff, ord=1)
            elif metric == 'linf' or metric == 'chebyshev':
                distances[v, c] = np.linalg.norm(diff, ord=np.inf)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return distances


def test_normalization_impact():
    """Test if first normalization makes a difference."""
    np.random.seed(42)

    # Setup: 10 voters, 5 candidates in 2D unit square
    n_voters = 10
    n_candidates = 5
    n_dim = 2
    pmin, pmax = 0.0, 1.0

    # Generate positions
    voter_positions = np.random.uniform(pmin, pmax, (n_voters, n_dim))
    candidate_positions = np.random.uniform(pmin, pmax, (n_candidates, n_dim))

    # Assign heterogeneous metrics: mix of L1, L2, L∞
    voter_metrics = np.array(['l2'] * 3 + ['l1'] * 3 + ['linf'] * 4)
    np.random.shuffle(voter_metrics)

    print("=" * 80)
    print("TESTING: Does first normalization (opposite corner) matter?")
    print("=" * 80)
    print(f"\nVoters: {n_voters}, Candidates: {n_candidates}, Dimensions: {n_dim}")
    print(f"Metric distribution: L2={np.sum(voter_metrics=='l2')}, "
          f"L1={np.sum(voter_metrics=='l1')}, Linf={np.sum(voter_metrics=='linf')}")

    # Compute distances
    distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)

    # Method 1: WITH first normalization
    utilities_with = compute_utilities_with_normalization(
        distances, voter_positions, voter_metrics, pmin, pmax
    )

    # Method 2: WITHOUT first normalization (just best-worst)
    utilities_without = compute_utilities_without_normalization(distances)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: Utilities WITH vs WITHOUT first normalization")
    print("=" * 80)

    # Show differences
    diff = utilities_with - utilities_without
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    print(f"\nMaximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Show detailed comparison for a few voters
    print("\n" + "-" * 80)
    print("DETAILED COMPARISON (first 3 voters):")
    print("-" * 80)

    for v in range(min(3, n_voters)):
        metric = voter_metrics[v]
        print(f"\nVoter {v} (metric: {metric}):")
        print(f"  Distances to candidates: {distances[v]}")
        print(f"  Utilities WITH normalization:  {utilities_with[v]}")
        print(f"  Utilities WITHOUT normalization: {utilities_without[v]}")
        print(f"  Difference: {diff[v]}")

        # Show rankings
        ranking_with = np.argsort(-utilities_with[v])
        ranking_without = np.argsort(-utilities_without[v])
        print(f"  Ranking WITH:  {ranking_with}")
        print(f"  Ranking WITHOUT: {ranking_without}")
        if not np.array_equal(ranking_with, ranking_without):
            print(f"  WARNING: RANKINGS DIFFER!")

    # Check if any rankings changed
    rankings_with = np.argsort(-utilities_with, axis=1)
    rankings_without = np.argsort(-utilities_without, axis=1)
    rankings_differ = not np.array_equal(rankings_with, rankings_without)

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Rankings differ: {rankings_differ}")
    print(f"Max utility difference: {max_diff:.6f}")
    print(f"Mean utility difference: {mean_diff:.6f}")

    if max_diff < 1e-6:
        print("\nResult: First normalization makes NO PRACTICAL DIFFERENCE")
        print("   (utilities are essentially identical)")
    elif rankings_differ:
        print("\nResult: First normalization AFFECTS RANKINGS")
        print("   (this could change voting outcomes)")
    else:
        print("\nResult: First normalization changes utility VALUES")
        print("   (but rankings remain the same)")

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rankings_differ': rankings_differ,
        'utilities_with': utilities_with,
        'utilities_without': utilities_without
    }


def test_extreme_cases():
    """Test extreme cases where normalization might matter more."""
    print("\n\n" + "=" * 80)
    print("TESTING EXTREME CASES")
    print("=" * 80)

    np.random.seed(123)
    n_voters = 20
    n_candidates = 10
    n_dim = 3
    pmin, pmax = -1.0, 1.0

    # Case 1: Voters at corners, candidates clustered
    print("\nCase 1: Voters at corners, candidates clustered at center")
    voter_positions = np.random.choice([pmin, pmax], size=(n_voters, n_dim))
    candidate_positions = np.random.normal(0, 0.1, (n_candidates, n_dim))
    candidate_positions = np.clip(candidate_positions, pmin, pmax)

    voter_metrics = np.array(['l2'] * 7 + ['l1'] * 7 + ['linf'] * 6)
    np.random.shuffle(voter_metrics)

    distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)
    utilities_with = compute_utilities_with_normalization(
        distances, voter_positions, voter_metrics, pmin, pmax
    )
    utilities_without = compute_utilities_without_normalization(distances)

    diff = utilities_with - utilities_without
    max_diff = np.max(np.abs(diff))
    rankings_with = np.argsort(-utilities_with, axis=1)
    rankings_without = np.argsort(-utilities_without, axis=1)
    rankings_differ = not np.array_equal(rankings_with, rankings_without)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Rankings differ: {rankings_differ}")

    # Case 2: Voters clustered, candidates spread out
    print("\nCase 2: Voters clustered at center, candidates spread out")
    voter_positions = np.random.normal(0, 0.1, (n_voters, n_dim))
    voter_positions = np.clip(voter_positions, pmin, pmax)
    candidate_positions = np.random.uniform(pmin, pmax, (n_candidates, n_dim))

    distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)
    utilities_with = compute_utilities_with_normalization(
        distances, voter_positions, voter_metrics, pmin, pmax
    )
    utilities_without = compute_utilities_without_normalization(distances)

    diff = utilities_with - utilities_without
    max_diff = np.max(np.abs(diff))
    rankings_with = np.argsort(-utilities_with, axis=1)
    rankings_without = np.argsort(-utilities_without, axis=1)
    rankings_differ = not np.array_equal(rankings_with, rankings_without)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Rankings differ: {rankings_differ}")

    # Case 3: High dimensions
    print("\nCase 3: High dimensions (10D)")
    n_dim_high = 10
    voter_positions = np.random.uniform(pmin, pmax, (n_voters, n_dim_high))
    candidate_positions = np.random.uniform(pmin, pmax, (n_candidates, n_dim_high))

    distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)
    utilities_with = compute_utilities_with_normalization(
        distances, voter_positions, voter_metrics, pmin, pmax
    )
    utilities_without = compute_utilities_without_normalization(distances)

    diff = utilities_with - utilities_without
    max_diff = np.max(np.abs(diff))
    rankings_with = np.argsort(-utilities_with, axis=1)
    rankings_without = np.argsort(-utilities_without, axis=1)
    rankings_differ = not np.array_equal(rankings_with, rankings_without)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Rankings differ: {rankings_differ}")


if __name__ == "__main__":
    results = test_normalization_impact()
    test_extreme_cases()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThe simulation tests whether the first normalization (opposite corner)")
    print("actually produces different results than just best-worst scaling.")
    print("\nIf differences are small and rankings don't change, the first")
    print("normalization may be unnecessary for ordinal voting rules.")
    print("If differences are large or rankings change, it matters!")
