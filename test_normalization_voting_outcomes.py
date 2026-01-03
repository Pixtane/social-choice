"""
Test if first normalization affects actual voting outcomes.
We'll test with cardinal voting rules that use utility values directly.
"""

import numpy as np
from simulator.utility import UtilityComputer
from simulator.config import UtilityConfig, GeometryConfig, HeterogeneousDistanceConfig
from simulator.heterogeneous_distance import create_strategy


def compute_utilities_with_normalization(distances, voter_positions, voter_metrics, pmin, pmax):
    """Compute utilities WITH first normalization (opposite corner)."""
    from simulator.utility import _compute_opposite_corner, _pointwise_distance

    opposite = _compute_opposite_corner(voter_positions, pmin, pmax)
    d_max_v = np.empty((voter_positions.shape[0],), dtype=float)
    for metric in np.unique(voter_metrics):
        mask = voter_metrics == metric
        if np.any(mask):
            d_max_v[mask] = _pointwise_distance(voter_positions[mask], opposite[mask], metric)

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
            utilities[v] = 1.0 - (distances[v] - d_min) / d_range
        else:
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


def score_voting(utilities, max_score=5):
    """Score voting: each voter gives scores 0-max_score based on utilities."""
    n_voters, n_candidates = utilities.shape
    scores = np.zeros((n_voters, n_candidates))

    for v in range(n_voters):
        # Convert utilities to scores: best=max_score, worst=0
        u_min = np.min(utilities[v])
        u_max = np.max(utilities[v])
        u_range = u_max - u_min

        if u_range > 1e-12:
            scores[v] = max_score * (utilities[v] - u_min) / u_range
        else:
            scores[v] = max_score / 2  # All equal

    # Sum scores across voters
    total_scores = np.sum(scores, axis=0)
    winner = np.argmax(total_scores)
    return winner, total_scores


def approval_voting(utilities, threshold=0.5):
    """Approval voting: approve candidates above threshold."""
    approvals = (utilities >= threshold).astype(int)
    vote_counts = np.sum(approvals, axis=0)
    winner = np.argmax(vote_counts)
    return winner, vote_counts


def test_voting_outcomes():
    """Test if normalization affects voting outcomes."""
    np.random.seed(42)

    n_voters = 50
    n_candidates = 5
    n_dim = 2
    pmin, pmax = 0.0, 1.0
    n_trials = 100

    print("=" * 80)
    print("TESTING: Does first normalization affect voting outcomes?")
    print("=" * 80)
    print(f"Voters: {n_voters}, Candidates: {n_candidates}, Trials: {n_trials}")

    score_voting_differences = 0
    approval_voting_differences = 0

    for trial in range(n_trials):
        # Generate random positions
        voter_positions = np.random.uniform(pmin, pmax, (n_voters, n_dim))
        candidate_positions = np.random.uniform(pmin, pmax, (n_candidates, n_dim))

        # Assign heterogeneous metrics
        voter_metrics = np.random.choice(['l2', 'l1', 'linf'], size=n_voters)

        # Compute distances
        distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)

        # Compute utilities both ways
        utilities_with = compute_utilities_with_normalization(
            distances, voter_positions, voter_metrics, pmin, pmax
        )
        utilities_without = compute_utilities_without_normalization(distances)

        # Test Score Voting
        winner_with, scores_with = score_voting(utilities_with)
        winner_without, scores_without = score_voting(utilities_without)
        if winner_with != winner_without:
            score_voting_differences += 1

        # Test Approval Voting (using threshold of 0.5)
        winner_with_approval, _ = approval_voting(utilities_with, threshold=0.5)
        winner_without_approval, _ = approval_voting(utilities_without, threshold=0.5)
        if winner_with_approval != winner_without_approval:
            approval_voting_differences += 1

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Score Voting: {score_voting_differences}/{n_trials} trials with different winners ({100*score_voting_differences/n_trials:.1f}%)")
    print(f"Approval Voting: {approval_voting_differences}/{n_trials} trials with different winners ({100*approval_voting_differences/n_trials:.1f}%)")

    if score_voting_differences == 0 and approval_voting_differences == 0:
        print("\nCONCLUSION: First normalization does NOT affect voting outcomes")
        print("   (even with cardinal voting rules)")
    else:
        print("\nCONCLUSION: First normalization DOES affect voting outcomes")
        print("   (this matters for cardinal voting rules!)")

    return {
        'score_differences': score_voting_differences,
        'approval_differences': approval_voting_differences,
        'n_trials': n_trials
    }


def test_utility_magnitudes():
    """Test if utility magnitudes differ significantly."""
    np.random.seed(42)

    n_voters = 20
    n_candidates = 5
    n_dim = 2
    pmin, pmax = 0.0, 1.0
    n_trials = 50

    print("\n\n" + "=" * 80)
    print("TESTING: Utility magnitude differences")
    print("=" * 80)

    max_diffs = []
    mean_diffs = []

    for trial in range(n_trials):
        voter_positions = np.random.uniform(pmin, pmax, (n_voters, n_dim))
        candidate_positions = np.random.uniform(pmin, pmax, (n_candidates, n_dim))
        voter_metrics = np.random.choice(['l2', 'l1', 'linf'], size=n_voters)

        distances = compute_distances_heterogeneous(voter_positions, candidate_positions, voter_metrics)
        utilities_with = compute_utilities_with_normalization(
            distances, voter_positions, voter_metrics, pmin, pmax
        )
        utilities_without = compute_utilities_without_normalization(distances)

        diff = utilities_with - utilities_without
        max_diffs.append(np.max(np.abs(diff)))
        mean_diffs.append(np.mean(np.abs(diff)))

    print(f"\nAcross {n_trials} trials:")
    print(f"  Max difference: mean={np.mean(max_diffs):.4f}, std={np.std(max_diffs):.4f}")
    print(f"  Mean difference: mean={np.mean(mean_diffs):.4f}, std={np.std(mean_diffs):.4f}")
    print(f"  Largest max difference: {np.max(max_diffs):.4f}")
    print(f"  Smallest max difference: {np.min(max_diffs):.4f}")


if __name__ == "__main__":
    results = test_voting_outcomes()
    test_utility_magnitudes()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nThe first normalization (opposite corner) changes utility VALUES")
    print("but may or may not change voting OUTCOMES depending on:")
    print("  1. The voting rule (ordinal vs cardinal)")
    print("  2. The specific profile")
    print("  3. Whether utilities are used directly or just rankings")
