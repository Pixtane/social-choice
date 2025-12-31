"""
Expanded Voting Rules Module

Implements utility-based voting simulation with:
- Configurable distance metrics (L2, L1, cosine)
- Dimension-aware utility functions
- Method-appropriate ballot conversion
- Cardinal and ordinal voting rules

Architecture: Belief space → Distance → Utility → Ballot → Winner
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable, Any

# =============================================================================
# Distance Metrics
# =============================================================================

def compute_distances(voter_positions: np.ndarray, 
                      candidate_positions: np.ndarray, 
                      metric: str = 'l2') -> np.ndarray:
    """
    VECTORIZED distance computation between voters and candidates.
    
    Args:
        voter_positions: (n_voters, n_dim) array of voter positions
        candidate_positions: (n_candidates, n_dim) array of candidate positions
        metric: 'l2' (Euclidean), 'l1' (Manhattan), or 'cosine' (experimental)
    
    Returns:
        (n_voters, n_candidates) distance matrix
    
    Notes:
        - L2 and L1 are standard spatial metrics
        - Cosine is EXPERIMENTAL: assumes directional vectors, not bounded spatial points
          Vectors are normalized to unit length before comparison (belief-direction mode)
    """
    if metric == 'l2':
        # Euclidean: ||v - c||_2
        # Broadcasting: (n_voters, 1, n_dim) - (1, n_candidates, n_dim)
        diff = voter_positions[:, np.newaxis, :] - candidate_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
    
    elif metric == 'l1':
        # Manhattan: sum(|v - c|)
        diff = voter_positions[:, np.newaxis, :] - candidate_positions[np.newaxis, :, :]
        distances = np.sum(np.abs(diff), axis=2)
    
    elif metric == 'cosine':
        # Cosine distance: 1 - cosine_similarity
        # IMPORTANT: Normalize vectors to unit length for proper directional comparison
        # This transforms [0,1]^n spatial coords into belief-direction representation
        
        # Normalize to unit length
        v_norms = np.linalg.norm(voter_positions, axis=1, keepdims=True)
        c_norms = np.linalg.norm(candidate_positions, axis=1, keepdims=True)
        v_normalized = voter_positions / (v_norms + 1e-10)  # (n_voters, n_dim)
        c_normalized = candidate_positions / (c_norms + 1e-10)  # (n_candidates, n_dim)
        
        # Compute cosine similarity
        cos_sim = np.dot(v_normalized, c_normalized.T)  # (n_voters, n_candidates)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Handle numerical errors
        
        # Cosine distance = 1 - cosine_similarity (range [0, 2])
        distances = 1.0 - cos_sim
    
    else:
        raise ValueError(f"Unknown distance metric: {metric}. Use 'l2', 'l1', or 'cosine'.")
    
    return distances


def compute_d_max(n_dim: int, metric: str = 'l2') -> Optional[float]:
    """
    Compute maximum possible distance in unit hypercube [0,1]^n.
    NOT USED for cosine metric (cosine shouldn't normalize with d_max).
    
    Args:
        n_dim: Dimension of the space
        metric: Distance metric
    
    Returns:
        Maximum distance, or None for cosine (not applicable)
    """
    if metric == 'l2':
        return np.sqrt(n_dim)  # Diagonal of unit hypercube
    elif metric == 'l1':
        return float(n_dim)  # Sum of max differences
    elif metric == 'cosine':
        # Return None to signal d_max is not applicable
        # Cosine distance should only use Gaussian utility
        return None
    
    return 1.0


# =============================================================================
# Utility Functions (dimension-aware)
# =============================================================================

def compute_utilities(distances: np.ndarray, 
                      n_dim: int,
                      utility_func: str = 'gaussian',
                      sigma_factor: float = 0.5,
                      d_max: Optional[float] = None,
                      distance_metric: str = 'l2') -> np.ndarray:
    """
    Convert distances to utilities using specified decay function.
    
    Args:
        distances: (n_voters, n_candidates) distance matrix
        n_dim: Dimension of space (for sigma scaling)
        utility_func: 'gaussian', 'quadratic', or 'linear'
        sigma_factor: Sigma = sigma_factor * sqrt(n_dim) for Gaussian
        d_max: Maximum distance for normalization (required for quadratic/linear)
        distance_metric: The distance metric used (for validation)
    
    Returns:
        (n_voters, n_candidates) utility matrix in range [0, 1]
    
    Raises:
        ValueError: If quadratic/linear requested with cosine metric
    """
    # Validate: cosine only works with Gaussian
    if distance_metric == 'cosine' and utility_func in ('quadratic', 'linear'):
        raise ValueError(
            f"Utility function '{utility_func}' is incompatible with cosine distance. "
            "Cosine distance should only use 'gaussian' utility (no d_max normalization)."
        )
    
    if utility_func == 'gaussian':
        # Gaussian: u = exp(-d² / (2 * sigma²))
        # Sigma scales with sqrt(n_dim) to handle distance concentration in high dimensions
        sigma = sigma_factor * np.sqrt(n_dim)
        utilities = np.exp(-distances**2 / (2 * sigma**2))
    
    elif utility_func == 'quadratic':
        # Quadratic: u = max(0, 1 - (d / d_max)²)
        # Always clamp to [0, 1] to prevent negative utilities
        if d_max is None or d_max <= 0:
            raise ValueError("d_max must be positive for quadratic utility")
        utilities = np.maximum(0.0, 1.0 - (distances / d_max)**2)
    
    elif utility_func == 'linear':
        # Linear: u = max(0, 1 - d / d_max)
        if d_max is None or d_max <= 0:
            raise ValueError("d_max must be positive for linear utility")
        utilities = np.maximum(0.0, 1.0 - distances / d_max)
    
    else:
        raise ValueError(f"Unknown utility function: {utility_func}. "
                        "Use 'gaussian', 'quadratic', or 'linear'.")
    
    return utilities


# =============================================================================
# Ballot Conversion Functions
# =============================================================================

def utilities_to_scores(utilities: np.ndarray, score_max: int = 5) -> np.ndarray:
    """
    Convert utilities to score ballots with per-voter normalization.
    
    Each voter normalizes their scores so that:
    - Best candidate gets score_max
    - Worst candidate gets 0
    - Others linearly interpolated
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        score_max: Maximum score value (e.g., 5 for 0-5 scale)
    
    Returns:
        (n_voters, n_candidates) score ballot matrix
    """
    n_voters = utilities.shape[0]
    scores = np.zeros_like(utilities)
    
    for v in range(n_voters):
        u_min = np.min(utilities[v])
        u_max = np.max(utilities[v])
        
        if u_max - u_min < 1e-10:
            # All candidates equally good for this voter
            scores[v] = score_max / 2  # Middle score
        else:
            # Normalize to [0, score_max]
            scores[v] = score_max * (utilities[v] - u_min) / (u_max - u_min)
    
    return scores


def utilities_to_approval(utilities: np.ndarray, 
                          policy: str = 'top_k', 
                          param: float = 0.5) -> np.ndarray:
    """
    Convert utilities to approval ballots using explicit threshold policy.
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        policy: Threshold policy:
            - 'top_k': Approve top k% of candidates (param = k, e.g., 0.5 for top 50%)
                       Uses ceil(param * n_candidates) for consistency
            - 'delta': Approve within param of best (u >= u_best - param)
            - 'mean': Approve above mean utility (param ignored)
            - 'absolute': Approve if u >= param (fixed threshold)
        param: Policy parameter
    
    Returns:
        (n_voters, n_candidates) boolean approval matrix
    
    Notes:
        - With 3 candidates, top_k is coarse: 0.33→1, 0.5→2, 0.67→2, 1.0→3
        - Consider 'delta' or 'mean' for finer control
    """
    n_voters, n_candidates = utilities.shape
    approvals = np.zeros((n_voters, n_candidates), dtype=bool)
    
    for v in range(n_voters):
        u_voter = utilities[v]
        
        if policy == 'top_k':
            # Approve top k% (use ceiling for consistency)
            k_count = int(np.ceil(param * n_candidates))
            k_count = max(1, min(k_count, n_candidates))  # At least 1, at most all
            threshold_idx = np.argsort(u_voter)[::-1][k_count - 1]
            threshold = u_voter[threshold_idx]
            approvals[v] = u_voter >= threshold
        
        elif policy == 'delta':
            # Approve within delta of best
            u_best = np.max(u_voter)
            approvals[v] = u_voter >= (u_best - param)
        
        elif policy == 'mean':
            # Approve above mean utility
            u_mean = np.mean(u_voter)
            approvals[v] = u_voter >= u_mean
        
        elif policy == 'absolute':
            # Fixed threshold
            approvals[v] = u_voter >= param
        
        else:
            raise ValueError(f"Unknown approval policy: {policy}. "
                           "Use 'top_k', 'delta', 'mean', or 'absolute'.")
    
    return approvals


def utilities_to_rankings(utilities: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Convert utilities to preference rankings.
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        epsilon: Tolerance for considering utilities as tied
    
    Returns:
        (n_voters, n_candidates) ranking matrix where ranking[v] is a permutation
        of [0, 1, ..., n_candidates-1] in preference order (best first)
    
    Notes:
        - Epsilon-ties are broken arbitrarily (by index order via argsort stability)
    """
    # argsort in descending order of utility
    rankings = np.argsort(-utilities, axis=1)
    return rankings.astype(np.int8)


def utilities_to_plurality(utilities: np.ndarray) -> np.ndarray:
    """
    Convert utilities to plurality votes (argmax per voter).
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
    
    Returns:
        (n_voters,) array of candidate indices (each voter's top choice)
    """
    return np.argmax(utilities, axis=1)


# =============================================================================
# Cardinal Voting Rules
# =============================================================================

def score_voting(utilities: np.ndarray, score_max: int = 5) -> int:
    """
    Score (Range) voting: Sum normalized scores, highest total wins.
    
    This is the utilitarian baseline - maximizes social utility by definition.
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        score_max: Maximum score for normalization
    
    Returns:
        Winner index (0, 1, or 2 for 3 candidates)
    """
    scores = utilities_to_scores(utilities, score_max)
    totals = np.sum(scores, axis=0)
    return int(np.argmax(totals))


def approval_voting(utilities: np.ndarray, 
                    policy: str = 'top_k', 
                    param: float = 0.5) -> int:
    """
    Approval voting: Count approvals, most approved wins.
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        policy: Approval threshold policy
        param: Policy parameter
    
    Returns:
        Winner index
    """
    approvals = utilities_to_approval(utilities, policy, param)
    counts = np.sum(approvals, axis=0)
    return int(np.argmax(counts))


def plurality_cardinal(utilities: np.ndarray) -> int:
    """
    Plurality voting from utilities: Each voter's top choice gets 1 vote.
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
    
    Returns:
        Winner index
    """
    votes = utilities_to_plurality(utilities)
    n_candidates = utilities.shape[1]
    counts = np.bincount(votes, minlength=n_candidates)
    return int(np.argmax(counts))


def star_voting(utilities: np.ndarray, score_max: int = 5) -> int:
    """
    STAR voting: Score Then Automatic Runoff.
    
    Round 1: Normalize utilities to scores per voter, sum to find top 2
    Round 2: Automatic runoff using ORDINAL preferences
             (who did each voter score higher, not raw utility difference)
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        score_max: Maximum score for normalization
    
    Returns:
        Winner index
    """
    n_voters, n_candidates = utilities.shape
    
    # Round 1: Score phase
    scores = utilities_to_scores(utilities, score_max)
    totals = np.sum(scores, axis=0)
    
    # Find top 2 candidates by total score
    sorted_indices = np.argsort(totals)[::-1]
    finalist_a = sorted_indices[0]
    finalist_b = sorted_indices[1]
    
    # Round 2: Automatic Runoff using ORDINAL comparison
    # For each voter, determine which finalist they scored higher
    a_preferred = 0
    b_preferred = 0
    
    for v in range(n_voters):
        # Compare scores (which reflect ordinal preference)
        score_a = scores[v, finalist_a]
        score_b = scores[v, finalist_b]
        
        if score_a > score_b:
            a_preferred += 1
        elif score_b > score_a:
            b_preferred += 1
        # Ties don't add to either count
    
    if a_preferred > b_preferred:
        return int(finalist_a)
    elif b_preferred > a_preferred:
        return int(finalist_b)
    else:
        # Tie in runoff - use total scores as tiebreaker
        return int(finalist_a if totals[finalist_a] >= totals[finalist_b] else finalist_b)


# =============================================================================
# Ordinal Voting Rules (take pre-computed rankings)
# =============================================================================

def _get_positions_from_rankings(rankings: np.ndarray) -> np.ndarray:
    """
    Convert ranking matrix to position matrix.
    
    Rankings: ranking[v, i] = candidate at position i for voter v
    Positions: position[v, c] = position of candidate c for voter v
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        (n_voters, n_candidates) position matrix
    """
    n_voters, n_candidates = rankings.shape
    positions = np.empty((n_voters, n_candidates), dtype=np.int8)
    
    for v in range(n_voters):
        for pos, candidate in enumerate(rankings[v]):
            positions[v, candidate] = pos
    
    return positions


def borda(rankings: np.ndarray) -> int:
    """
    Borda count: Points by position (n-1 for 1st, n-2 for 2nd, ..., 0 for last).
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    positions = _get_positions_from_rankings(rankings)
    n_candidates = rankings.shape[1]
    
    # Borda score = (n_candidates - 1) - position
    scores = np.sum((n_candidates - 1) - positions, axis=0)
    return int(np.argmax(scores))


def instant_runoff(rankings: np.ndarray) -> int:
    """
    Instant Runoff Voting (IRV): Eliminate candidate with fewest 1st-place votes.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    n_voters, n_candidates = rankings.shape
    remaining = set(range(n_candidates))
    
    while len(remaining) > 1:
        # Count first-place votes among remaining candidates
        counts = np.zeros(n_candidates, dtype=np.int32)
        
        for v in range(n_voters):
            for candidate in rankings[v]:
                if candidate in remaining:
                    counts[candidate] += 1
                    break
        
        # Find loser among remaining
        loser = min(remaining, key=lambda c: counts[c])
        remaining.remove(loser)
    
    return int(remaining.pop())


def copeland(rankings: np.ndarray) -> int:
    """
    Copeland's method: Win pairwise matchup = +1, lose = -1, tie = 0.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    positions = _get_positions_from_rankings(rankings)
    n_voters, n_candidates = rankings.shape
    scores = np.zeros(n_candidates, dtype=np.int32)
    
    for a in range(n_candidates):
        for b in range(a + 1, n_candidates):
            a_wins = np.sum(positions[:, a] < positions[:, b])
            b_wins = n_voters - a_wins
            
            if a_wins > b_wins:
                scores[a] += 1
                scores[b] -= 1
            elif b_wins > a_wins:
                scores[b] += 1
                scores[a] -= 1
            # Tie: no change
    
    return int(np.argmax(scores))


def minimax(rankings: np.ndarray) -> int:
    """
    Minimax method: Minimize worst pairwise defeat margin.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    positions = _get_positions_from_rankings(rankings)
    n_voters, n_candidates = rankings.shape
    worst_defeat = np.zeros(n_candidates, dtype=np.int32)
    
    for a in range(n_candidates):
        for b in range(n_candidates):
            if a != b:
                b_wins = np.sum(positions[:, b] < positions[:, a])
                margin = 2 * b_wins - n_voters
                worst_defeat[a] = max(worst_defeat[a], margin)
    
    return int(np.argmin(worst_defeat))


def plurality_runoff(rankings: np.ndarray) -> int:
    """
    Plurality runoff: Top two from plurality face off in pairwise comparison.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    positions = _get_positions_from_rankings(rankings)
    n_voters, n_candidates = rankings.shape
    
    # Count first-place votes
    first_choices = rankings[:, 0]
    counts = np.bincount(first_choices, minlength=n_candidates)
    
    # Top two
    top_two = np.argsort(counts)[-2:]
    a, b = top_two[0], top_two[1]
    
    # Pairwise runoff
    a_wins = np.sum(positions[:, a] < positions[:, b])
    return int(a if a_wins > n_voters - a_wins else b)


def _condorcet_winner(rankings: np.ndarray) -> Optional[int]:
    """
    Find Condorcet winner if one exists.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index, or None if no Condorcet winner
    """
    positions = _get_positions_from_rankings(rankings)
    n_voters, n_candidates = rankings.shape
    
    for a in range(n_candidates):
        beats_all = True
        for b in range(n_candidates):
            if a != b:
                a_wins = np.sum(positions[:, a] < positions[:, b])
                if a_wins <= n_voters / 2:
                    beats_all = False
                    break
        if beats_all:
            return a
    
    return None


def black(rankings: np.ndarray) -> int:
    """
    Black's method: Condorcet winner if exists, otherwise Borda winner.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Winner index
    """
    condorcet = _condorcet_winner(rankings)
    if condorcet is not None:
        return condorcet
    return borda(rankings)


# =============================================================================
# VSE and Metrics Computation
# =============================================================================

def compute_vse_and_metrics(utilities: np.ndarray, 
                            winner_idx: int) -> Tuple[float, float, float, int]:
    """
    Compute NORMALIZED VSE and enhanced metrics based on true utilities.
    
    Metrics:
    - VSE = (U_elected - U_random) / (U_optimal - U_random)
    - Regret = U_optimal - U_elected
    - Winner rank = social utility rank of winner (0=best, 1=2nd, 2=3rd)
    
    Args:
        utilities: (n_voters, n_candidates) utility matrix
        winner_idx: Index of elected candidate
    
    Returns:
        Tuple of (normalized_vse, raw_social_utility, regret, winner_rank)
    """
    # Social utility = average utility across all voters
    avg_utilities = np.mean(utilities, axis=0)  # Per candidate
    
    u_elected = avg_utilities[winner_idx]  # Social utility of winner
    u_optimal = np.max(avg_utilities)  # Utilitarian optimum
    u_random = np.mean(avg_utilities)  # Random selection baseline
    
    # Compute regret
    regret = u_optimal - u_elected
    
    # Compute winner's rank (0=1st, 1=2nd, 2=3rd)
    sorted_indices = np.argsort(avg_utilities)[::-1]  # Descending order
    winner_rank = int(np.where(sorted_indices == winner_idx)[0][0])
    
    # Compute normalized VSE
    # Handle edge case: all candidates equally good
    if abs(u_optimal - u_random) < 1e-10:
        normalized_vse = 1.0 if abs(regret) < 1e-10 else 0.0
    else:
        normalized_vse = (u_elected - u_random) / (u_optimal - u_random)
    
    return normalized_vse, u_elected, regret, winner_rank


def compute_margins_from_rankings(rankings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute pairwise margins for Condorcet cycle detection.
    
    Args:
        rankings: (n_voters, n_candidates) ranking matrix
    
    Returns:
        Tuple of (m_ab, m_bc, m_ca) normalized margins
        
    Note:
        Only works for 3 candidates.
    """
    positions = _get_positions_from_rankings(rankings)
    n_voters = rankings.shape[0]
    
    a_wins_ab = np.sum(positions[:, 0] < positions[:, 1])
    m_ab = (2 * a_wins_ab - n_voters) / n_voters
    
    b_wins_bc = np.sum(positions[:, 1] < positions[:, 2])
    m_bc = (2 * b_wins_bc - n_voters) / n_voters
    
    c_wins_ca = np.sum(positions[:, 2] < positions[:, 0])
    m_ca = (2 * c_wins_ca - n_voters) / n_voters
    
    return m_ab, m_bc, m_ca


def is_cycle(margins: Tuple[float, float, float]) -> bool:
    """
    Check if pairwise margins indicate a Condorcet cycle.
    
    A cycle exists when all three margins have the same sign.
    
    Args:
        margins: Tuple of (m_ab, m_bc, m_ca)
    
    Returns:
        True if cycle exists
    """
    m_ab, m_bc, m_ca = margins
    
    if m_ab == 0 or m_bc == 0 or m_ca == 0:
        return False
    
    return (m_ab > 0 and m_bc > 0 and m_ca > 0) or \
           (m_ab < 0 and m_bc < 0 and m_ca < 0)


def classify_cycle_type(margins: Tuple[float, float, float]) -> Optional[int]:
    """
    Classify cycle into Type 1 (A>B>C>A) or Type 2 (B>A>C>B).
    
    Args:
        margins: Tuple of (m_ab, m_bc, m_ca)
    
    Returns:
        1 for Type 1, 2 for Type 2, None if no cycle
    """
    m_ab, m_bc, m_ca = margins
    
    if m_ab > 0 and m_bc > 0 and m_ca > 0:
        return 1
    elif m_ab < 0 and m_bc < 0 and m_ca < 0:
        return 2
    return None


# =============================================================================
# Voting Rules Registry
# =============================================================================

VOTING_RULES: Dict[str, Dict[str, Any]] = {
    # Cardinal rules (take utilities)
    'score': {
        'func': score_voting,
        'type': 'cardinal',
        'requires': 'utilities',
        'description': 'Score/Range voting - utilitarian baseline',
    },
    'approval': {
        'func': approval_voting,
        'type': 'cardinal',
        'requires': 'utilities',
        'description': 'Approval voting with configurable threshold',
    },
    'plurality': {
        'func': plurality_cardinal,
        'type': 'cardinal',
        'requires': 'utilities',
        'description': 'Plurality - each voter\'s top choice',
    },
    'star': {
        'func': star_voting,
        'type': 'cardinal',
        'requires': 'utilities',
        'description': 'STAR - Score Then Automatic Runoff',
    },
    
    # Ordinal rules (take rankings)
    'borda': {
        'func': borda,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Borda count - positional scoring',
    },
    'irv': {
        'func': instant_runoff,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Instant Runoff Voting',
    },
    'copeland': {
        'func': copeland,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Copeland - pairwise win/loss scores',
    },
    'minimax': {
        'func': minimax,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Minimax - minimize worst defeat',
    },
    'plurality_runoff': {
        'func': plurality_runoff,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Plurality runoff - top two face off',
    },
    'black': {
        'func': black,
        'type': 'ordinal',
        'requires': 'rankings',
        'description': 'Black - Condorcet then Borda',
    },
}

# Convenience list of rule names
CARDINAL_RULES = [name for name, info in VOTING_RULES.items() if info['type'] == 'cardinal']
ORDINAL_RULES = [name for name, info in VOTING_RULES.items() if info['type'] == 'ordinal']




