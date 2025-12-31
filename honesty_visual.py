"""
Honesty & Strategic Voting Visualizer

A comprehensive visualization tool for comparing voting rules with:
- Multiple voting methods including niche ones like STAR, Approval, Range, Schulze
- Adjustable honesty score (0% = fully strategic, 100% = fully honest)
- Draggable candidate positions in 2D policy space
- Yee diagram visualization showing which candidate wins at each location
- Multiple profile generation methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from matplotlib.backend_bases import MouseButton

# =============================================================================
# Constants
# =============================================================================

CANDIDATES = ['A', 'B', 'C']
N_CANDIDATES = 3

# Candidate colors
CANDIDATE_COLORS = {
    0: '#e74c3c',  # A - Red
    1: '#3498db',  # B - Blue
    2: '#2ecc71',  # C - Green
    'A': '#e74c3c',
    'B': '#3498db',
    'C': '#2ecc71',
    None: '#95a5a6',  # Gray for ties
}

# Default candidate positions in 2D space [0, 1] x [0, 1]
DEFAULT_CANDIDATE_POSITIONS = np.array([
    [0.25, 0.35],  # A - bottom left
    [0.75, 0.35],  # B - bottom right
    [0.50, 0.80],  # C - top center
], dtype=np.float64)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_distances(voter_positions, candidate_positions):
    """
    Compute distances from each voter to each candidate.
    
    Parameters:
        voter_positions: (n_voters, 2) array
        candidate_positions: (n_candidates, 2) array
    
    Returns:
        distances: (n_voters, n_candidates) array
    """
    n_voters = voter_positions.shape[0]
    n_candidates = candidate_positions.shape[0]
    
    distances = np.zeros((n_voters, n_candidates), dtype=np.float64)
    for c in range(n_candidates):
        diff = voter_positions - candidate_positions[c]
        distances[:, c] = np.sqrt(np.sum(diff**2, axis=1))
    
    return distances


def compute_utilities(voter_positions, candidate_positions):
    """
    Compute utilities for each voter-candidate pair based on spatial distance.
    Utility = 1 - normalized_distance (closer = higher utility).
    
    Returns:
        utilities: (n_voters, n_candidates) array in [0, 1]
    """
    distances = compute_distances(voter_positions, candidate_positions)
    
    # Max possible distance in unit square is sqrt(2)
    max_dist = np.sqrt(2.0)
    utilities = 1.0 - (distances / max_dist)
    utilities = np.clip(utilities, 0.0, 1.0)
    
    return utilities


def utility_to_ranking(utilities):
    """
    Convert utilities to rankings (0 = best, 2 = worst).
    
    Returns:
        rankings: (n_voters, n_candidates) array of rankings
    """
    n_voters = utilities.shape[0]
    rankings = np.zeros_like(utilities, dtype=np.int32)
    
    for v in range(n_voters):
        # argsort gives indices that would sort the array in ascending order
        # We want descending (highest utility first), so we negate
        sorted_indices = np.argsort(-utilities[v])
        for rank, idx in enumerate(sorted_indices):
            rankings[v, idx] = rank
    
    return rankings


def rankings_to_profile(rankings):
    """
    Convert rankings matrix to profile format (list of permutations).
    
    rankings: (n_voters, n_candidates) where rankings[v, c] = rank of candidate c for voter v
    
    Returns:
        profile: (n_voters, n_candidates) where profile[v] = permutation of candidates by preference
    """
    n_voters, n_candidates = rankings.shape
    profile = np.zeros_like(rankings, dtype=np.int8)
    
    for v in range(n_voters):
        # Sort candidates by their ranking (lowest rank = most preferred)
        profile[v] = np.argsort(rankings[v])
    
    return profile


def _get_positions(profile):
    """Get position matrix: positions[i,c] = position of candidate c in voter i's ranking."""
    n = profile.shape[0]
    positions = np.empty((n, 3), dtype=np.int8)
    for c in range(3):
        positions[:, c] = np.where(profile == c)[1].reshape(n)
    return positions


# =============================================================================
# Voter Generation Methods
# =============================================================================

def generate_voters_uniform(n_voters, seed=None):
    """Generate uniformly distributed voters in [0, 1] x [0, 1]."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.random((n_voters, 2))


def generate_voters_gaussian(n_voters, center=(0.5, 0.5), spread=0.25, seed=None):
    """Generate Gaussian distributed voters around a center point."""
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.random.normal(loc=center, scale=spread, size=(n_voters, 2))
    return np.clip(positions, 0.0, 1.0)


def generate_voters_clustered(n_voters, candidate_positions, seed=None):
    """Generate voters clustered near candidates."""
    if seed is not None:
        np.random.seed(seed)
    
    n_candidates = candidate_positions.shape[0]
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    for v in range(n_voters):
        c = np.random.randint(0, n_candidates)
        spread = 0.12
        pos = np.random.normal(loc=candidate_positions[c], scale=spread)
        positions[v] = np.clip(pos, 0.0, 1.0)
    
    return positions


def generate_voters_polarized(n_voters, candidate_positions, seed=None):
    """
    Generate a polarized electorate with two major factions supporting A and B,
    with C as a potential compromise candidate.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    faction_a_size = int(n_voters * np.random.uniform(0.35, 0.50))
    faction_b_size = int(n_voters * np.random.uniform(0.35, 0.50))
    faction_c_size = n_voters - faction_a_size - faction_b_size
    
    idx = 0
    
    # Faction A
    for _ in range(faction_a_size):
        pos = np.random.normal(loc=candidate_positions[0], scale=0.10)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    # Faction B
    for _ in range(faction_b_size):
        pos = np.random.normal(loc=candidate_positions[1], scale=0.10)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    # Faction C / swing voters
    for _ in range(faction_c_size):
        pos = np.random.normal(loc=candidate_positions[2], scale=0.12)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    np.random.shuffle(positions)
    return positions


def generate_voters_weighted(n_voters, candidate_positions, seed=None):
    """Generate voters with random Dirichlet-weighted distribution toward candidates."""
    if seed is not None:
        np.random.seed(seed)
    
    n_candidates = candidate_positions.shape[0]
    weights = np.random.dirichlet(np.ones(n_candidates) * 2)
    
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    for v in range(n_voters):
        c = np.random.choice(n_candidates, p=weights)
        spread = 0.15
        pos = np.random.normal(loc=candidate_positions[c], scale=spread)
        positions[v] = np.clip(pos, 0.0, 1.0)
    
    return positions


GENERATION_METHODS = {
    'uniform': generate_voters_uniform,
    'gaussian': generate_voters_gaussian,
    'clustered': generate_voters_clustered,
    'polarized': generate_voters_polarized,
    'weighted': generate_voters_weighted,
}


# =============================================================================
# Honest vs Strategic Voting
# =============================================================================

def generate_honest_profile(utilities):
    """
    Generate honest rankings from utilities.
    Voters rank candidates by true preference (highest utility first).
    """
    rankings = utility_to_ranking(utilities)
    return rankings_to_profile(rankings)


def generate_honest_scores(utilities, max_score=5):
    """
    Generate honest scores for score-based voting.
    Maps utility [0, 1] to score [0, max_score].
    """
    scores = np.round(utilities * max_score).astype(np.int32)
    return np.clip(scores, 0, max_score)


def generate_strategic_profile(utilities):
    """
    Generate strategic rankings.
    For ranked voting, voters might try to bury compromise candidates.
    This is a simplified model - voters just vote honestly in rankings.
    """
    # For ranked choice, strategy is complex - we use honest rankings
    # but could implement burial or compromise strategies
    return generate_honest_profile(utilities)


def generate_strategic_scores(utilities, max_score=5):
    """
    Generate strategic scores for score-based voting.
    Voters use min-max strategy: max score to favorite, min to least favorite.
    """
    n_voters, n_candidates = utilities.shape
    scores = np.zeros((n_voters, n_candidates), dtype=np.int32)
    
    for v in range(n_voters):
        u = utilities[v]
        sorted_idx = np.argsort(u)[::-1]  # Best to worst
        
        scores[v, sorted_idx[0]] = max_score
        scores[v, sorted_idx[-1]] = 0
        
        # Middle candidates get proportional scores
        for i in range(1, n_candidates - 1):
            c = sorted_idx[i]
            u_range = u[sorted_idx[0]] - u[sorted_idx[-1]]
            if u_range > 0:
                rel_u = (u[c] - u[sorted_idx[-1]]) / u_range
                scores[v, c] = int(np.round(rel_u * max_score))
            else:
                scores[v, c] = max_score // 2
    
    return scores


def generate_strategic_approval(utilities, threshold='median'):
    """
    Generate strategic approval ballots.
    Voters approve candidates above their utility threshold.
    """
    n_voters, n_candidates = utilities.shape
    approvals = np.zeros((n_voters, n_candidates), dtype=np.int32)
    
    for v in range(n_voters):
        u = utilities[v]
        if threshold == 'median':
            # Approve top half
            thresh = np.median(u)
        elif threshold == 'mean':
            thresh = np.mean(u)
        else:
            thresh = 0.5
        
        # Strategic: always approve best, never approve worst
        sorted_idx = np.argsort(u)[::-1]
        approvals[v, sorted_idx[0]] = 1  # Always approve best
        approvals[v, sorted_idx[-1]] = 0  # Never approve worst
        
        # Middle candidates: approve if above threshold
        for i in range(1, n_candidates - 1):
            c = sorted_idx[i]
            approvals[v, c] = 1 if u[c] >= thresh else 0
    
    return approvals


def generate_mixed_profile(utilities, honesty_pct, profile_type='ranking'):
    """
    Generate a mixed profile with some honest and some strategic voters.
    
    Parameters:
        utilities: (n_voters, n_candidates) utility matrix
        honesty_pct: 0-100, percentage of honest voters
        profile_type: 'ranking', 'score', or 'approval'
    
    Returns:
        profile based on type
    """
    n_voters = utilities.shape[0]
    n_honest = int(n_voters * honesty_pct / 100.0)
    
    honest_mask = np.zeros(n_voters, dtype=bool)
    if n_honest > 0 and n_honest < n_voters:
        honest_indices = np.random.choice(n_voters, size=n_honest, replace=False)
        honest_mask[honest_indices] = True
    elif n_honest >= n_voters:
        honest_mask[:] = True
    
    if profile_type == 'ranking':
        honest = generate_honest_profile(utilities)
        strategic = generate_strategic_profile(utilities)
        return np.where(honest_mask[:, np.newaxis], honest, strategic)
    
    elif profile_type == 'score':
        honest = generate_honest_scores(utilities)
        strategic = generate_strategic_scores(utilities)
        return np.where(honest_mask[:, np.newaxis], honest, strategic)
    
    elif profile_type == 'approval':
        # For approval, honest voters use mean threshold, strategic use median
        honest = (utilities > np.mean(utilities, axis=1, keepdims=True)).astype(np.int32)
        strategic = generate_strategic_approval(utilities)
        return np.where(honest_mask[:, np.newaxis], honest, strategic)
    
    return generate_honest_profile(utilities)


# =============================================================================
# Voting Rules
# =============================================================================

def plurality(profile, **kwargs):
    """Each voter's top choice gets 1 point."""
    first_choices = profile[:, 0]
    counts = np.bincount(first_choices.astype(int), minlength=3)
    return CANDIDATES[np.argmax(counts)]


def borda(profile, **kwargs):
    """Points by position: 1st=2pts, 2nd=1pt, 3rd=0pts."""
    positions = _get_positions(profile)
    scores = np.sum(2 - positions, axis=0)
    return CANDIDATES[np.argmax(scores)]


def antiplurality(profile, **kwargs):
    """Each voter's last choice gets -1 point (least last-places wins)."""
    last_choices = profile[:, 2]
    counts = np.bincount(last_choices.astype(int), minlength=3)
    return CANDIDATES[np.argmin(counts)]


def instant_runoff(profile, **kwargs):
    """IRV: Eliminate candidate with fewest first-place votes, repeat until winner."""
    remaining = {0, 1, 2}
    profile_copy = profile.copy()
    
    while len(remaining) > 1:
        counts = np.zeros(3, dtype=np.int32)
        for v in range(profile_copy.shape[0]):
            for c in profile_copy[v]:
                if c in remaining:
                    counts[c] += 1
                    break
        
        loser = min(remaining, key=lambda c: counts[c])
        remaining.remove(loser)
    
    return CANDIDATES[remaining.pop()]


def copeland(profile, **kwargs):
    """Win pairwise matchup = +1, lose = -1, tie = 0."""
    positions = _get_positions(profile)
    n = profile.shape[0]
    scores = np.zeros(3, dtype=np.int32)
    
    for a in range(3):
        for b in range(a + 1, 3):
            a_wins = np.sum(positions[:, a] < positions[:, b])
            if a_wins > n / 2:
                scores[a] += 1
                scores[b] -= 1
            elif a_wins < n / 2:
                scores[b] += 1
                scores[a] -= 1
    
    return CANDIDATES[np.argmax(scores)]


def minimax(profile, **kwargs):
    """Minimize worst pairwise defeat margin."""
    positions = _get_positions(profile)
    n = profile.shape[0]
    worst_defeat = np.zeros(3, dtype=np.int32)
    
    for a in range(3):
        for b in range(3):
            if a != b:
                b_wins = np.sum(positions[:, b] < positions[:, a])
                margin = 2 * b_wins - n
                worst_defeat[a] = max(worst_defeat[a], margin)
    
    return CANDIDATES[np.argmin(worst_defeat)]


def plurality_runoff(profile, **kwargs):
    """Top two from plurality face off in pairwise comparison."""
    positions = _get_positions(profile)
    first_choices = profile[:, 0]
    counts = np.bincount(first_choices.astype(int), minlength=3)
    top_two = np.argsort(counts)[-2:]
    
    a, b = top_two
    a_wins = np.sum(positions[:, a] < positions[:, b])
    return CANDIDATES[a if a_wins >= profile.shape[0] - a_wins else b]


def condorcet(profile, **kwargs):
    """Return Condorcet winner if exists, else None."""
    positions = _get_positions(profile)
    n = profile.shape[0]
    
    for a in range(3):
        beats_all = True
        for b in range(3):
            if a != b:
                a_wins = np.sum(positions[:, a] < positions[:, b])
                if a_wins <= n / 2:
                    beats_all = False
                    break
        if beats_all:
            return CANDIDATES[a]
    return None


def black(profile, **kwargs):
    """Condorcet winner if exists, otherwise Borda winner."""
    winner = condorcet(profile)
    return winner if winner else borda(profile)


# Score-based voting rules

def score_voting(utilities, honesty_pct=100, max_score=5, **kwargs):
    """
    Score/Range voting: Candidates get sum of scores.
    Uses utilities and honesty to generate scores.
    """
    scores = generate_mixed_profile(utilities, honesty_pct, 'score')
    totals = np.sum(scores, axis=0)
    return CANDIDATES[np.argmax(totals)]


def star_voting(utilities, honesty_pct=100, max_score=5, **kwargs):
    """
    STAR: Score Then Automatic Runoff.
    First round: sum scores to find top 2.
    Second round: head-to-head between top 2 based on who each voter scored higher.
    """
    scores = generate_mixed_profile(utilities, honesty_pct, 'score')
    n_voters = scores.shape[0]
    
    # Round 1: Sum scores
    totals = np.sum(scores, axis=0)
    sorted_indices = np.argsort(totals)[::-1]
    finalist_a = sorted_indices[0]
    finalist_b = sorted_indices[1]
    
    # Round 2: Head-to-head
    a_preferred = np.sum(scores[:, finalist_a] > scores[:, finalist_b])
    b_preferred = np.sum(scores[:, finalist_b] > scores[:, finalist_a])
    
    if a_preferred > b_preferred:
        return CANDIDATES[finalist_a]
    elif b_preferred > a_preferred:
        return CANDIDATES[finalist_b]
    else:
        # Tie: use scores as tiebreaker
        return CANDIDATES[finalist_a if totals[finalist_a] >= totals[finalist_b] else finalist_b]


def approval_voting(utilities, honesty_pct=100, **kwargs):
    """
    Approval voting: Voters approve or disapprove each candidate.
    Most approvals wins.
    """
    approvals = generate_mixed_profile(utilities, honesty_pct, 'approval')
    totals = np.sum(approvals, axis=0)
    return CANDIDATES[np.argmax(totals)]


def three_two_one_voting(utilities, honesty_pct=100, **kwargs):
    """
    3-2-1 Voting:
    1. Rate candidates Good/Acceptable/Bad
    2. Semifinalists: top 3 by "good" ratings
    3. Finalists: top 2 of semifinalists by "good+acceptable"
    4. Winner: preferred by more voters head-to-head
    """
    # Map utilities to ratings: 0-0.33 = Bad, 0.33-0.66 = Acceptable, 0.66-1 = Good
    n_voters, n_candidates = utilities.shape
    
    good = np.sum(utilities > 0.66, axis=0)
    acceptable = np.sum((utilities > 0.33) & (utilities <= 0.66), axis=0)
    
    # Top 3 by good ratings (for 3 candidates, all are semifinalists)
    semifinalists = np.argsort(good)[::-1][:3]
    
    # Top 2 by good + acceptable
    combined = good + acceptable
    finalists_combined = combined[semifinalists]
    finalist_order = np.argsort(finalists_combined)[::-1][:2]
    finalist_a = semifinalists[finalist_order[0]]
    finalist_b = semifinalists[finalist_order[1]]
    
    # Head-to-head using utilities
    a_preferred = np.sum(utilities[:, finalist_a] > utilities[:, finalist_b])
    b_preferred = np.sum(utilities[:, finalist_b] > utilities[:, finalist_a])
    
    if a_preferred > b_preferred:
        return CANDIDATES[finalist_a]
    elif b_preferred > a_preferred:
        return CANDIDATES[finalist_b]
    else:
        return CANDIDATES[finalist_a if good[finalist_a] >= good[finalist_b] else finalist_b]


def bucklin(profile, **kwargs):
    """
    Bucklin voting: Count first choices; if majority, done.
    Otherwise, add second choices; if majority, done. Continue.
    """
    n_voters = profile.shape[0]
    majority = n_voters / 2
    
    counts = np.zeros(3, dtype=np.int32)
    
    for rank in range(3):
        for v in range(n_voters):
            counts[profile[v, rank]] += 1
        
        # Check if any candidate has majority
        if np.max(counts) > majority:
            return CANDIDATES[np.argmax(counts)]
    
    # If no majority after all ranks, highest count wins
    return CANDIDATES[np.argmax(counts)]


def coombs(profile, **kwargs):
    """
    Coombs method: Like IRV but eliminate candidate with most last-place votes.
    """
    remaining = {0, 1, 2}
    
    while len(remaining) > 1:
        # Count last-place votes among remaining
        last_counts = np.zeros(3, dtype=np.int32)
        for v in range(profile.shape[0]):
            # Find last remaining candidate in this voter's ranking
            for c in reversed(profile[v]):
                if c in remaining:
                    last_counts[c] += 1
                    break
        
        # Eliminate candidate with most last-place votes
        loser = max(remaining, key=lambda c: last_counts[c])
        remaining.remove(loser)
    
    return CANDIDATES[remaining.pop()]


def nanson(profile, **kwargs):
    """
    Nanson's method: Eliminate candidates with below-average Borda score.
    Repeat until one remains.
    """
    remaining = {0, 1, 2}
    
    while len(remaining) > 1:
        # Calculate Borda scores for remaining candidates
        scores = np.zeros(3, dtype=np.float64)
        n_remaining = len(remaining)
        
        for v in range(profile.shape[0]):
            rank = 0
            for c in profile[v]:
                if c in remaining:
                    scores[c] += (n_remaining - 1 - rank)
                    rank += 1
        
        # Average among remaining
        avg = sum(scores[c] for c in remaining) / n_remaining
        
        # Eliminate below-average
        to_remove = [c for c in remaining if scores[c] < avg]
        if not to_remove:
            # All tied - eliminate one
            to_remove = [min(remaining, key=lambda c: scores[c])]
        
        for c in to_remove:
            remaining.remove(c)
    
    return CANDIDATES[remaining.pop()] if remaining else None


def baldwin(profile, **kwargs):
    """
    Baldwin's method: Like Nanson but eliminate only the single lowest Borda score.
    """
    remaining = {0, 1, 2}
    
    while len(remaining) > 1:
        scores = np.zeros(3, dtype=np.float64)
        n_remaining = len(remaining)
        
        for v in range(profile.shape[0]):
            rank = 0
            for c in profile[v]:
                if c in remaining:
                    scores[c] += (n_remaining - 1 - rank)
                    rank += 1
        
        # Eliminate single lowest
        loser = min(remaining, key=lambda c: scores[c])
        remaining.remove(loser)
    
    return CANDIDATES[remaining.pop()]


def ranked_pairs(profile, **kwargs):
    """
    Ranked Pairs (Tideman): Lock in pairwise victories from largest to smallest,
    skipping any that would create a cycle.
    """
    positions = _get_positions(profile)
    n = profile.shape[0]
    
    # Calculate all pairwise margins
    margins = []
    for a in range(3):
        for b in range(a + 1, 3):
            a_wins = np.sum(positions[:, a] < positions[:, b])
            b_wins = n - a_wins
            if a_wins > b_wins:
                margins.append((a_wins - b_wins, a, b))
            elif b_wins > a_wins:
                margins.append((b_wins - a_wins, b, a))
    
    # Sort by margin (largest first)
    margins.sort(reverse=True)
    
    # Lock in edges, checking for cycles
    locked = set()  # Set of (winner, loser) pairs
    
    def creates_cycle(winner, loser):
        # Check if adding winner->loser creates a path loser->...->winner
        visited = set()
        stack = [loser]
        while stack:
            current = stack.pop()
            if current == winner:
                return True
            if current in visited:
                continue
            visited.add(current)
            for w, l in locked:
                if w == current:
                    stack.append(l)
        return False
    
    for margin, winner, loser in margins:
        if not creates_cycle(winner, loser):
            locked.add((winner, loser))
    
    # Find the source (candidate with no incoming edges)
    incoming = {0: 0, 1: 0, 2: 0}
    for w, l in locked:
        incoming[l] += 1
    
    for c in range(3):
        if incoming[c] == 0:
            return CANDIDATES[c]
    
    # Fallback
    return CANDIDATES[0]


def schulze(profile, **kwargs):
    """
    Schulze method: Find strongest paths between all pairs of candidates.
    A beats B if the strongest path A->B is stronger than B->A.
    """
    positions = _get_positions(profile)
    n = profile.shape[0]
    
    # Preference matrix d[a,b] = votes preferring a over b
    d = np.zeros((3, 3), dtype=np.int32)
    for a in range(3):
        for b in range(3):
            if a != b:
                d[a, b] = np.sum(positions[:, a] < positions[:, b])
    
    # Strongest path matrix
    p = np.zeros((3, 3), dtype=np.int32)
    for a in range(3):
        for b in range(3):
            if a != b:
                if d[a, b] > d[b, a]:
                    p[a, b] = d[a, b]
    
    # Floyd-Warshall for strongest paths
    for k in range(3):
        for a in range(3):
            for b in range(3):
                if a != b:
                    p[a, b] = max(p[a, b], min(p[a, k], p[k, b]))
    
    # Find winner: a wins if p[a,b] > p[b,a] for all b
    for a in range(3):
        wins_all = True
        for b in range(3):
            if a != b:
                if p[a, b] <= p[b, a]:
                    wins_all = False
                    break
        if wins_all:
            return CANDIDATES[a]
    
    # Fallback: most pairwise wins
    pairwise_wins = np.zeros(3, dtype=np.int32)
    for a in range(3):
        for b in range(3):
            if a != b and p[a, b] > p[b, a]:
                pairwise_wins[a] += 1
    
    return CANDIDATES[np.argmax(pairwise_wins)]


def kemeny_young(profile, **kwargs):
    """
    Kemeny-Young: Find the ranking that minimizes total "Kendall tau distance"
    to all voters' rankings.
    """
    from itertools import permutations
    
    n_voters = profile.shape[0]
    
    # For each possible ranking, count agreement with voter preferences
    best_ranking = None
    best_score = -1
    
    for perm in permutations([0, 1, 2]):
        score = 0
        for v in range(n_voters):
            voter_ranking = profile[v]
            for i in range(3):
                for j in range(i + 1, 3):
                    # Check if this pair is in same order
                    perm_order = np.where(np.array(perm) == voter_ranking[i])[0][0] < \
                                 np.where(np.array(perm) == voter_ranking[j])[0][0]
                    voter_order = True  # i comes before j in voter's ranking
                    if perm_order == voter_order:
                        score += 1
        
        if score > best_score:
            best_score = score
            best_ranking = perm
    
    return CANDIDATES[best_ranking[0]] if best_ranking else None


def majority_judgment(utilities, honesty_pct=100, **kwargs):
    """
    Majority Judgment: Rate candidates on a scale, winner is the one with
    highest median rating.
    """
    # Map utilities to grades 0-5
    n_voters, n_candidates = utilities.shape
    
    # Mix honest and strategic ratings
    if honesty_pct < 100:
        n_honest = int(n_voters * honesty_pct / 100.0)
        honest_mask = np.zeros(n_voters, dtype=bool)
        if n_honest > 0:
            honest_indices = np.random.choice(n_voters, size=n_honest, replace=False)
            honest_mask[honest_indices] = True
        
        honest_grades = np.round(utilities * 5).astype(np.int32)
        strategic_grades = generate_strategic_scores(utilities, max_score=5)
        grades = np.where(honest_mask[:, np.newaxis], honest_grades, strategic_grades)
    else:
        grades = np.round(utilities * 5).astype(np.int32)
    
    # Find median for each candidate
    medians = np.median(grades, axis=0)
    
    # Tiebreaker: use mean of grades above/below median
    if np.sum(medians == np.max(medians)) > 1:
        return CANDIDATES[np.argmax(np.mean(grades, axis=0))]
    
    return CANDIDATES[np.argmax(medians)]


# Registry of all voting rules
VOTING_RULES = {
    'plurality': ('ranking', plurality),
    'borda': ('ranking', borda),
    'antiplurality': ('ranking', antiplurality),
    'instant_runoff': ('ranking', instant_runoff),
    'copeland': ('ranking', copeland),
    'minimax': ('ranking', minimax),
    'plurality_runoff': ('ranking', plurality_runoff),
    'black': ('ranking', black),
    'condorcet': ('ranking', condorcet),
    'bucklin': ('ranking', bucklin),
    'coombs': ('ranking', coombs),
    'nanson': ('ranking', nanson),
    'baldwin': ('ranking', baldwin),
    'ranked_pairs': ('ranking', ranked_pairs),
    'schulze': ('ranking', schulze),
    'kemeny_young': ('ranking', kemeny_young),
    'score_voting': ('utility', score_voting),
    'star': ('utility', star_voting),
    'approval': ('utility', approval_voting),
    '3-2-1': ('utility', three_two_one_voting),
    'majority_judgment': ('utility', majority_judgment),
}

# Nice display names
RULE_DISPLAY_NAMES = {
    'plurality': 'Plurality',
    'borda': 'Borda Count',
    'antiplurality': 'Anti-plurality',
    'instant_runoff': 'Instant Runoff (IRV)',
    'copeland': 'Copeland',
    'minimax': 'Minimax',
    'plurality_runoff': 'Plurality Runoff',
    'black': 'Black',
    'condorcet': 'Condorcet',
    'bucklin': 'Bucklin',
    'coombs': 'Coombs',
    'nanson': 'Nanson',
    'baldwin': 'Baldwin',
    'ranked_pairs': 'Ranked Pairs',
    'schulze': 'Schulze',
    'kemeny_young': 'Kemeny-Young',
    'score_voting': 'Score/Range',
    'star': 'STAR',
    'approval': 'Approval',
    '3-2-1': '3-2-1 Voting',
    'majority_judgment': 'Majority Judgment',
}


def run_voting_rule(rule_name, voter_positions, candidate_positions, honesty_pct=100):
    """
    Run a voting rule and return the winner.
    
    Parameters:
        rule_name: Name of the voting rule
        voter_positions: (n_voters, 2) positions
        candidate_positions: (n_candidates, 2) positions
        honesty_pct: Honesty percentage (0-100)
    
    Returns:
        winner: 'A', 'B', 'C', or None
    """
    rule_type, rule_func = VOTING_RULES[rule_name]
    utilities = compute_utilities(voter_positions, candidate_positions)
    
    if rule_type == 'utility':
        # Utility-based rules get utilities directly
        return rule_func(utilities, honesty_pct=honesty_pct)
    else:
        # Ranking-based rules get a profile
        profile = generate_mixed_profile(utilities, honesty_pct, 'ranking')
        return rule_func(profile)


# =============================================================================
# Yee Diagram Generation
# =============================================================================

def generate_yee_diagram(candidate_positions, rule_name, n_voters_per_point=51,
                         honesty_pct=100, resolution=50, gen_method='gaussian',
                         seed=None):
    """
    Generate a Yee diagram for a voting rule.
    
    For each point in a grid, simulate an election with voters centered there.
    
    Returns:
        winners: (resolution, resolution) array of winner indices (0, 1, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_coords = np.linspace(0.05, 0.95, resolution)
    y_coords = np.linspace(0.05, 0.95, resolution)
    
    winners = np.zeros((resolution, resolution), dtype=np.int32)
    
    rule_type, rule_func = VOTING_RULES[rule_name]
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Generate voters centered at (x, y)
            voter_positions = generate_voters_gaussian(
                n_voters_per_point, center=(x, y), spread=0.12
            )
            
            utilities = compute_utilities(voter_positions, candidate_positions)
            
            if rule_type == 'utility':
                winner = rule_func(utilities, honesty_pct=honesty_pct)
            else:
                profile = generate_mixed_profile(utilities, honesty_pct, 'ranking')
                winner = rule_func(profile)
            
            if winner == 'A':
                winners[j, i] = 0
            elif winner == 'B':
                winners[j, i] = 1
            elif winner == 'C':
                winners[j, i] = 2
            else:
                # No winner or tie - use nearest candidate
                center = np.array([x, y])
                dists = np.sqrt(np.sum((candidate_positions - center)**2, axis=1))
                winners[j, i] = np.argmin(dists)
    
    return winners, x_coords, y_coords


# =============================================================================
# Global State
# =============================================================================

# Candidate positions (mutable - can be dragged)
candidate_positions = DEFAULT_CANDIDATE_POSITIONS.copy()

# Simulation parameters
n_voters_per_point = 51
honesty_pct = 100
yee_resolution = 40
generation_method = 'gaussian'
selected_rule = 'plurality'

# UI state
dragging_candidate = None
yee_data = None


# =============================================================================
# UI Setup
# =============================================================================

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Voting Method & Honesty Visualizer", fontsize=16, weight='bold')

# Main Yee diagram area
ax_yee = fig.add_axes([0.05, 0.25, 0.55, 0.65])

# Statistics/info area
ax_info = fig.add_axes([0.65, 0.50, 0.32, 0.40])
ax_info.axis('off')

# Rule comparison area
ax_compare = fig.add_axes([0.65, 0.25, 0.32, 0.22])
ax_compare.axis('off')


def draw_yee_diagram():
    """Draw the Yee diagram with candidate positions."""
    global yee_data
    
    ax_yee.clear()
    ax_yee.set_title(f"Yee Diagram: {RULE_DISPLAY_NAMES.get(selected_rule, selected_rule)}\n"
                     f"(Honesty: {honesty_pct}%)", fontsize=12)
    
    # Generate Yee diagram
    winners, x_coords, y_coords = generate_yee_diagram(
        candidate_positions,
        selected_rule,
        n_voters_per_point=n_voters_per_point,
        honesty_pct=honesty_pct,
        resolution=yee_resolution,
        gen_method=generation_method,
        seed=42
    )
    yee_data = (winners, x_coords, y_coords)
    
    # Create color map
    cmap_colors = [CANDIDATE_COLORS[0], CANDIDATE_COLORS[1], CANDIDATE_COLORS[2]]
    cmap = mcolors.ListedColormap(cmap_colors)
    
    # Plot as image
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    ax_yee.imshow(winners, origin='lower', extent=extent, cmap=cmap,
                  vmin=0, vmax=2, aspect='equal', alpha=0.7)
    
    # Draw candidate positions (draggable markers)
    for c in range(N_CANDIDATES):
        # Draw candidate marker
        circle = Circle(
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            radius=0.03,
            facecolor=CANDIDATE_COLORS[c],
            edgecolor='white',
            linewidth=3,
            zorder=10
        )
        ax_yee.add_patch(circle)
        
        # Add label
        ax_yee.annotate(
            CANDIDATES[c],
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=14, weight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor=CANDIDATE_COLORS[c], alpha=0.9)
        )
    
    ax_yee.set_xlim(0, 1)
    ax_yee.set_ylim(0, 1)
    ax_yee.set_xlabel("Policy Dimension X")
    ax_yee.set_ylabel("Policy Dimension Y")
    ax_yee.grid(True, alpha=0.3)
    
    # Add instructions
    ax_yee.text(0.5, -0.08, "Drag candidate markers to reposition them",
                ha='center', transform=ax_yee.transAxes, fontsize=9, style='italic')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CANDIDATE_COLORS[0], label='A wins'),
        Patch(facecolor=CANDIDATE_COLORS[1], label='B wins'),
        Patch(facecolor=CANDIDATE_COLORS[2], label='C wins'),
    ]
    ax_yee.legend(handles=legend_elements, loc='upper right', fontsize=8)


def update_info():
    """Update the information panel."""
    ax_info.clear()
    ax_info.axis('off')
    
    lines = []
    lines.append("VOTING RULE INFORMATION")
    lines.append("=" * 45)
    lines.append("")
    lines.append(f"Selected Rule: {RULE_DISPLAY_NAMES.get(selected_rule, selected_rule)}")
    lines.append(f"Honesty: {honesty_pct}%")
    lines.append(f"Voters per point: {n_voters_per_point}")
    lines.append(f"Resolution: {yee_resolution}x{yee_resolution}")
    lines.append(f"Generation: {generation_method}")
    lines.append("")
    lines.append("Candidate Positions:")
    for c in range(N_CANDIDATES):
        x, y = candidate_positions[c]
        lines.append(f"  {CANDIDATES[c]}: ({x:.2f}, {y:.2f})")
    
    lines.append("")
    lines.append("-" * 45)
    lines.append("")
    
    # Calculate win percentages from Yee diagram
    if yee_data is not None:
        winners, _, _ = yee_data
        total = winners.size
        for c in range(N_CANDIDATES):
            count = np.sum(winners == c)
            pct = 100 * count / total
            bar = '█' * int(pct / 3)
            lines.append(f"  {CANDIDATES[c]}: {pct:>5.1f}% {bar}")
    
    lines.append("")
    lines.append("-" * 45)
    lines.append("RULE DESCRIPTION:")
    lines.append("-" * 45)
    
    # Add rule descriptions
    descriptions = {
        'plurality': "First-past-the-post: Most first-choice votes wins.",
        'borda': "Rank points: 1st=2pts, 2nd=1pt, 3rd=0pts.",
        'antiplurality': "Fewest last-place votes wins.",
        'instant_runoff': "Eliminate lowest, transfer votes. Repeat.",
        'copeland': "Pairwise wins minus losses.",
        'minimax': "Minimize worst pairwise defeat margin.",
        'plurality_runoff': "Top 2 plurality face off head-to-head.",
        'black': "Condorcet winner, else Borda.",
        'condorcet': "Beats all others head-to-head (if exists).",
        'bucklin': "Add ranks until majority reached.",
        'coombs': "Eliminate most last-place votes.",
        'nanson': "Eliminate below-average Borda. Repeat.",
        'baldwin': "Eliminate lowest Borda. Repeat.",
        'ranked_pairs': "Lock pairwise wins (no cycles).",
        'schulze': "Strongest path comparison.",
        'kemeny_young': "Ranking closest to all voters.",
        'score_voting': "Sum of scores (0-5) wins.",
        'star': "Top 2 scorers, then head-to-head.",
        'approval': "Most approvals wins.",
        '3-2-1': "Rate Good/OK/Bad, narrow to 2, head-to-head.",
        'majority_judgment': "Highest median rating wins.",
    }
    
    desc = descriptions.get(selected_rule, "No description available.")
    # Word wrap
    words = desc.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= 45:
            line += (" " if line else "") + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    
    text = '\n'.join(lines)
    ax_info.text(0.02, 0.98, text, transform=ax_info.transAxes,
                 fontsize=9, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_comparison():
    """Update the rule comparison panel showing all rules' results."""
    ax_compare.clear()
    ax_compare.axis('off')
    
    # Generate one profile at center and show all rules' results
    center = np.mean(candidate_positions, axis=0)
    voter_positions = generate_voters_gaussian(
        n_voters_per_point, center=tuple(center), spread=0.15
    )
    
    lines = []
    lines.append("QUICK COMPARISON (voters at center)")
    lines.append("-" * 35)
    
    results = {}
    for rule_name in VOTING_RULES:
        winner = run_voting_rule(rule_name, voter_positions, candidate_positions, honesty_pct)
        results[rule_name] = winner if winner else '-'
    
    # Group results
    for rule_name in ['plurality', 'borda', 'instant_runoff', 'star', 'approval', 'schulze']:
        display = RULE_DISPLAY_NAMES.get(rule_name, rule_name)[:15]
        winner = results.get(rule_name, '-')
        lines.append(f"  {display:<15}: {winner}")
    
    text = '\n'.join(lines)
    ax_compare.text(0.02, 0.98, text, transform=ax_compare.transAxes,
                    fontsize=8, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))


def update_display():
    """Full display update."""
    draw_yee_diagram()
    update_info()
    update_comparison()
    fig.canvas.draw_idle()


# Initial draw
update_display()


# =============================================================================
# UI Controls
# =============================================================================

# Voting rule selector (main rules)
ax_rule1 = fig.add_axes([0.05, 0.12, 0.12, 0.10])
ax_rule1.set_title('Ranked Choice', fontsize=8)
rule_list1 = ['plurality', 'borda', 'instant_runoff', 'copeland', 'schulze']
radio_rule1 = RadioButtons(ax_rule1, rule_list1, active=0)

ax_rule2 = fig.add_axes([0.18, 0.12, 0.12, 0.10])
ax_rule2.set_title('More Ranked', fontsize=8)
rule_list2 = ['minimax', 'ranked_pairs', 'coombs', 'bucklin', 'black']
radio_rule2 = RadioButtons(ax_rule2, rule_list2, active=0)

ax_rule3 = fig.add_axes([0.31, 0.12, 0.12, 0.10])
ax_rule3.set_title('Score-Based', fontsize=8)
rule_list3 = ['score_voting', 'star', 'approval', '3-2-1', 'majority_judgment']
radio_rule3 = RadioButtons(ax_rule3, rule_list3, active=0)

# Honesty slider
ax_honesty = fig.add_axes([0.05, 0.05, 0.25, 0.025])
slider_honesty = Slider(ax_honesty, 'Honesty %', 0, 100, valinit=100, valstep=5)

# Voters per point slider
ax_voters = fig.add_axes([0.35, 0.05, 0.20, 0.025])
slider_voters = Slider(ax_voters, 'Voters', 11, 101, valinit=51, valstep=10)

# Resolution slider
ax_res = fig.add_axes([0.60, 0.05, 0.15, 0.025])
slider_res = Slider(ax_res, 'Resolution', 20, 80, valinit=40, valstep=5)

# Generation method selector
ax_gen = fig.add_axes([0.45, 0.12, 0.10, 0.10])
ax_gen.set_title('Voter Dist', fontsize=8)
radio_gen = RadioButtons(ax_gen, ['uniform', 'gaussian', 'clustered', 'polarized'], active=1)

# Reset button
ax_reset = fig.add_axes([0.80, 0.05, 0.08, 0.04])
button_reset = Button(ax_reset, 'Reset Pos')

# Generate button
ax_generate = fig.add_axes([0.90, 0.05, 0.08, 0.04])
button_generate = Button(ax_generate, 'Regenerate')


# =============================================================================
# Callbacks
# =============================================================================

def on_rule_change1(label):
    global selected_rule
    selected_rule = label
    # Uncheck other radio buttons
    for i, btn in enumerate(radio_rule2.circles):
        btn.set_facecolor('white')
    for i, btn in enumerate(radio_rule3.circles):
        btn.set_facecolor('white')
    update_display()

def on_rule_change2(label):
    global selected_rule
    selected_rule = label
    for i, btn in enumerate(radio_rule1.circles):
        btn.set_facecolor('white')
    for i, btn in enumerate(radio_rule3.circles):
        btn.set_facecolor('white')
    update_display()

def on_rule_change3(label):
    global selected_rule
    selected_rule = label
    for i, btn in enumerate(radio_rule1.circles):
        btn.set_facecolor('white')
    for i, btn in enumerate(radio_rule2.circles):
        btn.set_facecolor('white')
    update_display()

def on_honesty_change(val):
    global honesty_pct
    honesty_pct = int(val)
    update_display()

def on_voters_change(val):
    global n_voters_per_point
    n_voters_per_point = int(val)
    update_display()

def on_res_change(val):
    global yee_resolution
    yee_resolution = int(val)
    update_display()

def on_gen_change(label):
    global generation_method
    generation_method = label
    update_display()

def on_reset(event):
    global candidate_positions
    candidate_positions = DEFAULT_CANDIDATE_POSITIONS.copy()
    update_display()

def on_generate(event):
    update_display()


# Mouse event handlers for dragging candidates
def on_press(event):
    global dragging_candidate
    if event.inaxes != ax_yee:
        return
    
    # Check if click is near any candidate
    click_pos = np.array([event.xdata, event.ydata])
    for c in range(N_CANDIDATES):
        dist = np.sqrt(np.sum((candidate_positions[c] - click_pos)**2))
        if dist < 0.05:  # Threshold for clicking on candidate
            dragging_candidate = c
            return


def on_motion(event):
    global candidate_positions
    if dragging_candidate is None:
        return
    if event.inaxes != ax_yee:
        return
    if event.xdata is None or event.ydata is None:
        return
    
    # Update candidate position
    new_x = np.clip(event.xdata, 0.05, 0.95)
    new_y = np.clip(event.ydata, 0.05, 0.95)
    candidate_positions[dragging_candidate] = [new_x, new_y]
    
    # Quick visual update (just redraw candidates)
    ax_yee.clear()
    if yee_data is not None:
        winners, x_coords, y_coords = yee_data
        cmap_colors = [CANDIDATE_COLORS[0], CANDIDATE_COLORS[1], CANDIDATE_COLORS[2]]
        cmap = mcolors.ListedColormap(cmap_colors)
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        ax_yee.imshow(winners, origin='lower', extent=extent, cmap=cmap,
                      vmin=0, vmax=2, aspect='equal', alpha=0.7)
    
    for c in range(N_CANDIDATES):
        circle = Circle(
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            radius=0.03,
            facecolor=CANDIDATE_COLORS[c],
            edgecolor='white' if c != dragging_candidate else 'yellow',
            linewidth=3 if c != dragging_candidate else 4,
            zorder=10
        )
        ax_yee.add_patch(circle)
        ax_yee.annotate(
            CANDIDATES[c],
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=14, weight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor=CANDIDATE_COLORS[c], alpha=0.9)
        )
    
    ax_yee.set_xlim(0, 1)
    ax_yee.set_ylim(0, 1)
    ax_yee.set_xlabel("Policy Dimension X")
    ax_yee.set_ylabel("Policy Dimension Y")
    ax_yee.set_title(f"Yee Diagram: {RULE_DISPLAY_NAMES.get(selected_rule, selected_rule)}\n"
                     f"(Honesty: {honesty_pct}%) - Dragging...", fontsize=12)
    
    fig.canvas.draw_idle()


def on_release(event):
    global dragging_candidate
    if dragging_candidate is not None:
        dragging_candidate = None
        update_display()  # Full regeneration on release


# Connect callbacks
radio_rule1.on_clicked(on_rule_change1)
radio_rule2.on_clicked(on_rule_change2)
radio_rule3.on_clicked(on_rule_change3)
slider_honesty.on_changed(on_honesty_change)
slider_voters.on_changed(on_voters_change)
slider_res.on_changed(on_res_change)
radio_gen.on_clicked(on_gen_change)
button_reset.on_clicked(on_reset)
button_generate.on_clicked(on_generate)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)


plt.show()






