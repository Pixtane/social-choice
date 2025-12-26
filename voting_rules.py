import numpy as np

# Optimized voting rules using numpy arrays
# Profile format: (n_voters, 3) array where each row is a ranking [0,1,2] permutation
# 0=A, 1=B, 2=C

CANDIDATES = ['A', 'B', 'C']

def _get_positions(profile):
    """Get position matrix: positions[i,c] = position of candidate c in voter i's ranking."""
    n = profile.shape[0]
    positions = np.empty((n, 3), dtype=np.int8)
    for c in range(3):
        positions[:, c] = np.where(profile == c)[1].reshape(n)
    return positions

def plurality(profile, positions=None):
    """Each voter's top choice gets 1 point."""
    first_choices = profile[:, 0]
    counts = np.bincount(first_choices, minlength=3)
    return CANDIDATES[np.argmax(counts)]

def borda(profile, positions=None):
    """Points by position: 1st=2pts, 2nd=1pt, 3rd=0pts."""
    if positions is None:
        positions = _get_positions(profile)
    scores = np.sum(2 - positions, axis=0)
    return CANDIDATES[np.argmax(scores)]

def antiplurality(profile, positions=None):
    """Each voter's last choice gets -1 point."""
    last_choices = profile[:, 2]
    counts = np.bincount(last_choices, minlength=3)
    return CANDIDATES[np.argmin(counts)]

def instant_runoff(profile, positions=None):
    """Eliminate candidate with fewest first-place votes, repeat until winner."""
    if positions is None:
        positions = _get_positions(profile)
    remaining = {0, 1, 2}
    
    while len(remaining) > 1:
        counts = np.zeros(3, dtype=np.int32)
        for v in range(profile.shape[0]):
            for c in profile[v]:
                if c in remaining:
                    counts[c] += 1
                    break
        loser = min(remaining, key=lambda c: counts[c])
        remaining.remove(loser)
    
    return CANDIDATES[remaining.pop()]

def copeland(profile, positions=None):
    """Win pairwise matchup = +1, lose = -1, tie = 0."""
    if positions is None:
        positions = _get_positions(profile)
    n = profile.shape[0]
    scores = np.zeros(3, dtype=np.int32)
    
    for a in range(3):
        for b in range(a + 1, 3):
            a_wins = np.sum(positions[:, a] < positions[:, b])
            if a_wins > n // 2:
                scores[a] += 1
                scores[b] -= 1
            elif a_wins < n // 2:
                scores[b] += 1
                scores[a] -= 1
    
    return CANDIDATES[np.argmax(scores)]

def minimax(profile, positions=None):
    """Minimize worst pairwise defeat margin."""
    if positions is None:
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

def plurality_runoff(profile, positions=None):
    """Top two from plurality face off in pairwise comparison."""
    if positions is None:
        positions = _get_positions(profile)
    
    first_choices = profile[:, 0]
    counts = np.bincount(first_choices, minlength=3)
    top_two = np.argsort(counts)[-2:]
    
    a, b = top_two
    a_wins = np.sum(positions[:, a] < positions[:, b])
    return CANDIDATES[a if a_wins >= profile.shape[0] - a_wins else b]

def condorcet(profile, positions=None):
    """Return Condorcet winner if exists, else None."""
    if positions is None:
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

def black(profile, positions=None):
    """Condorcet winner if exists, otherwise Borda winner."""
    if positions is None:
        positions = _get_positions(profile)
    winner = condorcet(profile, positions)
    return winner if winner else borda(profile, positions)

VOTING_RULES = {
    'plurality': plurality,
    'borda': borda,
    'antiplurality': antiplurality,
    'instant_runoff': instant_runoff,
    'copeland': copeland,
    'minimax': minimax,
    'plurality_runoff': plurality_runoff,
    'black': black,
}
