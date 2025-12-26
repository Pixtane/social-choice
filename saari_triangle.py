import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from voting_rules import VOTING_RULES, CANDIDATES, _get_positions

# Rank order convention used throughout (0=A, 1=B, 2=C):
#   0: ABC, 1: ACB, 2: BAC, 3: BCA, 4: CAB, 5: CBA
RANKINGS_6_STR = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]

# Pairwise contributions per ranking, in order: [A>B, A>C, B>C]
# (matches the example the user provided)
PAIRWISE = np.array(
    [
        [1, 1, 1],  # ABC
        [1, 1, 0],  # ACB
        [0, 1, 1],  # BAC
        [0, 0, 1],  # BCA
        [1, 0, 0],  # CAB
        [0, 0, 0],  # CBA
    ],
    dtype=np.int64,
)

# Equilateral triangle vertices (barycentric coordinates)
# A at bottom-left, B at bottom-right, C at top
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],      # A (100% A)
    [1.0, 0.0],      # B (100% B)
    [0.5, np.sqrt(3)/2]  # C (100% C)
])

def barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (x, y, z) to Cartesian (x, y) in triangle."""
    x, y, z = bary
    cart_x = TRIANGLE_VERTICES[0][0] * x + TRIANGLE_VERTICES[1][0] * y + TRIANGLE_VERTICES[2][0] * z
    cart_y = TRIANGLE_VERTICES[0][1] * x + TRIANGLE_VERTICES[1][1] * y + TRIANGLE_VERTICES[2][1] * z
    return cart_x, cart_y

def cartesian_to_barycentric(cart_x, cart_y):
    """Convert Cartesian coordinates to barycentric coordinates (x, y, z)."""
    # Using Cramer's rule
    v0 = TRIANGLE_VERTICES[0]
    v1 = TRIANGLE_VERTICES[1]
    v2 = TRIANGLE_VERTICES[2]
    
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return None
    
    a = ((v1[1] - v2[1]) * (cart_x - v2[0]) + (v2[0] - v1[0]) * (cart_y - v2[1])) / denom
    b = ((v2[1] - v0[1]) * (cart_x - v2[0]) + (v0[0] - v2[0]) * (cart_y - v2[1])) / denom
    c = 1 - a - b
    
    if a < 0 or b < 0 or c < 0 or a > 1 or b > 1 or c > 1:
        return None
    
    return (a, b, c)

def normalize_barycentric(x, y):
    """Normalize x, y to valid barycentric coordinates (x, y, z) where x+y+z=1."""
    if x + y > 1:
        # Scale down proportionally
        scale = 1.0 / (x + y)
        x *= scale
        y *= scale
    z = 1.0 - x - y
    return (x, y, z)

def barycentric_to_profile_distribution(bary, n_voters):
    """
    Convert barycentric coordinates to a distribution of preference profiles.
    Uses Saari's triangle method where the position determines the distribution
    of the 6 possible rankings.
    
    The 6 rankings are:
    0: ABC, 1: ACB, 2: BAC, 3: BCA, 4: CAB, 5: CBA
    """
    x, y, z = bary
    
    # Saari's triangle: The position in the triangle determines the distribution
    # of preference orderings. We use a model where:
    # - Near vertex A: More ABC and ACB rankings
    # - Near vertex B: More BAC and BCA rankings  
    # - Near vertex C: More CAB and CBA rankings
    # - The exact distribution depends on the position relative to median lines
    
    # Calculate weights for each ranking type
    # ABC: A first, B second, C third - favored when x is high
    w_abc = x * (1 - y * 0.5) * (1 - z * 0.5)
    # ACB: A first, C second, B third - favored when x is high, z > y
    w_acb = x * (1 - y) * z
    # BAC: B first, A second, C third - favored when y is high, x > z
    w_bac = y * (1 - z) * x
    # BCA: B first, C second, A third - favored when y is high
    w_bca = y * (1 - x * 0.5) * (1 - z * 0.5)
    # CAB: C first, A second, B third - favored when z is high, x > y
    w_cab = z * (1 - y) * x
    # CBA: C first, B second, A third - favored when z is high
    w_cba = z * (1 - x * 0.5) * (1 - y * 0.5)
    
    weights = np.array([w_abc, w_acb, w_bac, w_bca, w_cab, w_cba])
    
    # Add small base probability to avoid zeros
    weights = weights + 0.01
    
    # Normalize weights
    total = np.sum(weights)
    if total < 1e-10:
        # Uniform distribution if all weights are zero
        weights = np.ones(6) / 6.0
    else:
        weights = weights / total
    
    # Generate actual voter counts
    counts = np.random.multinomial(n_voters, weights)
    
    # Create profile
    rankings = [
        [0, 1, 2],  # ABC
        [0, 2, 1],  # ACB
        [1, 0, 2],  # BAC
        [1, 2, 0],  # BCA
        [2, 0, 1],  # CAB
        [2, 1, 0]   # CBA
    ]
    
    profile = []
    for i, count in enumerate(counts):
        for _ in range(count):
            profile.append(rankings[i])
    
    # Convert to numpy array and shuffle to randomize order
    profile_array = np.array(profile, dtype=np.int8)
    np.random.shuffle(profile_array)
    
    return profile_array


def _rankings_6():
    """
    The 6 strict rankings of 3 candidates (0=A,1=B,2=C), matching the index
    convention used throughout this file:
      0: ABC, 1: ACB, 2: BAC, 3: BCA, 4: CAB, 5: CBA
    """
    return [
        [0, 1, 2],  # ABC
        [0, 2, 1],  # ACB
        [1, 0, 2],  # BAC
        [1, 2, 0],  # BCA
        [2, 0, 1],  # CAB
        [2, 1, 0],  # CBA
    ]


_RANKING_TUPLE_TO_IDX = {
    (0, 1, 2): 0,  # ABC
    (0, 2, 1): 1,  # ACB
    (1, 0, 2): 2,  # BAC
    (1, 2, 0): 3,  # BCA
    (2, 0, 1): 4,  # CAB
    (2, 1, 0): 5,  # CBA
}


def profile_to_ranking_freq(profile):
    """
    Convert a (n_voters, 3) profile of permutations into a length-6 frequency vector
    in the canonical order [ABC, ACB, BAC, BCA, CAB, CBA].
    """
    freq = np.zeros(6, dtype=np.int64)
    for row in profile:
        idx = _RANKING_TUPLE_TO_IDX.get(tuple(int(x) for x in row))
        if idx is None:
            raise ValueError(f"Unexpected ranking row: {row}")
        freq[idx] += 1
    return freq


def pairwise_counts_from_freq(freq):
    """
    Given a length-6 ranking frequency vector, return (A_B, A_C, B_C, total).
    """
    # Accept either integer counts or float weights.
    f = np.asarray(freq, dtype=np.float64)
    if f.shape != (6,):
        raise ValueError("freq must be length 6")
    A_B, A_C, B_C = f @ PAIRWISE
    total = float(np.sum(f))
    return float(A_B), float(A_C), float(B_C), total


def majority_type_and_cycle_from_freq(freq):
    """
    Determine majority type (1-6) and whether the majority relation is cyclic OR tied,
    using the PAIRWISE-matrix method from the provided example.

    Returns:
      (type_num, is_circular)
      - type_num is int 1..6 when strict transitive majority exists
      - type_num is None when cyclic OR any pairwise tie occurs
    """
    t, relation = classify_majority_relation_from_freq(freq)
    return (t, relation != "transitive")


def classify_majority_relation_from_freq(freq, eps=1e-12):
    """
    Classify the majority relation implied by a 6-ranking distribution using PAIRWISE.

    Returns: (type_num, relation)
      - type_num: int 1..6 when relation == 'transitive', else None
      - relation: 'transitive' | 'cycle' | 'tie'
    """
    A_B, A_C, B_C, total = pairwise_counts_from_freq(freq)
    if total <= 0:
        return None, "tie"

    # Any pairwise tie (within eps) => relation 'tie'
    if abs(A_B * 2 - total) <= eps or abs(A_C * 2 - total) <= eps or abs(B_C * 2 - total) <= eps:
        return None, "tie"

    wins = {
        "A": int(A_B > total / 2) + int(A_C > total / 2),
        "B": int((total - A_B) > total / 2) + int(B_C > total / 2),
        "C": int((total - A_C) > total / 2) + int((total - B_C) > total / 2),
    }

    # Strict Condorcet cycle iff each candidate has exactly one win.
    if set(wins.values()) == {1}:
        return None, "cycle"

    # Otherwise transitive (one candidate has 2 wins)
    if not any(w == 2 for w in wins.values()):
        # Shouldn't happen for strict relations with 3 candidates, but keep safe.
        return None, "cycle"

    order = tuple(sorted(wins.keys(), key=lambda k: wins[k], reverse=True))
    order_to_type = {
        ("A", "B", "C"): 1,  # A>B>C
        ("A", "C", "B"): 2,  # A>C>B
        ("B", "A", "C"): 3,  # B>A>C
        ("B", "C", "A"): 4,  # B>C>A
        ("C", "A", "B"): 5,  # C>A>B
        ("C", "B", "A"): 6,  # C>B>A
    }
    return order_to_type.get(order), "transitive"


def is_strict_cycle_from_freq(freq):
    """
    True iff the ranking distribution implies a strict Condorcet cycle.
    (Ties return False.)
    """
    _, rel = classify_majority_relation_from_freq(freq)
    return rel == "cycle"


def profile_from_ranking_weights(weights, n_voters, seed=None):
    """
    Sample a profile (n_voters rankings) from an explicit 6-ranking distribution.

    weights: length-6 array-like, nonnegative, not necessarily normalized.
    """
    if seed is not None:
        np.random.seed(seed)

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (6,):
        raise ValueError("weights must be length 6")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")

    total = float(np.sum(w))
    if total <= 0:
        w = np.ones(6, dtype=np.float64) / 6.0
    else:
        w = w / total

    counts = np.random.multinomial(n_voters, w)
    rankings = _rankings_6()

    profile = []
    for i, count in enumerate(counts):
        for _ in range(int(count)):
            profile.append(rankings[i])

    profile_array = np.array(profile, dtype=np.int8)
    np.random.shuffle(profile_array)
    return profile_array


def barycentric_from_ranking_weights(weights):
    """
    Map a 6-ranking distribution to barycentric coords by first-choice shares:
      x = P(A first) = w_ABC + w_ACB
      y = P(B first) = w_BAC + w_BCA
      z = P(C first) = w_CAB + w_CBA
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    w = w / total
    x = float(w[0] + w[1])
    y = float(w[2] + w[3])
    z = float(w[4] + w[5])
    return (x, y, z)


def _pairwise_majority_winner_from_weights(weights, a, b, eps=1e-12):
    """
    Majority winner between candidates a and b given a 6-ranking distribution.
    Returns a, b, or None for tie.
    """
    rankings = _rankings_6()
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0:
        w = np.ones(6, dtype=np.float64) / 6.0
    else:
        w = w / total

    # p(a > b) is sum of weights of rankings where a appears before b
    p_a = 0.0
    for i, r in enumerate(rankings):
        pos = {r[0]: 0, r[1]: 1, r[2]: 2}
        if pos[a] < pos[b]:
            p_a += float(w[i])

    if abs(p_a - 0.5) <= eps:
        return None
    return a if p_a > 0.5 else b


def get_majority_type_from_ranking_weights(weights):
    """
    Determine majority type (1-6) and circularity directly from a 6-ranking distribution.
    Mirrors get_majority_type_and_cycle(profile), but is deterministic (no sampling).
    """
    return majority_type_and_cycle_from_freq(weights)


def has_condorcet_cycle_from_ranking_weights(weights):
    """
    True iff the pairwise-majority relation implied by the 6-ranking distribution
    is a strict Condorcet cycle (ignores ties; ties return False).
    """
    return is_strict_cycle_from_freq(weights)

def generate_profile_from_point(bary, n_voters, method='saari', seed=None, ranking_weights=None):
    """Generate a voting profile from a point in Saari's triangle."""
    # Use seed for deterministic generation based on coordinates
    if seed is not None:
        np.random.seed(seed)

    # If the point comes with an explicit 6-ranking distribution, use it.
    if ranking_weights is not None:
        return profile_from_ranking_weights(ranking_weights, n_voters, seed=seed)
    
    if method == 'saari':
        return barycentric_to_profile_distribution(bary, n_voters)
    elif method == 'uniform':
        # Uniform random distribution
        base = np.array([0, 1, 2], dtype=np.int8)
        profile = np.empty((n_voters, 3), dtype=np.int8)
        for j in range(n_voters):
            profile[j] = np.random.permutation(base)
        return profile
    elif method == 'weighted':
        # Weight by barycentric coordinates
        x, y, z = bary
        # Probability of each candidate being first choice
        probs = np.array([x, y, z])
        probs = probs / np.sum(probs)
        
        base = np.array([0, 1, 2], dtype=np.int8)
        profile = np.empty((n_voters, 3), dtype=np.int8)
        for j in range(n_voters):
            # Choose first choice based on probabilities
            first = np.random.choice(3, p=probs)
            remaining = [c for c in range(3) if c != first]
            # Choose second choice weighted by remaining candidates' strengths,
            # instead of uniformly at random (uniform second choice makes outcomes
            # look visually "arbitrary" across the triangle).
            rem_probs = probs[remaining]
            rem_probs = rem_probs / np.sum(rem_probs)
            second = np.random.choice(remaining, p=rem_probs)
            third = [c for c in remaining if c != second][0]
            profile[j] = np.array([first, second, third], dtype=np.int8)
        return profile
    else:
        return barycentric_to_profile_distribution(bary, n_voters)

def draw_median_lines(ax, n_voters):
    """Draw median lines that divide the triangle into voting regions."""
    # Median lines connect vertices to midpoints of opposite sides
    # These represent boundaries where pairwise comparisons are equal
    
    # Midpoints of edges
    mid_ab = (TRIANGLE_VERTICES[0] + TRIANGLE_VERTICES[1]) / 2  # midpoint of AB
    mid_bc = (TRIANGLE_VERTICES[1] + TRIANGLE_VERTICES[2]) / 2  # midpoint of BC
    mid_ca = (TRIANGLE_VERTICES[2] + TRIANGLE_VERTICES[0]) / 2  # midpoint of CA
    
    # Median lines (from vertex to opposite midpoint)
    medians = [
        [TRIANGLE_VERTICES[2], mid_ab],  # C to midpoint of AB (line where A and B are equal in pairwise)
        [TRIANGLE_VERTICES[0], mid_bc],  # A to midpoint of BC (line where B and C are equal)
        [TRIANGLE_VERTICES[1], mid_ca],  # B to midpoint of CA (line where C and A are equal)
    ]
    
    for i, median in enumerate(medians):
        ax.plot([median[0][0], median[1][0]], [median[0][1], median[1][1]], 
                'b--', linewidth=2, alpha=0.7, 
                label='Median Lines' if i == 0 else '')
    
    # Draw lines representing boundaries between different voting regions
    # These are lines where one candidate beats another by exactly 50%
    # In Saari's triangle, these create regions with different voting outcomes
    
    # Additional boundary lines for visualization (parallel to edges)
    for i in range(3):
        t = (i + 1) / 4.0
        # Lines parallel to each edge
        # Parallel to AB (from A-C edge to B-C edge)
        p1 = TRIANGLE_VERTICES[0] * (1-t) + TRIANGLE_VERTICES[2] * t
        p2 = TRIANGLE_VERTICES[1] * (1-t) + TRIANGLE_VERTICES[2] * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', linewidth=0.8, alpha=0.4, linestyle=':')
        
        # Parallel to BC
        p1 = TRIANGLE_VERTICES[1] * (1-t) + TRIANGLE_VERTICES[0] * t
        p2 = TRIANGLE_VERTICES[2] * (1-t) + TRIANGLE_VERTICES[0] * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', linewidth=0.8, alpha=0.4, linestyle=':')
        
        # Parallel to CA
        p1 = TRIANGLE_VERTICES[2] * (1-t) + TRIANGLE_VERTICES[1] * t
        p2 = TRIANGLE_VERTICES[0] * (1-t) + TRIANGLE_VERTICES[1] * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', linewidth=0.8, alpha=0.4, linestyle=':')

def _pairwise_majority_winner(profile, a, b):
    """
    Return the majority winner between candidates a and b.
    Returns:
      - a if a beats b
      - b if b beats a
      - None if tie
    """
    positions = _get_positions(profile)
    n = profile.shape[0]
    a_wins = int(np.sum(positions[:, a] < positions[:, b]))
    b_wins = n - a_wins
    if a_wins == b_wins:
        return None
    return a if a_wins > b_wins else b


def get_majority_type_and_cycle(profile):
    """
    Determine the *majority relation* type (1-6) and whether it's circular.

    Type mapping (transitive majority only):
      1: A > B > C
      2: A > C > B
      3: B > A > C
      4: B > C > A
      5: C > A > B
      6: C > B > A

    Returns:
      (type_num, is_circular)
      - type_num is int 1..6 when majority is transitive
      - type_num is None when cyclic or any pairwise tie
    """
    freq = profile_to_ranking_freq(profile)
    return majority_type_and_cycle_from_freq(freq)


def has_condorcet_cycle(profile):
    """Backwards-compatible name: True when pairwise majority is cyclic (or tied)."""
    _, is_circular = get_majority_type_and_cycle(profile)
    return is_circular


def get_majority_type_from_barycentric(bary, eps=1e-9):
    """
    Deterministic "type" classification from barycentric coordinates.

    In this UI, barycentric coordinates (x,y,z) represent the strength of support
    for A,B,C respectively. The 3 medians partition the simplex into 6 regions
    corresponding to the 6 strict orders of (x,y,z). This function returns that
    order mapped to the same 1..6 type numbering used by get_majority_type_and_cycle.

    Returns:
      (type_num, is_circular)
      - type_num is int 1..6 when x,y,z are all distinct
      - (None, True) when any pair is (near) equal (treated like "X" in the UI)
    """
    x, y, z = bary

    # Treat near-equalities as ambiguous (show "X")
    if abs(x - y) < eps or abs(x - z) < eps or abs(y - z) < eps:
        return None, True

    strengths = {0: x, 1: y, 2: z}
    order = tuple(sorted(strengths.keys(), key=lambda c: strengths[c], reverse=True))

    order_to_type = {
        (0, 1, 2): 1,  # A>B>C
        (0, 2, 1): 2,  # A>C>B
        (1, 0, 2): 3,  # B>A>C
        (1, 2, 0): 4,  # B>C>A
        (2, 0, 1): 5,  # C>A>B
        (2, 1, 0): 6,  # C>B>A
    }
    return order_to_type.get(order), False

def has_conflicting_results(results_dict, index):
    """
    Check if different voting rules give different winners for a point.
    """
    winners = []
    for rule_name, winners_list in results_dict.items():
        if index < len(winners_list):
            winner = winners_list[index]
            if winner is not None:
                winners.append(winner)
    
    # Conflicting if we have more than one unique winner
    return len(set(winners)) > 1

def calculate_voting_outcomes(points_data, n_voters, method='saari', rule_names=None):
    """Calculate voting outcomes for all points using selected voting rules."""
    results = {}
    if rule_names is None:
        rule_names = [name for name in VOTING_RULES.keys() if name in selected_voting_rules]
    for rule_name in rule_names:
        results[rule_name] = []
    
    for i, point_data in enumerate(points_data):
        bary = point_data['barycentric']
        ranking_weights = point_data.get('ranking_weights')
        # Use deterministic seed based on coordinates for consistency
        coord_seed = int((bary[0] * 1000 + bary[1] * 100 + bary[2] * 10) * 1000) % (2**31)
        profile = generate_profile_from_point(bary, n_voters, method, seed=coord_seed, ranking_weights=ranking_weights)
        
        for rule_name in results.keys():
            winner = VOTING_RULES[rule_name](profile)
            results[rule_name].append(winner)
    
    return results

# Global state
points_data = []  # List of dicts: {'barycentric': (x,y,z), 'cartesian': (x,y), 'label': str}
current_view = 'triangle'  # 'triangle' or 'calculate' or 'statistics'
show_medians = False
n_voters_sim = 100
n_profiles_sim = 100
generation_method = 'saari'
stats_text_full = ""  # Full statistics text
stats_scroll_pos = 0  # Current scroll position (line number)
stats_lines_per_page = 50  # Approximate lines visible per page
selected_point_idx = None  # index into points_data, or None when nothing selected
selected_voting_rules = set(VOTING_RULES.keys())  # rules included in calculation/statistics
point_generation_mode = 'coordinates'  # 'coordinates' or 'ranking_vector'

# Create figure
fig = plt.figure(figsize=(16, 10))
ax_triangle = fig.add_axes([0.05, 0.30, 0.45, 0.65])
ax_stats = fig.add_axes([0.65, 0.30, 0.42, 0.65])
ax_stats.axis('off')

# Draw the triangle
def draw_triangle(ax, show_points=True, show_medians_flag=False):
    """Draw the equilateral triangle with vertices labeled."""
    ax.clear()
    ax.set_aspect('equal')
    
    # Draw triangle outline
    triangle = Polygon(TRIANGLE_VERTICES, closed=True, fill=False, 
                       edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Label vertices
    ax.text(TRIANGLE_VERTICES[0][0] - 0.05, TRIANGLE_VERTICES[0][1] - 0.05, 
            'A (100%)', fontsize=12, ha='right', weight='bold', color='red')
    ax.text(TRIANGLE_VERTICES[1][0] + 0.05, TRIANGLE_VERTICES[1][1] - 0.05, 
            'B (100%)', fontsize=12, ha='left', weight='bold', color='blue')
    ax.text(TRIANGLE_VERTICES[2][0], TRIANGLE_VERTICES[2][1] + 0.05, 
            'C (100%)', fontsize=12, ha='center', weight='bold', color='green')
    
    # Draw median lines if requested
    if show_medians_flag:
        draw_median_lines(ax, n_voters_sim)
    
    # Draw points
    if show_points and points_data:
        # Color points based on profile type and voting outcomes
        if current_view in ['calculate', 'statistics'] and points_data:
            results = calculate_voting_outcomes(points_data, n_voters_sim, generation_method)
            
            # Color map for profile types (1-6)
            type_colors = {
                1: 'red',      # ABC
                2: 'orange',   # ACB
                3: 'blue',     # BAC
                4: 'cyan',     # BCA
                5: 'green',    # CAB
                6: 'lime'      # CBA
            }
            
            for i, point_data in enumerate(points_data):
                cart_x, cart_y = point_data['cartesian']
                bary = point_data['barycentric']
                ranking_weights = point_data.get('ranking_weights')
                
                # Determine type/circularity.
                # - If the point has an explicit 6-ranking distribution, use it.
                # - Otherwise use the deterministic barycentric-region type.
                if ranking_weights is not None:
                    # Type is based on first-choice shares (barycentric location),
                    # while circularity is based on the pairwise-majority relation
                    # implied by the 6-ranking distribution (cycles only; ties are not cycles).
                    profile_type, _ = get_majority_type_from_barycentric(bary)
                    _, rel = classify_majority_relation_from_freq(ranking_weights)
                    is_circular = (rel != "transitive")
                else:
                    profile_type, is_circular = get_majority_type_from_barycentric(bary)
                
                # Check for conflicting results
                has_conflict = has_conflicting_results(results, i)
                
                # Get color based on majority type (or gray if circular/tie)
                color = type_colors.get(profile_type, 'gray')
                
                # Draw point with appropriate styling
                marker_size = 10 if len(points_data) <= 100 else 8
                
                # Draw point with black border if conflicting
                edge_width = 3 if has_conflict else 0
                edge_color = 'black' if has_conflict else 'none'
                
                ax.plot(cart_x, cart_y, 'o', color=color, markersize=marker_size, 
                       markeredgecolor=edge_color, markeredgewidth=edge_width, alpha=0.8)
                
                # Add profile type number or X
                if is_circular or profile_type is None:
                    # Show C (cycle) or T (tie) instead of number
                    marker = 'X'
                    bbox_color = 'red'
                    if ranking_weights is not None:
                        _, rel = classify_majority_relation_from_freq(ranking_weights)
                        if rel == "cycle":
                            marker = 'C'
                            bbox_color = 'red'
                        elif rel == "tie":
                            marker = 'T'
                            bbox_color = 'gray'
                    else:
                        # Barycentric boundary -> treat as tie marker
                        marker = 'T'
                        bbox_color = 'gray'
                    ax.text(
                        cart_x,
                        cart_y,
                        marker,
                        ha='center',
                        va='center',
                        fontsize=9,
                        weight='bold',
                        color='white',
                        bbox=dict(boxstyle='circle', facecolor=bbox_color, alpha=0.7, pad=0.3),
                    )
                else:
                    # Show majority type number
                    text_color = 'white' if color in ['blue', 'green', 'cyan', 'lime', 'red'] else 'black'
                    ax.text(
                        cart_x,
                        cart_y,
                        str(profile_type),
                        ha='center',
                        va='center',
                        fontsize=8,
                        weight='bold',
                        color=text_color,
                    )
                
                # Only show labels for manually added points (not generated ones)
                if len(points_data) <= 20:  # Only label if not too many points
                    label = point_data.get('label', f'P{i+1}')
                    ax.text(cart_x + 0.03, cart_y + 0.03, label, fontsize=7, alpha=0.7)
        else:
            # Normal view - show points with profile type
            for i, point_data in enumerate(points_data):
                cart_x, cart_y = point_data['cartesian']
                bary = point_data['barycentric']
                ranking_weights = point_data.get('ranking_weights')
                
                # Determine type/circularity.
                if ranking_weights is not None:
                    profile_type, _ = get_majority_type_from_barycentric(bary)
                    _, rel = classify_majority_relation_from_freq(ranking_weights)
                    is_circular = (rel != "transitive")
                else:
                    profile_type, is_circular = get_majority_type_from_barycentric(bary)
                
                # Color map for profile types
                type_colors = {
                    1: 'red',      # ABC
                    2: 'orange',    # ACB
                    3: 'blue',      # BAC
                    4: 'cyan',      # BCA
                    5: 'green',     # CAB
                    6: 'lime'       # CBA
                }
                
                color = type_colors.get(profile_type, 'gray')
                marker_size = 10 if len(points_data) <= 20 else 8
                
                ax.plot(cart_x, cart_y, 'o', color=color, markersize=marker_size, alpha=0.8)
                
                # Add profile type number or X
                if is_circular:
                    # Show C (cycle) or T (tie)
                    marker = 'T'
                    bbox_color = 'gray'
                    if ranking_weights is not None:
                        _, rel = classify_majority_relation_from_freq(ranking_weights)
                        if rel == "cycle":
                            marker = 'C'
                            bbox_color = 'red'
                        elif rel == "tie":
                            marker = 'T'
                            bbox_color = 'gray'
                    ax.text(cart_x, cart_y, marker, ha='center', va='center',
                           fontsize=9, weight='bold', color='white',
                           bbox=dict(boxstyle='circle', facecolor=bbox_color, alpha=0.7, pad=0.3))
                else:
                    # Show profile type number
                    text_color = 'white' if color in ['blue', 'green', 'cyan', 'lime', 'red'] else 'black'
                    ax.text(cart_x, cart_y, str(profile_type), ha='center', va='center',
                           fontsize=8, weight='bold', color=text_color)
                
                if len(points_data) <= 20:  # Only label if not too many points
                    label = point_data.get('label', f'P{i+1}')
                    ax.text(cart_x + 0.02, cart_y + 0.02, label, fontsize=9)

    # Highlight selected point (overlay so it appears on top)
    global selected_point_idx
    if show_points and selected_point_idx is not None and 0 <= selected_point_idx < len(points_data):
        sel = points_data[selected_point_idx]
        sx, sy = sel['cartesian']
        ax.plot(
            sx,
            sy,
            marker='o',
            markersize=18,
            markerfacecolor='none',
            markeredgecolor='magenta',
            markeredgewidth=3,
            zorder=10,
        )
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel('Barycentric X coordinate', fontsize=10)
    ax.set_ylabel('Barycentric Y coordinate', fontsize=10)
    title = "Saari's Triangle"
    if current_view == 'calculate':
        title += " - Calculate View (with Median Lines)"
    elif current_view == 'statistics':
        title += " - Statistics View"
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

def update_statistics(reset_scroll=False):
    """Update the statistics display with scrolling support."""
    global stats_text_full, stats_scroll_pos
    
    ax_stats.clear()
    ax_stats.axis('off')
    
    if not points_data:
        ax_stats.text(0.5, 0.5, 'No points added yet.\nAdd points to see statistics.', 
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        stats_text_full = ""
        stats_scroll_pos = 0
        return
    
    # Reset scroll position if requested (when new data is calculated)
    if reset_scroll:
        stats_scroll_pos = 0
    
    rule_names = [name for name in VOTING_RULES.keys() if name in selected_voting_rules]
    if not rule_names:
        stats_text = "VOTING RULE STATISTICS\n"
        stats_text += "=" * 60 + "\n\n"
        stats_text += f"Number of Points: {len(points_data)}\n"
        stats_text += f"Voters per Profile: {n_voters_sim}\n"
        stats_text += f"Generation Method: {generation_method}\n"
        stats_text += "Selected Rules: (none)\n\n"
        stats_text += "Select at least one rule to compute outcomes.\n"
        stats_text_full = stats_text
        stats_scroll_pos = 0
        ax_stats.text(0.02, 0.98, stats_text_full, transform=ax_stats.transAxes,
                     fontsize=9, fontfamily='monospace', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        return

    # Calculate outcomes for all points (selected rules only)
    results = calculate_voting_outcomes(points_data, n_voters_sim, generation_method, rule_names=rule_names)
    
    # Count winners for each rule
    stats_text = "VOTING RULE STATISTICS\n"
    stats_text += "=" * 60 + "\n\n"
    stats_text += f"Number of Points: {len(points_data)}\n"
    stats_text += f"Voters per Profile: {n_voters_sim}\n"
    stats_text += f"Generation Method: {generation_method}\n\n"
    stats_text += "-" * 60 + "\n\n"
    stats_text += "AGGREGATE RESULTS (across all points):\n"
    stats_text += "-" * 60 + "\n\n"
    
    for rule_name in rule_names:
        winners = results[rule_name]
        winner_counts = {'A': 0, 'B': 0, 'C': 0, None: 0}
        for winner in winners:
            if winner in winner_counts:
                winner_counts[winner] += 1
            else:
                winner_counts[None] += 1
        
        display_name = rule_name.replace('_', ' ').title()
        stats_text += f"{display_name:<20}:\n"
        for candidate in ['A', 'B', 'C']:
            count = winner_counts[candidate]
            pct = 100 * count / len(winners) if winners else 0
            bar = '█' * int(pct / 2)  # Simple bar chart
            stats_text += f"  {candidate}: {count:>3} ({pct:>5.1f}%) {bar}\n"
        if winner_counts[None] > 0:
            stats_text += f"  None: {winner_counts[None]}\n"
        stats_text += "\n"
    
    # Show per-point results (show all points now since we have scrolling)
    stats_text += "\n" + "=" * 60 + "\n"
    stats_text += f"PER-POINT RESULTS ({len(points_data)} points)\n"
    stats_text += "=" * 60 + "\n\n"
    
    for i, point_data in enumerate(points_data):
        label = point_data.get('label', f'Point {i+1}')
        bary = point_data['barycentric']
        stats_text += f"{label} (x={bary[0]:.3f}, y={bary[1]:.3f}, z={bary[2]:.3f}):\n"
        for rule_name in rule_names:
            winner = results[rule_name][i]
            display_name = rule_name.replace('_', ' ').title()
            stats_text += f"  {display_name:<20}: {winner}\n"
        stats_text += "\n"
    
    # Store full text
    stats_text_full = stats_text
    stats_lines = stats_text_full.split('\n')
    total_lines = len(stats_lines)
    
    # Reset scroll position if it's beyond the text
    if stats_scroll_pos >= total_lines:
        stats_scroll_pos = 0
    
    # Calculate visible range
    start_line = stats_scroll_pos
    end_line = min(start_line + stats_lines_per_page, total_lines)
    
    # Get visible portion
    visible_lines = stats_lines[start_line:end_line]
    visible_text = '\n'.join(visible_lines)
    
    # Add scroll indicator
    if total_lines > stats_lines_per_page:
        scroll_info = f"[Lines {start_line+1}-{end_line} of {total_lines}]"
        visible_text = scroll_info + "\n\n" + visible_text
    
    ax_stats.text(0.02, 0.98, visible_text, transform=ax_stats.transAxes,
                 fontsize=8, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

def update_display(reset_scroll=False):
    """Update the main display based on current view."""
    global stats_scroll_pos
    if current_view == 'triangle':
        draw_triangle(ax_triangle, show_points=True, show_medians_flag=show_medians)
        # Show selected point details in the right panel if a point is selected
        if selected_point_idx is None:
            ax_stats.clear()
            ax_stats.axis('off')
        else:
            update_point_details(selected_point_idx)
    elif current_view == 'calculate':
        draw_triangle(ax_triangle, show_points=True, show_medians_flag=True)
        # Prefer showing point details when user selected a point
        if selected_point_idx is not None:
            update_point_details(selected_point_idx)
        else:
            update_statistics(reset_scroll=reset_scroll)
    elif current_view == 'statistics':
        draw_triangle(ax_triangle, show_points=True, show_medians_flag=False)
        # Prefer showing point details when user selected a point
        if selected_point_idx is not None:
            update_point_details(selected_point_idx)
        else:
            update_statistics(reset_scroll=reset_scroll)
    
    fig.canvas.draw_idle()

# Initial draw
draw_triangle(ax_triangle)


def calculate_point_outcomes(point_data, n_voters, method):
    """Compute winners for all rules (and some diagnostics) for a single point."""
    bary = point_data['barycentric']
    ranking_weights = point_data.get('ranking_weights')
    coord_seed = int((bary[0] * 1000 + bary[1] * 100 + bary[2] * 10) * 1000) % (2**31)
    profile = generate_profile_from_point(bary, n_voters, method, seed=coord_seed, ranking_weights=ranking_weights)

    outcomes = {}
    for rule_name in [name for name in VOTING_RULES.keys() if name in selected_voting_rules]:
        outcomes[rule_name] = VOTING_RULES[rule_name](profile)

    # Display type is based on barycentric region (first-choice shares).
    type_region, _ = get_majority_type_from_barycentric(bary)
    # Circularity (cycle) can be derived from explicit weights, when present.
    is_cycle = has_condorcet_cycle_from_ranking_weights(ranking_weights) if ranking_weights is not None else False
    type_sim, sim_is_circular = get_majority_type_and_cycle(profile)

    winners = [w for w in outcomes.values() if w is not None]
    has_conflict = len(set(winners)) > 1

    return {
        'profile': profile,
        'outcomes': outcomes,
        'type_region': type_region,
        'type_sim': type_sim,
        'sim_is_circular': sim_is_circular,
        'is_cycle_from_weights': is_cycle,
        'has_conflict': has_conflict,
    }


def update_point_details(index):
    """Render details for one selected point into the stats panel."""
    ax_stats.clear()
    ax_stats.axis('off')

    if index is None or not (0 <= index < len(points_data)):
        ax_stats.text(
            0.5, 0.5, 'No point selected.',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
        )
        return

    point_data = points_data[index]
    label = point_data.get('label', f'P{index+1}')
    bary = point_data['barycentric']
    cart = point_data['cartesian']
    ranking_weights = point_data.get('ranking_weights')

    info = calculate_point_outcomes(point_data, n_voters_sim, generation_method)
    outcomes = info['outcomes']

    lines = []
    lines.append("POINT DETAILS (click a point to select, click empty space to clear)")
    lines.append("=" * 68)
    lines.append(f"Label: {label}  (index {index+1} of {len(points_data)})")
    lines.append(f"Barycentric: x={bary[0]:.4f}, y={bary[1]:.4f}, z={bary[2]:.4f}")
    lines.append(f"Cartesian:   x={cart[0]:.4f}, y={cart[1]:.4f}")
    if ranking_weights is not None:
        w = np.asarray(ranking_weights, dtype=np.float64)
        total = float(np.sum(w)) if float(np.sum(w)) > 0 else 1.0
        pct = np.round((w / total) * 100.0, 2)
        lines.append("")
        lines.append("Ranking distribution (%): [ABC, ACB, BAC, BCA, CAB, CBA]")
        lines.append(f"  {pct.tolist()}")
    lines.append("")
    lines.append(f"Type (region): {info['type_region'] if info['type_region'] is not None else 'X'}")
    lines.append(f"Type (sample): {info['type_sim'] if info['type_sim'] is not None else 'X'}")
    lines.append(f"Sample majority: {'cycle/tie' if info['sim_is_circular'] else 'transitive'}")
    if ranking_weights is not None:
        lines.append(f"Condorcet cycle (from vector): {'YES' if info.get('is_cycle_from_weights') else 'no'}")
    lines.append(f"Rule conflict (sample): {'YES' if info['has_conflict'] else 'no'}")
    lines.append("")
    lines.append("Winners by rule (this point's sampled profile):")
    lines.append("-" * 68)
    rule_names = [name for name in VOTING_RULES.keys() if name in selected_voting_rules]
    if not rule_names:
        lines.append("(No rules selected)")
    else:
        for rule_name in rule_names:
            display_name = rule_name.replace('_', ' ').title()
            lines.append(f"{display_name:<22}: {outcomes[rule_name]}")

    text = "\n".join(lines)
    ax_stats.text(
        0.02, 0.98, text, transform=ax_stats.transAxes,
        fontsize=9, fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
    )

# Add point controls
ax_x_input = fig.add_axes([0.05, 0.28, 0.1, 0.03])
textbox_x = TextBox(ax_x_input, 'X: ', initial='0.33')
ax_x_input.set_xlim(0, 1)

ax_y_input = fig.add_axes([0.17, 0.28, 0.1, 0.03])
textbox_y = TextBox(ax_y_input, 'Y: ', initial='0.33')
ax_y_input.set_xlim(0, 1)

ax_add_button = fig.add_axes([0.29, 0.28, 0.08, 0.03])
button_add = Button(ax_add_button, 'Add Point')

ax_clear_button = fig.add_axes([0.39, 0.28, 0.08, 0.03])
button_clear = Button(ax_clear_button, 'Clear All')

ax_generate_button = fig.add_axes([0.49, 0.28, 0.12, 0.03])
button_generate = Button(ax_generate_button, 'Generate Points')

# Simulation controls (moved under point controls)
ax_sim_voters = fig.add_axes([0.05, 0.23, 0.15, 0.025])
slider_voters = Slider(ax_sim_voters, 'Voters', 10, 500, valinit=100, valstep=10)

ax_sim_profiles = fig.add_axes([0.05, 0.18, 0.15, 0.025])
slider_profiles = Slider(ax_sim_profiles, 'Profiles', 10, 1000, valinit=100, valstep=10)

ax_method = fig.add_axes([0.22, 0.15, 0.12, 0.08])
ax_method.set_title('Generation Method', fontsize=9)
radio_method = RadioButtons(ax_method, ['saari', 'uniform', 'weighted'], active=0)

# View controls (moved under simulation controls)
ax_view = fig.add_axes([0.36, 0.15, 0.12, 0.08])
ax_view.set_title('View Mode', fontsize=9)
radio_view = RadioButtons(ax_view, ['triangle', 'calculate', 'statistics'], active=0)

ax_medians = fig.add_axes([0.50, 0.15, 0.12, 0.05])
check_medians = CheckButtons(ax_medians, ['Show Medians'], [False])

# Point generation mode (used by "Generate Points")
ax_pointgen = fig.add_axes([0.50, 0.05, 0.12, 0.08])
ax_pointgen.set_title('Point Gen', fontsize=9)
radio_pointgen = RadioButtons(ax_pointgen, ['coordinates', 'ranking_vector'], active=0)

# Voting rule selection (affects calculation/statistics/conflict borders)
ax_rules = fig.add_axes([0.65, 0.05, 0.32, 0.22])
ax_rules.set_title('Voting Rules (included)', fontsize=9)
_rule_names_ordered = list(VOTING_RULES.keys())
check_rules = CheckButtons(ax_rules, _rule_names_ordered, [True] * len(_rule_names_ordered))

# Scroll buttons for statistics
ax_scroll_up = fig.add_axes([0.96, 0.50, 0.02, 0.05])
button_scroll_up = Button(ax_scroll_up, '▲')

ax_scroll_down = fig.add_axes([0.96, 0.30, 0.02, 0.05])
button_scroll_down = Button(ax_scroll_down, '▼')

def on_add_point(event):
    """Add a point to the triangle."""
    try:
        x = float(textbox_x.text)
        y = float(textbox_y.text)
        
        # Normalize to valid barycentric coordinates
        bary = normalize_barycentric(x, y)
        cart_x, cart_y = barycentric_to_cartesian(bary)
        
        # Add point
        label = f'P{len(points_data) + 1}'
        points_data.append({
            'barycentric': bary,
            'cartesian': (cart_x, cart_y),
            'label': label
        })
        
        update_display(reset_scroll=True)
    except ValueError:
        pass

def on_clear_all(event):
    """Clear all points."""
    global points_data, selected_point_idx
    points_data = []
    selected_point_idx = None
    update_display(reset_scroll=True)

def on_generate_points(event):
    """Generate random points for simulation."""
    global points_data, n_profiles_sim, selected_point_idx, point_generation_mode
    points_data = []
    selected_point_idx = None
    
    def _sample_ranking_percentages(total=100):
        # Dirichlet -> integer percentages summing to total
        p = np.random.dirichlet(np.ones(6))
        raw = p * total
        base = np.floor(raw).astype(int)
        remainder = int(total - np.sum(base))
        if remainder > 0:
            frac = raw - base
            idxs = np.argsort(frac)[::-1][:remainder]
            base[idxs] += 1
        return base

    for i in range(n_profiles_sim):
        if point_generation_mode == 'ranking_vector':
            pct = _sample_ranking_percentages(total=100)  # int percentages
            weights = pct.astype(np.float64) / 100.0
            bary = barycentric_from_ranking_weights(weights)
        else:
            # Generate random barycentric coordinates (rejection sampling)
            while True:
                x = np.random.random()
                y = np.random.random()
                if x + y <= 1:
                    break
            bary = normalize_barycentric(x, y)
            weights = None

        cart_x, cart_y = barycentric_to_cartesian(bary)

        label = f'P{i+1}'
        pd = {
            'barycentric': bary,
            'cartesian': (cart_x, cart_y),
            'label': label
        }
        if weights is not None:
            pd['ranking_weights'] = weights
        points_data.append(pd)
    
    update_display(reset_scroll=True)

def on_voters_change(val):
    """Update number of voters."""
    global n_voters_sim
    n_voters_sim = int(val)
    if current_view in ['calculate', 'statistics']:
        update_display(reset_scroll=True)

def on_profiles_change(val):
    """Update number of profiles."""
    global n_profiles_sim
    n_profiles_sim = int(val)

def on_method_change(label):
    """Change generation method."""
    global generation_method
    generation_method = label
    if current_view in ['calculate', 'statistics']:
        update_display(reset_scroll=True)

def on_view_change(label):
    """Change view mode."""
    global current_view
    current_view = label
    update_display()

def on_medians_change(label):
    """Toggle median lines."""
    global show_medians
    show_medians = check_medians.get_status()[0]
    update_display()

def on_rules_change(label):
    """Toggle which voting rules are included in calculations/statistics."""
    global selected_voting_rules
    if label in selected_voting_rules:
        selected_voting_rules.remove(label)
    else:
        selected_voting_rules.add(label)
    # Refresh if we are showing any computed information
    if current_view in ['calculate', 'statistics'] or selected_point_idx is not None:
        update_display(reset_scroll=True)


def on_pointgen_change(label):
    """Change how the Generate Points button produces points."""
    global point_generation_mode
    point_generation_mode = label

def on_scroll_up(event):
    """Scroll statistics up."""
    global stats_scroll_pos
    if stats_text_full:
        stats_lines = stats_text_full.split('\n')
        total_lines = len(stats_lines)
        stats_scroll_pos = max(0, stats_scroll_pos - stats_lines_per_page // 2)
        if current_view in ['calculate', 'statistics']:
            update_statistics()
            fig.canvas.draw_idle()

def on_scroll_down(event):
    """Scroll statistics down."""
    global stats_scroll_pos
    if stats_text_full:
        stats_lines = stats_text_full.split('\n')
        total_lines = len(stats_lines)
        stats_scroll_pos = min(total_lines - stats_lines_per_page, 
                              stats_scroll_pos + stats_lines_per_page // 2)
        if current_view in ['calculate', 'statistics']:
            update_statistics()
            fig.canvas.draw_idle()


def on_triangle_click(event):
    """Select a point by clicking near it; clear selection if clicking empty space."""
    global selected_point_idx
    if event.inaxes != ax_triangle:
        return
    if not points_data:
        return

    click_xy = np.array([event.x, event.y], dtype=np.float64)  # display (pixel) coords
    pts_xy = np.array([ax_triangle.transData.transform(p['cartesian']) for p in points_data], dtype=np.float64)

    dists = np.sqrt(np.sum((pts_xy - click_xy) ** 2, axis=1))
    idx = int(np.argmin(dists))

    # Pixel threshold for selection; scale a bit with point count
    threshold = 12 if len(points_data) <= 200 else 8
    if dists[idx] <= threshold:
        selected_point_idx = idx
    else:
        selected_point_idx = None

    update_display()


# Connect callbacks
button_add.on_clicked(on_add_point)
button_clear.on_clicked(on_clear_all)
button_generate.on_clicked(on_generate_points)
slider_voters.on_changed(on_voters_change)
slider_profiles.on_changed(on_profiles_change)
radio_method.on_clicked(on_method_change)
radio_view.on_clicked(on_view_change)
check_medians.on_clicked(on_medians_change)
radio_pointgen.on_clicked(on_pointgen_change)
check_rules.on_clicked(on_rules_change)
button_scroll_up.on_clicked(on_scroll_up)
button_scroll_down.on_clicked(on_scroll_down)
fig.canvas.mpl_connect('button_press_event', on_triangle_click)

plt.show()

