"""
STAR Voting Simulator

A comprehensive simulation of STAR (Score Then Automatic Runoff) voting with:
- Single profile editing and visualization
- Multi-profile spatial visualization (Saari triangle or 3D cube)
- VSE (Voter Satisfaction Efficiency) calculation
- PVSI (Pivotal Voter Strategic Incentive) calculation
- Yee diagram visualization
- Honesty percentage control for strategic vs honest voting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# =============================================================================
# Constants
# =============================================================================

CANDIDATES = ['A', 'B', 'C']
N_CANDIDATES = 3
SCORE_MIN = 0
SCORE_MAX = 5

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

# Saari triangle vertices (for spatial model)
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],           # A (bottom-left)
    [1.0, 0.0],           # B (bottom-right)
    [0.5, np.sqrt(3)/2]   # C (top)
])

# Default candidate positions in 2D space (for utility model)
DEFAULT_CANDIDATE_POSITIONS = np.array([
    [0.2, 0.3],   # A
    [0.8, 0.3],   # B
    [0.5, 0.8],   # C
])


# =============================================================================
# STAR Voting Algorithm
# =============================================================================

def star_voting(scores):
    """
    STAR voting: Score Then Automatic Runoff.
    
    Parameters:
        scores: (n_voters, n_candidates) array with values 0-5
    
    Returns:
        winner: int (candidate index)
        details: dict with scoring round and runoff details
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_voters, n_candidates = scores.shape
    
    # Round 1: Scoring - sum all scores
    totals = np.sum(scores, axis=0)
    
    # Find top 2 candidates
    sorted_indices = np.argsort(totals)[::-1]
    finalist_a = sorted_indices[0]
    finalist_b = sorted_indices[1]
    
    # Round 2: Automatic Runoff - who is preferred by more voters
    a_preferred = np.sum(scores[:, finalist_a] > scores[:, finalist_b])
    b_preferred = np.sum(scores[:, finalist_b] > scores[:, finalist_a])
    equal_scores = np.sum(scores[:, finalist_a] == scores[:, finalist_b])
    
    if a_preferred > b_preferred:
        winner = finalist_a
    elif b_preferred > a_preferred:
        winner = finalist_b
    else:
        # Tie in runoff - use total scores as tiebreaker
        winner = finalist_a if totals[finalist_a] >= totals[finalist_b] else finalist_b
    
    details = {
        'totals': totals,
        'finalist_a': finalist_a,
        'finalist_b': finalist_b,
        'a_preferred': a_preferred,
        'b_preferred': b_preferred,
        'equal_scores': equal_scores,
        'winner': winner,
    }
    
    return winner, details


def star_voting_simple(scores):
    """Simple wrapper that just returns the winner."""
    winner, _ = star_voting(scores)
    return CANDIDATES[winner]


# =============================================================================
# Utility Model
# =============================================================================

def compute_utilities(voter_positions, candidate_positions):
    """
    Compute utilities for each voter-candidate pair based on spatial distance.
    
    Utility = -distance (closer = higher utility)
    Normalized to [0, 1] range.
    
    Parameters:
        voter_positions: (n_voters, 2) array
        candidate_positions: (n_candidates, 2) array
    
    Returns:
        utilities: (n_voters, n_candidates) array in [0, 1]
    """
    voter_positions = np.asarray(voter_positions, dtype=np.float64)
    candidate_positions = np.asarray(candidate_positions, dtype=np.float64)
    
    n_voters = voter_positions.shape[0]
    n_candidates = candidate_positions.shape[0]
    
    # Compute distances
    distances = np.zeros((n_voters, n_candidates), dtype=np.float64)
    for c in range(n_candidates):
        diff = voter_positions - candidate_positions[c]
        distances[:, c] = np.sqrt(np.sum(diff**2, axis=1))
    
    # Convert distance to utility (negative distance, then normalize)
    # Max possible distance in unit square is sqrt(2)
    max_dist = np.sqrt(2.0)
    utilities = 1.0 - (distances / max_dist)
    utilities = np.clip(utilities, 0.0, 1.0)
    
    return utilities


def utility_to_score(utility, score_min=SCORE_MIN, score_max=SCORE_MAX):
    """Convert utility [0, 1] to score [score_min, score_max]."""
    return score_min + utility * (score_max - score_min)


def generate_voter_positions(n_voters, center=(0.5, 0.5), spread=0.3, seed=None):
    """
    Generate voter positions in 2D space using Gaussian distribution.
    
    Parameters:
        n_voters: number of voters
        center: (x, y) center of voter distribution
        spread: standard deviation of distribution
        seed: random seed
    
    Returns:
        positions: (n_voters, 2) array
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.random.normal(loc=center, scale=spread, size=(n_voters, 2))
    positions = np.clip(positions, 0.0, 1.0)
    
    return positions


def generate_voter_positions_uniform(n_voters, seed=None):
    """Generate uniformly distributed voter positions."""
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.random((n_voters, 2))


def generate_voter_positions_clustered(n_voters, candidate_positions, seed=None):
    """
    Generate voters clustered near candidates.
    Each voter is placed near a randomly chosen candidate.
    Creates polarized electorates where strategic voting matters.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_candidates = candidate_positions.shape[0]
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    for v in range(n_voters):
        # Pick a random candidate to be near
        c = np.random.randint(0, n_candidates)
        # Generate position near that candidate
        spread = 0.15
        pos = np.random.normal(loc=candidate_positions[c], scale=spread)
        positions[v] = np.clip(pos, 0.0, 1.0)
    
    return positions


def generate_voter_positions_polarized(n_voters, candidate_positions, seed=None):
    """
    Generate a polarized electorate with two major factions.
    One faction supports candidate A, another supports B.
    This creates scenarios where C might win as a compromise,
    and strategic voting becomes important.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    # Faction sizes (randomized slightly)
    faction_a_size = int(n_voters * np.random.uniform(0.35, 0.50))
    faction_b_size = int(n_voters * np.random.uniform(0.35, 0.50))
    faction_c_size = n_voters - faction_a_size - faction_b_size
    
    idx = 0
    
    # Faction A (near candidate A)
    for _ in range(faction_a_size):
        pos = np.random.normal(loc=candidate_positions[0], scale=0.12)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    # Faction B (near candidate B)
    for _ in range(faction_b_size):
        pos = np.random.normal(loc=candidate_positions[1], scale=0.12)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    # Faction C / swing voters (near candidate C or center)
    for _ in range(faction_c_size):
        pos = np.random.normal(loc=candidate_positions[2], scale=0.15)
        positions[idx] = np.clip(pos, 0.0, 1.0)
        idx += 1
    
    # Shuffle to mix factions
    np.random.shuffle(positions)
    
    return positions


def generate_voter_positions_weighted(n_voters, candidate_positions, weights=None, seed=None):
    """
    Generate voters with weighted distribution toward candidates.
    
    Parameters:
        weights: (n_candidates,) array of weights for each candidate.
                 Higher weight = more voters near that candidate.
                 If None, uses random weights.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_candidates = candidate_positions.shape[0]
    
    if weights is None:
        # Random weights (Dirichlet distribution for variety)
        weights = np.random.dirichlet(np.ones(n_candidates) * 2)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
    
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    
    for v in range(n_voters):
        # Choose candidate based on weights
        c = np.random.choice(n_candidates, p=weights)
        # Generate position near that candidate with some spread
        spread = 0.18
        pos = np.random.normal(loc=candidate_positions[c], scale=spread)
        positions[v] = np.clip(pos, 0.0, 1.0)
    
    return positions


def generate_voter_positions_saari(n_voters, barycentric_center, seed=None):
    """
    Generate voters in Saari-style distribution.
    Voters are clustered around a barycentric center point.
    
    Parameters:
        barycentric_center: (3,) tuple (a, b, c) where a+b+c=1
    """
    if seed is not None:
        np.random.seed(seed)
    
    a, b, c = barycentric_center
    
    # Convert barycentric center to Cartesian (in [0,1]^2)
    center_x = b + 0.5 * c  # Simplified mapping
    center_y = c * np.sqrt(3) / 2
    
    # Normalize to [0,1]
    center_x = center_x / 1.0
    center_y = center_y / (np.sqrt(3) / 2)
    center = np.array([center_x, center_y])
    
    positions = np.zeros((n_voters, 2), dtype=np.float64)
    spread = 0.2
    
    for v in range(n_voters):
        pos = np.random.normal(loc=center, scale=spread)
        positions[v] = np.clip(pos, 0.0, 1.0)
    
    return positions


# Generation method registry
GENERATION_METHODS = {
    'uniform': 'uniform',
    'clustered': 'clustered',
    'polarized': 'polarized',
    'weighted': 'weighted',
    'saari': 'saari',
}


# =============================================================================
# Honest vs Strategic Scoring
# =============================================================================

def honest_scores(utilities):
    """
    Generate honest scores based on utilities.
    Score = utility mapped to 0-5 scale.
    """
    return np.round(utility_to_score(utilities)).astype(np.int32)


def strategic_scores(utilities, candidate_positions=None):
    """
    Generate strategic scores using min-max strategy.
    
    Strategy:
    - Give max score (5) to most preferred candidate
    - Give min score (0) to least preferred candidate
    - For middle candidate, score based on relative position
    
    This is a simplified strategic voting model.
    """
    n_voters, n_candidates = utilities.shape
    scores = np.zeros((n_voters, n_candidates), dtype=np.int32)
    
    for v in range(n_voters):
        u = utilities[v]
        sorted_idx = np.argsort(u)[::-1]  # Best to worst
        
        # Give max score to favorite
        scores[v, sorted_idx[0]] = SCORE_MAX
        # Give min score to least favorite
        scores[v, sorted_idx[-1]] = SCORE_MIN
        
        # For middle candidates, use normalized utility
        for i in range(1, n_candidates - 1):
            c = sorted_idx[i]
            # Scale between min and max based on relative utility
            u_range = u[sorted_idx[0]] - u[sorted_idx[-1]]
            if u_range > 0:
                rel_u = (u[c] - u[sorted_idx[-1]]) / u_range
                scores[v, c] = int(np.round(rel_u * SCORE_MAX))
            else:
                scores[v, c] = SCORE_MAX // 2
    
    return scores


def generate_scores(utilities, honesty_pct=100):
    """
    Generate scores based on utilities and honesty percentage.
    
    Parameters:
        utilities: (n_voters, n_candidates) array
        honesty_pct: 0-100, percentage of voters who vote honestly
    
    Returns:
        scores: (n_voters, n_candidates) array of integer scores 0-5
    """
    n_voters = utilities.shape[0]
    
    # Get both honest and strategic scores
    h_scores = honest_scores(utilities)
    s_scores = strategic_scores(utilities)
    
    # Randomly select which voters are honest
    n_honest = int(n_voters * honesty_pct / 100.0)
    honest_mask = np.zeros(n_voters, dtype=bool)
    honest_indices = np.random.choice(n_voters, size=n_honest, replace=False)
    honest_mask[honest_indices] = True
    
    # Combine scores
    scores = np.where(honest_mask[:, np.newaxis], h_scores, s_scores)
    
    return scores.astype(np.int32)


# =============================================================================
# VSE (Voter Satisfaction Efficiency)
# =============================================================================

def compute_vse(scores, utilities, winner_idx):
    """
    Compute Voter Satisfaction Efficiency.
    
    VSE = (U_elected - U_random) / (U_optimal - U_random)
    
    Parameters:
        scores: (n_voters, n_candidates) score matrix
        utilities: (n_voters, n_candidates) utility matrix
        winner_idx: index of elected candidate
    
    Returns:
        vse: float in [0, 1] (can be negative if worse than random)
    """
    n_voters, n_candidates = utilities.shape
    
    # Average utility of elected candidate
    u_elected = np.mean(utilities[:, winner_idx])
    
    # Average utility of optimal candidate (best average utility)
    avg_utilities = np.mean(utilities, axis=0)
    optimal_idx = np.argmax(avg_utilities)
    u_optimal = avg_utilities[optimal_idx]
    
    # Expected utility of random selection
    u_random = np.mean(avg_utilities)
    
    # VSE calculation
    if abs(u_optimal - u_random) < 1e-10:
        # All candidates have same average utility
        return 1.0 if abs(u_elected - u_optimal) < 1e-10 else 0.0
    
    vse = (u_elected - u_random) / (u_optimal - u_random)
    
    return vse


# =============================================================================
# PVSI (Pivotal Voter Strategic Incentive)
# =============================================================================

def compute_pvsi(scores, utilities, candidate_positions=None):
    """
    Compute Pivotal Voter Strategic Incentive.
    
    PVSI measures strategic voting incentive using multiple metrics:
    1. Pivotal voters: proportion who can change outcome by voting strategically
    2. Score manipulation potential: how much voters could exaggerate scores
    3. Runoff sensitivity: how close the runoff is
    
    Parameters:
        scores: (n_voters, n_candidates) current score matrix
        utilities: (n_voters, n_candidates) utility matrix
        candidate_positions: optional, for strategic scoring
    
    Returns:
        pvsi: float in [0, 1], combined strategic incentive measure
    """
    n_voters, n_candidates = scores.shape
    
    # Current winner and details
    current_winner, details = star_voting(scores)
    
    # Component 1: Pivotal voters (can change outcome alone)
    pivotal_count = 0
    for v in range(n_voters):
        current_utility = utilities[v, current_winner]
        test_scores = scores.copy()
        v_utilities = utilities[v:v+1, :]
        strategic = strategic_scores(v_utilities)[0]
        test_scores[v] = strategic
        new_winner, _ = star_voting(test_scores)
        if new_winner != current_winner and utilities[v, new_winner] > current_utility:
            pivotal_count += 1
    
    pivotal_rate = pivotal_count / n_voters if n_voters > 0 else 0.0
    
    # Component 2: Runoff closeness (close runoffs encourage strategic voting)
    runoff_total = details['a_preferred'] + details['b_preferred']
    if runoff_total > 0:
        runoff_margin = abs(details['a_preferred'] - details['b_preferred']) / runoff_total
        runoff_closeness = 1.0 - runoff_margin  # 1.0 = perfect tie, 0.0 = landslide
    else:
        runoff_closeness = 0.0
    
    # Component 3: Scoring round closeness (close scores encourage bullet voting)
    totals = details['totals']
    sorted_totals = np.sort(totals)[::-1]
    max_total = sorted_totals[0]
    second_total = sorted_totals[1]
    if max_total > 0:
        scoring_margin = (max_total - second_total) / max_total
        scoring_closeness = 1.0 - scoring_margin
    else:
        scoring_closeness = 0.0
    
    # Component 4: Score exaggeration potential
    # Measure how much voters could gain by bullet voting
    honest = honest_scores(utilities)
    strategic = strategic_scores(utilities)
    
    # Average absolute difference between honest and strategic scores
    exaggeration = np.mean(np.abs(strategic - honest)) / SCORE_MAX
    
    # Combined PVSI: weighted average of components
    pvsi = (
        0.4 * pivotal_rate +           # Pivotal voters are most important
        0.2 * runoff_closeness +       # Close runoffs encourage strategy
        0.2 * scoring_closeness +      # Close scoring rounds too
        0.2 * exaggeration             # Potential for score manipulation
    )
    
    return pvsi


# =============================================================================
# Profile Generation for Spatial Models
# =============================================================================

def barycentric_to_cartesian(bary, vertices=TRIANGLE_VERTICES):
    """Convert barycentric coordinates to Cartesian."""
    x, y, z = bary
    cart_x = vertices[0][0] * x + vertices[1][0] * y + vertices[2][0] * z
    cart_y = vertices[0][1] * x + vertices[1][1] * y + vertices[2][1] * z
    return cart_x, cart_y


def cartesian_to_barycentric(cart_x, cart_y, vertices=TRIANGLE_VERTICES):
    """Convert Cartesian to barycentric coordinates."""
    v0, v1, v2 = vertices
    
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-10:
        return None
    
    a = ((v1[1] - v2[1]) * (cart_x - v2[0]) + (v2[0] - v1[0]) * (cart_y - v2[1])) / denom
    b = ((v2[1] - v0[1]) * (cart_x - v2[0]) + (v0[0] - v2[0]) * (cart_y - v2[1])) / denom
    c = 1 - a - b
    
    return (a, b, c)


def generate_profile_from_barycentric(bary, n_voters, candidate_positions, honesty_pct=100, seed=None):
    """
    Generate a score profile from barycentric coordinates.
    
    The barycentric position determines the center of voter distribution.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert barycentric to voter center position
    center = barycentric_to_cartesian(bary)
    
    # Generate voter positions around this center
    spread = 0.15
    voter_positions = generate_voter_positions(n_voters, center=center, spread=spread)
    
    # Compute utilities
    utilities = compute_utilities(voter_positions, candidate_positions)
    
    # Generate scores with honesty parameter
    scores = generate_scores(utilities, honesty_pct)
    
    return scores, utilities, voter_positions


def generate_random_profile(n_voters, candidate_positions, honesty_pct=100, 
                            generation_method='uniform', seed=None):
    """
    Generate a random profile with the specified voter distribution method.
    
    Parameters:
        n_voters: number of voters
        candidate_positions: (n_candidates, 2) array
        honesty_pct: 0-100, percentage of honest voters
        generation_method: 'uniform', 'clustered', 'polarized', 'weighted', 'saari'
        seed: random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    if generation_method == 'uniform':
        voter_positions = generate_voter_positions_uniform(n_voters)
    elif generation_method == 'clustered':
        voter_positions = generate_voter_positions_clustered(n_voters, candidate_positions)
    elif generation_method == 'polarized':
        voter_positions = generate_voter_positions_polarized(n_voters, candidate_positions)
    elif generation_method == 'weighted':
        voter_positions = generate_voter_positions_weighted(n_voters, candidate_positions)
    elif generation_method == 'saari':
        # Random barycentric center
        while True:
            a = np.random.random()
            b = np.random.random()
            if a + b <= 1:
                break
        c = 1.0 - a - b
        voter_positions = generate_voter_positions_saari(n_voters, (a, b, c))
    else:
        voter_positions = generate_voter_positions_uniform(n_voters)
    
    utilities = compute_utilities(voter_positions, candidate_positions)
    scores = generate_scores(utilities, honesty_pct)
    
    return scores, utilities, voter_positions


# =============================================================================
# Yee Diagram Generation
# =============================================================================

def generate_yee_diagram(candidate_positions, n_voters_per_point=51, honesty_pct=100, 
                         resolution=50, gen_method='clustered', seed=None):
    """
    Generate a Yee diagram.
    
    For each point in a 2D grid, simulate an election with voters centered there.
    
    Parameters:
        candidate_positions: (n_candidates, 2) array
        n_voters_per_point: voters per grid point
        honesty_pct: honesty percentage
        resolution: grid resolution (resolution x resolution)
        gen_method: voter generation method
        seed: random seed
    
    Returns:
        winners: (resolution, resolution) array of winner indices
        x_coords: (resolution,) array of x coordinates
        y_coords: (resolution,) array of y coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_coords = np.linspace(0.05, 0.95, resolution)
    y_coords = np.linspace(0.05, 0.95, resolution)
    
    winners = np.zeros((resolution, resolution), dtype=np.int32)
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Generate voters centered at (x, y)
            # For Yee diagram, we always use centered distribution
            voter_positions = generate_voter_positions(
                n_voters_per_point, center=(x, y), spread=0.12
            )
            
            utilities = compute_utilities(voter_positions, candidate_positions)
            scores = generate_scores(utilities, honesty_pct)
            
            winner, _ = star_voting(scores)
            winners[j, i] = winner  # Note: j,i for proper orientation
    
    return winners, x_coords, y_coords


# =============================================================================
# Global State
# =============================================================================

# View mode
current_view = 'single'  # 'single', 'spatial', 'yee'
spatial_mode = 'triangle'  # 'triangle' or 'cube'

# Simulation parameters
n_voters_single = 21
n_voters_spatial = 51
n_profiles_spatial = 50
honesty_pct = 100
yee_resolution = 40
generation_method = 'clustered'  # 'uniform', 'clustered', 'polarized', 'weighted', 'saari'

# Candidate positions (can be moved interactively)
candidate_positions = DEFAULT_CANDIDATE_POSITIONS.copy()

# Current profile data (single view)
current_scores = None
current_utilities = None
current_voter_positions = None

# Score matrix edit mode
score_matrix_data = None  # Editable score matrix as list of lists

# Spatial view data
spatial_points = []  # List of profile data dicts

# Selected point index
selected_point_idx = None


# =============================================================================
# UI Setup
# =============================================================================

fig = plt.figure(figsize=(16, 10))
fig.suptitle("STAR Voting Simulator", fontsize=16, weight='bold')

# Main visualization area
ax_main = fig.add_axes([0.05, 0.30, 0.45, 0.60])

# Statistics/results area
ax_stats = fig.add_axes([0.55, 0.30, 0.42, 0.60])
ax_stats.axis('off')


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_single_view():
    """Draw the single profile view with voter/candidate positions and score matrix."""
    global current_scores, current_utilities, current_voter_positions
    
    ax_main.clear()
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.set_aspect('equal')
    ax_main.set_title("Voter-Candidate Space (Single Profile)", fontsize=12)
    ax_main.set_xlabel("X position")
    ax_main.set_ylabel("Y position")
    ax_main.grid(True, alpha=0.3)
    
    # Draw voters if we have them
    if current_voter_positions is not None and current_scores is not None:
        # Color voters by their preferred candidate (highest score)
        preferred = np.argmax(current_scores, axis=1)
        voter_colors = [CANDIDATE_COLORS[p] for p in preferred]
        ax_main.scatter(
            current_voter_positions[:, 0], 
            current_voter_positions[:, 1],
            c=voter_colors, s=40, alpha=0.6, edgecolors='white', linewidths=0.5
        )
    
    # Draw candidates
    for c in range(N_CANDIDATES):
        ax_main.scatter(
            candidate_positions[c, 0], 
            candidate_positions[c, 1],
            c=CANDIDATE_COLORS[c], s=300, marker='*', 
            edgecolors='black', linewidths=2,
            label=f'Candidate {CANDIDATES[c]}', zorder=10
        )
        ax_main.annotate(
            CANDIDATES[c], 
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=14, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax_main.legend(loc='upper right', fontsize=8)
    
    # Add score matrix visualization in corner if we have scores
    if current_scores is not None:
        draw_score_matrix_inset(ax_main)


def draw_score_matrix_inset(ax):
    """Draw a small score matrix visualization in the corner of the plot."""
    if current_scores is None:
        return
    
    n_voters = min(10, current_scores.shape[0])  # Show first 10 voters
    
    # Create inset text showing score matrix summary
    lines = ["Score Matrix (first 10 voters):"]
    lines.append("Voter   A   B   C")
    lines.append("-" * 20)
    for v in range(n_voters):
        scores_v = current_scores[v]
        lines.append(f"  {v+1:2d}    {scores_v[0]}   {scores_v[1]}   {scores_v[2]}")
    if current_scores.shape[0] > 10:
        lines.append(f"  ... ({current_scores.shape[0] - 10} more)")
    
    text = '\n'.join(lines)
    ax.text(0.02, 0.02, text, transform=ax.transAxes,
            fontsize=7, fontfamily='monospace', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def draw_spatial_triangle():
    """Draw the Saari triangle spatial view."""
    ax_main.clear()
    ax_main.set_aspect('equal')
    ax_main.set_title("Saari Triangle - STAR Voting Profiles", fontsize=12)
    
    # Draw triangle outline
    triangle = Polygon(TRIANGLE_VERTICES, closed=True, fill=False,
                       edgecolor='black', linewidth=2)
    ax_main.add_patch(triangle)
    
    # Label vertices
    ax_main.text(TRIANGLE_VERTICES[0][0] - 0.05, TRIANGLE_VERTICES[0][1] - 0.05,
                'A', fontsize=12, ha='right', weight='bold', color=CANDIDATE_COLORS[0])
    ax_main.text(TRIANGLE_VERTICES[1][0] + 0.05, TRIANGLE_VERTICES[1][1] - 0.05,
                'B', fontsize=12, ha='left', weight='bold', color=CANDIDATE_COLORS[1])
    ax_main.text(TRIANGLE_VERTICES[2][0], TRIANGLE_VERTICES[2][1] + 0.05,
                'C', fontsize=12, ha='center', weight='bold', color=CANDIDATE_COLORS[2])
    
    # Draw profile points
    if spatial_points:
        for i, point_data in enumerate(spatial_points):
            cart_x, cart_y = point_data['cartesian']
            winner = point_data['winner']
            color = CANDIDATE_COLORS.get(winner, 'gray')
            
            marker_size = 8 if len(spatial_points) > 100 else 12
            edge_width = 2 if i == selected_point_idx else 0.5
            edge_color = 'magenta' if i == selected_point_idx else 'black'
            
            ax_main.plot(cart_x, cart_y, 'o', color=color, markersize=marker_size,
                        markeredgecolor=edge_color, markeredgewidth=edge_width, alpha=0.8)
    
    ax_main.set_xlim(-0.1, 1.1)
    ax_main.set_ylim(-0.1, 1.0)
    ax_main.grid(True, alpha=0.3)


def draw_spatial_cube():
    """Draw the 3D cube spatial view using pairwise margins."""
    ax_main.clear()
    ax_main.set_aspect('equal')
    ax_main.set_title("3D Cube Projection (A-B vs B-C margins)", fontsize=12)
    
    # Project 3D margins to 2D for visualization
    # Use margins: m_AB (x-axis) and m_BC (y-axis)
    
    # Draw cube outline (projected to 2D)
    ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Draw boundary box
    ax_main.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=1, alpha=0.5)
    
    # Label axes
    ax_main.set_xlabel("m_AB (A vs B margin)", fontsize=10)
    ax_main.set_ylabel("m_BC (B vs C margin)", fontsize=10)
    
    # Draw profile points
    if spatial_points:
        for i, point_data in enumerate(spatial_points):
            # Compute pairwise margins from scores
            scores = point_data['scores']
            n_voters = scores.shape[0]
            
            # Margin A vs B: proportion who score A > B minus proportion who score B > A
            a_beats_b = np.sum(scores[:, 0] > scores[:, 1])
            b_beats_a = np.sum(scores[:, 1] > scores[:, 0])
            m_ab = (a_beats_b - b_beats_a) / n_voters
            
            # Margin B vs C
            b_beats_c = np.sum(scores[:, 1] > scores[:, 2])
            c_beats_b = np.sum(scores[:, 2] > scores[:, 1])
            m_bc = (b_beats_c - c_beats_b) / n_voters
            
            winner = point_data['winner']
            color = CANDIDATE_COLORS.get(winner, 'gray')
            
            marker_size = 8 if len(spatial_points) > 100 else 12
            edge_width = 2 if i == selected_point_idx else 0.5
            edge_color = 'magenta' if i == selected_point_idx else 'black'
            
            ax_main.plot(m_ab, m_bc, 'o', color=color, markersize=marker_size,
                        markeredgecolor=edge_color, markeredgewidth=edge_width, alpha=0.8)
    
    ax_main.set_xlim(-1.1, 1.1)
    ax_main.set_ylim(-1.1, 1.1)
    ax_main.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CANDIDATE_COLORS[0], label='A wins'),
        Patch(facecolor=CANDIDATE_COLORS[1], label='B wins'),
        Patch(facecolor=CANDIDATE_COLORS[2], label='C wins'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper right')


def draw_yee_diagram():
    """Draw the Yee diagram."""
    ax_main.clear()
    ax_main.set_title(f"Yee Diagram (Honesty: {honesty_pct}%)", fontsize=12)
    
    # Generate Yee diagram
    winners, x_coords, y_coords = generate_yee_diagram(
        candidate_positions, 
        n_voters_per_point=n_voters_spatial,
        honesty_pct=honesty_pct,
        resolution=yee_resolution,
        seed=42
    )
    
    # Create color map
    cmap_colors = [CANDIDATE_COLORS[0], CANDIDATE_COLORS[1], CANDIDATE_COLORS[2]]
    cmap = mcolors.ListedColormap(cmap_colors)
    
    # Plot as image
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    ax_main.imshow(winners, origin='lower', extent=extent, cmap=cmap, 
                   vmin=0, vmax=2, aspect='equal', alpha=0.8)
    
    # Draw candidate positions
    for c in range(N_CANDIDATES):
        ax_main.scatter(
            candidate_positions[c, 0], 
            candidate_positions[c, 1],
            c='white', s=300, marker='*', 
            edgecolors='black', linewidths=2, zorder=10
        )
        ax_main.annotate(
            CANDIDATES[c], 
            (candidate_positions[c, 0], candidate_positions[c, 1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=14, weight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
    
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("X position")
    ax_main.set_ylabel("Y position")
    ax_main.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CANDIDATE_COLORS[0], label='A wins'),
        Patch(facecolor=CANDIDATE_COLORS[1], label='B wins'),
        Patch(facecolor=CANDIDATE_COLORS[2], label='C wins'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper right')


def update_single_stats():
    """Update statistics display for single profile view."""
    ax_stats.clear()
    ax_stats.axis('off')
    
    if current_scores is None:
        ax_stats.text(0.5, 0.5, 'Click "Generate Profile" to start.\n\nYou can also drag candidates\nto reposition them.',
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                     transform=ax_stats.transAxes)
        return
    
    # Run STAR voting
    winner, details = star_voting(current_scores)
    
    # Compute metrics
    vse = compute_vse(current_scores, current_utilities, winner)
    pvsi = compute_pvsi(current_scores, current_utilities, candidate_positions)
    
    # Build statistics text
    lines = []
    lines.append("STAR VOTING RESULTS")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Number of Voters: {current_scores.shape[0]}")
    lines.append(f"Honesty: {honesty_pct}%")
    lines.append(f"Distribution: {generation_method}")
    lines.append("")
    lines.append("SCORING ROUND (Total Scores)")
    lines.append("-" * 50)
    for c in range(N_CANDIDATES):
        total = details['totals'][c]
        bar_len = int(total / (SCORE_MAX * current_scores.shape[0]) * 30)
        bar = '█' * bar_len
        lines.append(f"  {CANDIDATES[c]}: {total:>6.0f} {bar}")
    
    lines.append("")
    lines.append("AUTOMATIC RUNOFF")
    lines.append("-" * 50)
    a_idx = details['finalist_a']
    b_idx = details['finalist_b']
    lines.append(f"  Finalists: {CANDIDATES[a_idx]} vs {CANDIDATES[b_idx]}")
    lines.append(f"  {CANDIDATES[a_idx]} preferred by: {details['a_preferred']} voters")
    lines.append(f"  {CANDIDATES[b_idx]} preferred by: {details['b_preferred']} voters")
    lines.append(f"  Equal scores: {details['equal_scores']} voters")
    
    lines.append("")
    lines.append(f"  WINNER: {CANDIDATES[winner]}")
    lines.append("")
    lines.append("METRICS")
    lines.append("-" * 50)
    lines.append(f"  VSE (Voter Satisfaction Efficiency): {vse:.3f}")
    lines.append(f"  PVSI (Pivotal Voter Strategic Incentive): {pvsi:.3f}")
    lines.append("")
    lines.append("Score Distribution (per candidate):")
    for c in range(N_CANDIDATES):
        scores_c = current_scores[:, c]
        lines.append(f"  {CANDIDATES[c]}: mean={np.mean(scores_c):.2f}, "
                    f"min={np.min(scores_c)}, max={np.max(scores_c)}")
    
    text = '\n'.join(lines)
    ax_stats.text(0.02, 0.98, text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_spatial_stats():
    """Update statistics display for spatial view."""
    ax_stats.clear()
    ax_stats.axis('off')
    
    if not spatial_points:
        ax_stats.text(0.5, 0.5, 'Click "Generate Points" to start.',
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                     transform=ax_stats.transAxes)
        return
    
    # Compute aggregate statistics
    winner_counts = {0: 0, 1: 0, 2: 0}
    vse_values = []
    pvsi_values = []
    
    for point_data in spatial_points:
        winner_counts[point_data['winner']] += 1
        vse_values.append(point_data['vse'])
        pvsi_values.append(point_data['pvsi'])
    
    total = len(spatial_points)
    
    lines = []
    lines.append("SPATIAL SIMULATION STATISTICS")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Number of Profiles: {total}")
    lines.append(f"Voters per Profile: {n_voters_spatial}")
    lines.append(f"Honesty: {honesty_pct}%")
    lines.append(f"Distribution: {generation_method}")
    lines.append(f"Spatial Mode: {spatial_mode.title()}")
    lines.append("")
    lines.append("WINNER DISTRIBUTION")
    lines.append("-" * 50)
    
    for c in range(N_CANDIDATES):
        count = winner_counts[c]
        pct = 100 * count / total if total > 0 else 0
        bar_len = int(pct / 3)
        bar = '█' * bar_len
        lines.append(f"  {CANDIDATES[c]}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    lines.append("")
    lines.append("VSE STATISTICS")
    lines.append("-" * 50)
    lines.append(f"  Mean VSE: {np.mean(vse_values):.3f}")
    lines.append(f"  Std VSE:  {np.std(vse_values):.3f}")
    lines.append(f"  Min VSE:  {np.min(vse_values):.3f}")
    lines.append(f"  Max VSE:  {np.max(vse_values):.3f}")
    
    lines.append("")
    lines.append("PVSI STATISTICS")
    lines.append("-" * 50)
    lines.append(f"  Mean PVSI: {np.mean(pvsi_values):.3f}")
    lines.append(f"  Std PVSI:  {np.std(pvsi_values):.3f}")
    lines.append(f"  Min PVSI:  {np.min(pvsi_values):.3f}")
    lines.append(f"  Max PVSI:  {np.max(pvsi_values):.3f}")
    
    # Show selected point details if any
    if selected_point_idx is not None and 0 <= selected_point_idx < len(spatial_points):
        point = spatial_points[selected_point_idx]
        lines.append("")
        lines.append("SELECTED POINT DETAILS")
        lines.append("-" * 50)
        lines.append(f"  Point: {selected_point_idx + 1} of {total}")
        lines.append(f"  Winner: {CANDIDATES[point['winner']]}")
        lines.append(f"  VSE: {point['vse']:.3f}")
        lines.append(f"  PVSI: {point['pvsi']:.3f}")
    
    text = '\n'.join(lines)
    ax_stats.text(0.02, 0.98, text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_yee_stats():
    """Update statistics for Yee diagram view."""
    ax_stats.clear()
    ax_stats.axis('off')
    
    lines = []
    lines.append("YEE DIAGRAM")
    lines.append("=" * 50)
    lines.append("")
    lines.append("A Yee diagram shows which candidate")
    lines.append("would win if voters were centered")
    lines.append("at each point in the space.")
    lines.append("")
    lines.append(f"Resolution: {yee_resolution} x {yee_resolution}")
    lines.append(f"Voters per point: {n_voters_spatial}")
    lines.append(f"Honesty: {honesty_pct}%")
    lines.append("")
    lines.append("Candidate Positions:")
    lines.append("-" * 50)
    for c in range(N_CANDIDATES):
        x, y = candidate_positions[c]
        lines.append(f"  {CANDIDATES[c]}: ({x:.2f}, {y:.2f})")
    lines.append("")
    lines.append("TIP: The colored regions show where")
    lines.append("each candidate would win. Larger")
    lines.append("regions indicate stronger support.")
    
    text = '\n'.join(lines)
    ax_stats.text(0.02, 0.98, text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_display():
    """Update the display based on current view."""
    if current_view == 'single':
        draw_single_view()
        update_single_stats()
    elif current_view == 'spatial':
        if spatial_mode == 'triangle':
            draw_spatial_triangle()
        else:
            draw_spatial_cube()
        update_spatial_stats()
    elif current_view == 'yee':
        draw_yee_diagram()
        update_yee_stats()
    
    fig.canvas.draw_idle()


# Initial draw
draw_single_view()
update_single_stats()


# =============================================================================
# UI Controls
# =============================================================================

# View mode selector
ax_view = fig.add_axes([0.05, 0.20, 0.12, 0.08])
ax_view.set_title('View Mode', fontsize=9)
radio_view = RadioButtons(ax_view, ['single', 'spatial', 'yee'], active=0)

# Spatial mode selector
ax_spatial = fig.add_axes([0.19, 0.20, 0.10, 0.06])
ax_spatial.set_title('Spatial', fontsize=9)
radio_spatial = RadioButtons(ax_spatial, ['triangle', 'cube'], active=0)

# Generate button
ax_generate = fig.add_axes([0.31, 0.22, 0.12, 0.04])
button_generate = Button(ax_generate, 'Generate')

# Clear button
ax_clear = fig.add_axes([0.45, 0.22, 0.08, 0.04])
button_clear = Button(ax_clear, 'Clear')

# Edit Scores button (for single view)
ax_edit = fig.add_axes([0.55, 0.22, 0.10, 0.04])
button_edit = Button(ax_edit, 'Edit Scores')

# Voters slider
ax_voters = fig.add_axes([0.05, 0.12, 0.20, 0.025])
slider_voters = Slider(ax_voters, 'Voters', 11, 201, valinit=51, valstep=2)

# Profiles slider (for spatial view)
ax_profiles = fig.add_axes([0.30, 0.12, 0.20, 0.025])
slider_profiles = Slider(ax_profiles, 'Profiles', 10, 200, valinit=50, valstep=10)

# Honesty slider
ax_honesty = fig.add_axes([0.05, 0.06, 0.20, 0.025])
slider_honesty = Slider(ax_honesty, 'Honesty %', 0, 100, valinit=100, valstep=5)

# Yee resolution slider
ax_yee_res = fig.add_axes([0.30, 0.06, 0.20, 0.025])
slider_yee_res = Slider(ax_yee_res, 'Yee Res', 20, 80, valinit=40, valstep=5)

# Generation method selector
ax_gen_method = fig.add_axes([0.67, 0.15, 0.12, 0.12])
ax_gen_method.set_title('Voter Distribution', fontsize=9)
radio_gen_method = RadioButtons(ax_gen_method, ['uniform', 'clustered', 'polarized', 'weighted', 'saari'], active=1)

# Info text
ax_info = fig.add_axes([0.55, 0.05, 0.10, 0.10])
ax_info.axis('off')
info_text = ax_info.text(0.0, 1.0, 
    "STAR Voting\n"
    "VSE: higher=better\n"
    "PVSI: lower=better\n\n"
    "Try 'polarized'\n"
    "for strategic\n"
    "voting effects",
    fontsize=8, fontfamily='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))


# =============================================================================
# Callbacks
# =============================================================================

def on_view_change(label):
    """Handle view mode change."""
    global current_view
    current_view = label
    update_display()


def on_spatial_change(label):
    """Handle spatial mode change."""
    global spatial_mode
    spatial_mode = label
    if current_view == 'spatial':
        update_display()


def on_generate(event):
    """Handle generate button."""
    global current_scores, current_utilities, current_voter_positions
    global spatial_points, selected_point_idx
    
    if current_view == 'single':
        # Generate single profile with selected method
        scores, utilities, voter_pos = generate_random_profile(
            n_voters_single, candidate_positions, honesty_pct,
            generation_method=generation_method
        )
        current_scores = scores
        current_utilities = utilities
        current_voter_positions = voter_pos
    
    elif current_view == 'spatial':
        # Generate multiple profiles in spatial model
        spatial_points = []
        selected_point_idx = None
        
        for i in range(n_profiles_spatial):
            # Random barycentric coordinates (for position in triangle)
            while True:
                x = np.random.random()
                y = np.random.random()
                if x + y <= 1:
                    break
            z = 1.0 - x - y
            bary = (x, y, z)
            
            # Generate profile using selected generation method
            scores, utilities, voter_pos = generate_random_profile(
                n_voters_spatial, candidate_positions, honesty_pct,
                generation_method=generation_method
            )
            
            # Run STAR voting
            winner, details = star_voting(scores)
            
            # Compute metrics
            vse = compute_vse(scores, utilities, winner)
            pvsi = compute_pvsi(scores, utilities, candidate_positions)
            
            cart = barycentric_to_cartesian(bary)
            
            spatial_points.append({
                'barycentric': bary,
                'cartesian': cart,
                'scores': scores,
                'utilities': utilities,
                'voter_positions': voter_pos,
                'winner': winner,
                'details': details,
                'vse': vse,
                'pvsi': pvsi,
            })
    
    elif current_view == 'yee':
        # Yee diagram is generated on-the-fly in draw function
        pass
    
    update_display()


def on_clear(event):
    """Handle clear button."""
    global current_scores, current_utilities, current_voter_positions
    global spatial_points, selected_point_idx
    
    current_scores = None
    current_utilities = None
    current_voter_positions = None
    spatial_points = []
    selected_point_idx = None
    
    update_display()


def on_voters_change(val):
    """Handle voters slider change."""
    global n_voters_single, n_voters_spatial
    n_voters_single = int(val)
    n_voters_spatial = int(val)


def on_profiles_change(val):
    """Handle profiles slider change."""
    global n_profiles_spatial
    n_profiles_spatial = int(val)


def on_honesty_change(val):
    """Handle honesty slider change."""
    global honesty_pct
    honesty_pct = int(val)


def on_yee_res_change(val):
    """Handle Yee resolution slider change."""
    global yee_resolution
    yee_resolution = int(val)


def on_gen_method_change(label):
    """Handle generation method change."""
    global generation_method
    generation_method = label


def on_edit(event):
    """Handle edit scores button - opens a simple score editor dialog."""
    global current_scores, current_utilities, current_voter_positions
    
    if current_view != 'single':
        print("Edit mode only available in single profile view")
        return
    
    if current_scores is None:
        # Create a default small profile for editing
        n_voters = 5
        current_scores = np.array([
            [5, 3, 1],  # Voter 1: prefers A
            [4, 5, 2],  # Voter 2: prefers B
            [2, 3, 5],  # Voter 3: prefers C
            [5, 4, 3],  # Voter 4: prefers A
            [3, 4, 5],  # Voter 5: prefers C
        ], dtype=np.int32)
        
        # Generate corresponding positions and utilities
        current_voter_positions = generate_voter_positions_uniform(n_voters)
        current_utilities = compute_utilities(current_voter_positions, candidate_positions)
    
    # Print current scores to console for manual editing guidance
    print("\n" + "="*50)
    print("CURRENT SCORE MATRIX")
    print("="*50)
    print("Voter   A   B   C")
    print("-" * 20)
    for v in range(current_scores.shape[0]):
        print(f"  {v+1:2d}    {current_scores[v, 0]}   {current_scores[v, 1]}   {current_scores[v, 2]}")
    print("\nTo edit scores, modify current_scores array directly")
    print("or use the Generate button with different honesty %")
    print("="*50 + "\n")
    
    update_display()


def on_click(event):
    """Handle mouse click for point selection."""
    global selected_point_idx
    
    if event.inaxes != ax_main:
        return
    
    if current_view == 'spatial' and spatial_points:
        click_xy = np.array([event.xdata, event.ydata], dtype=np.float64)
        
        if spatial_mode == 'triangle':
            pts_xy = np.array([p['cartesian'] for p in spatial_points], dtype=np.float64)
        else:
            # For cube mode, compute projected positions
            pts_xy = []
            for point_data in spatial_points:
                scores = point_data['scores']
                n_voters = scores.shape[0]
                a_beats_b = np.sum(scores[:, 0] > scores[:, 1])
                b_beats_a = np.sum(scores[:, 1] > scores[:, 0])
                m_ab = (a_beats_b - b_beats_a) / n_voters
                b_beats_c = np.sum(scores[:, 1] > scores[:, 2])
                c_beats_b = np.sum(scores[:, 2] > scores[:, 1])
                m_bc = (b_beats_c - c_beats_b) / n_voters
                pts_xy.append([m_ab, m_bc])
            pts_xy = np.array(pts_xy, dtype=np.float64)
        
        dists = np.sqrt(np.sum((pts_xy - click_xy) ** 2, axis=1))
        idx = int(np.argmin(dists))
        
        threshold = 0.15 if spatial_mode == 'cube' else 0.08
        if dists[idx] <= threshold:
            selected_point_idx = idx
        else:
            selected_point_idx = None
        
        update_display()


# Connect callbacks
radio_view.on_clicked(on_view_change)
radio_spatial.on_clicked(on_spatial_change)
radio_gen_method.on_clicked(on_gen_method_change)
button_generate.on_clicked(on_generate)
button_clear.on_clicked(on_clear)
button_edit.on_clicked(on_edit)
slider_voters.on_changed(on_voters_change)
slider_profiles.on_changed(on_profiles_change)
slider_honesty.on_changed(on_honesty_change)
slider_yee_res.on_changed(on_yee_res_change)
fig.canvas.mpl_connect('button_press_event', on_click)


plt.show()

