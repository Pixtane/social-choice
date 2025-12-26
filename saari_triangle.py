import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from voting_rules import VOTING_RULES, CANDIDATES, _get_positions

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

def generate_profile_from_point(bary, n_voters, method='saari', seed=None):
    """Generate a voting profile from a point in Saari's triangle."""
    # Use seed for deterministic generation based on coordinates
    if seed is not None:
        np.random.seed(seed)
    
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
    # Pairwise majority winners
    w_ab = _pairwise_majority_winner(profile, 0, 1)
    w_ac = _pairwise_majority_winner(profile, 0, 2)
    w_bc = _pairwise_majority_winner(profile, 1, 2)

    # Any tie => not a strict order (treat as "circular" marker)
    if w_ab is None or w_ac is None or w_bc is None:
        return None, True

    wins = {0: 0, 1: 0, 2: 0}
    wins[w_ab] += 1
    wins[w_ac] += 1
    wins[w_bc] += 1

    # Cycle iff each candidate wins exactly one pairwise contest
    if set(wins.values()) == {1}:
        return None, True

    # Transitive case: one candidate has 2 wins, one has 0 wins, one has 1 win
    top = max(wins.keys(), key=lambda c: wins[c])
    bottom = min(wins.keys(), key=lambda c: wins[c])
    middle = ({0, 1, 2} - {top, bottom}).pop()

    order = (top, middle, bottom)
    order_to_type = {
        (0, 1, 2): 1,  # A>B>C
        (0, 2, 1): 2,  # A>C>B
        (1, 0, 2): 3,  # B>A>C
        (1, 2, 0): 4,  # B>C>A
        (2, 0, 1): 5,  # C>A>B
        (2, 1, 0): 6,  # C>B>A
    }
    return order_to_type.get(order), False


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

def calculate_voting_outcomes(points_data, n_voters, method='saari'):
    """Calculate voting outcomes for all points using all voting rules."""
    results = {}
    for rule_name in VOTING_RULES.keys():
        results[rule_name] = []
    
    for i, point_data in enumerate(points_data):
        bary = point_data['barycentric']
        # Use deterministic seed based on coordinates for consistency
        coord_seed = int((bary[0] * 1000 + bary[1] * 100 + bary[2] * 10) * 1000) % (2**31)
        profile = generate_profile_from_point(bary, n_voters, method, seed=coord_seed)
        
        for rule_name, rule_func in VOTING_RULES.items():
            winner = rule_func(profile)
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
                
                # Generate profile to determine type (use point index as seed for consistency)
                # Create a deterministic seed from coordinates
                coord_seed = int((bary[0] * 1000 + bary[1] * 100 + bary[2] * 10) * 1000) % (2**31)
                profile = generate_profile_from_point(bary, n_voters_sim, generation_method, seed=coord_seed)
                # Type is a deterministic classification of the point's region
                # (independent from the stochastic profile generator).
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
                    # Show X if circular (instead of number)
                    ax.text(
                        cart_x,
                        cart_y,
                        'X',
                        ha='center',
                        va='center',
                        fontsize=9,
                        weight='bold',
                        color='white',
                        bbox=dict(boxstyle='circle', facecolor='red', alpha=0.7, pad=0.3),
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
                
                # Generate profile to determine type (use deterministic seed)
                coord_seed = int((bary[0] * 1000 + bary[1] * 100 + bary[2] * 10) * 1000) % (2**31)
                profile = generate_profile_from_point(bary, n_voters_sim, generation_method, seed=coord_seed)
                # Type is a deterministic classification of the point's region
                # (independent from the stochastic profile generator).
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
                    # Show X if circular
                    ax.text(cart_x, cart_y, 'X', ha='center', va='center',
                           fontsize=9, weight='bold', color='white',
                           bbox=dict(boxstyle='circle', facecolor='red', alpha=0.7, pad=0.3))
                else:
                    # Show profile type number
                    text_color = 'white' if color in ['blue', 'green', 'cyan', 'lime', 'red'] else 'black'
                    ax.text(cart_x, cart_y, str(profile_type), ha='center', va='center',
                           fontsize=8, weight='bold', color=text_color)
                
                if len(points_data) <= 20:  # Only label if not too many points
                    label = point_data.get('label', f'P{i+1}')
                    ax.text(cart_x + 0.02, cart_y + 0.02, label, fontsize=9)
    
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
    
    # Calculate outcomes for all points
    results = calculate_voting_outcomes(points_data, n_voters_sim, generation_method)
    
    # Count winners for each rule
    stats_text = "VOTING RULE STATISTICS\n"
    stats_text += "=" * 60 + "\n\n"
    stats_text += f"Number of Points: {len(points_data)}\n"
    stats_text += f"Voters per Profile: {n_voters_sim}\n"
    stats_text += f"Generation Method: {generation_method}\n\n"
    stats_text += "-" * 60 + "\n\n"
    stats_text += "AGGREGATE RESULTS (across all points):\n"
    stats_text += "-" * 60 + "\n\n"
    
    for rule_name in VOTING_RULES.keys():
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
        for rule_name in VOTING_RULES.keys():
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
        ax_stats.clear()
        ax_stats.axis('off')
    elif current_view == 'calculate':
        draw_triangle(ax_triangle, show_points=True, show_medians_flag=True)
        update_statistics(reset_scroll=reset_scroll)
    elif current_view == 'statistics':
        draw_triangle(ax_triangle, show_points=True, show_medians_flag=False)
        update_statistics(reset_scroll=reset_scroll)
    
    fig.canvas.draw_idle()

# Initial draw
draw_triangle(ax_triangle)

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
    global points_data
    points_data = []
    update_display(reset_scroll=True)

def on_generate_points(event):
    """Generate random points for simulation."""
    global points_data, n_profiles_sim
    points_data = []
    
    # Generate random barycentric coordinates
    for i in range(n_profiles_sim):
        # Generate random point in triangle using rejection sampling
        while True:
            x = np.random.random()
            y = np.random.random()
            if x + y <= 1:
                break
        
        bary = normalize_barycentric(x, y)
        cart_x, cart_y = barycentric_to_cartesian(bary)
        
        label = f'P{i+1}'
        points_data.append({
            'barycentric': bary,
            'cartesian': (cart_x, cart_y),
            'label': label
        })
    
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

# Connect callbacks
button_add.on_clicked(on_add_point)
button_clear.on_clicked(on_clear_all)
button_generate.on_clicked(on_generate_points)
slider_voters.on_changed(on_voters_change)
slider_profiles.on_changed(on_profiles_change)
radio_method.on_clicked(on_method_change)
radio_view.on_clicked(on_view_change)
check_medians.on_clicked(on_medians_change)
button_scroll_up.on_clicked(on_scroll_up)
button_scroll_down.on_clicked(on_scroll_down)

plt.show()

