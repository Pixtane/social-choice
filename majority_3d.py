"""
3D Majority Geometry Simulator

Plots election profiles in 3D space where each axis represents a pairwise margin:
  - X axis: m_AB (margin A vs B)
  - Y axis: m_AC (margin A vs C)
  - Z axis: m_BC (margin B vs C)

Each margin is in [-1, 1], creating a cube visualization space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from voting_rules import VOTING_RULES, CANDIDATES, _get_positions

# =============================================================================
# Constants and rankings
# =============================================================================

# Rank order convention (0=A, 1=B, 2=C):
#   0: ABC, 1: ACB, 2: BAC, 3: BCA, 4: CAB, 5: CBA
RANKINGS_6_STR = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]

# The 6 strict rankings as permutations
def _rankings_6():
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

# Pairwise contributions per ranking, in order: [A>B, A>C, B>C]
PAIRWISE = np.array(
    [
        [1, 1, 1],  # ABC
        [1, 1, 0],  # ACB
        [0, 1, 1],  # BAC
        [0, 0, 1],  # BCA
        [1, 0, 0],  # CAB
        [0, 0, 0],  # CBA
    ],
    dtype=np.float64,
)

# Type labels for the 6 transitive orders + cycle
SEGMENT_LABELS = [
    "A>B>C",  # 1
    "A>C>B",  # 2
    "C>A>B",  # 3
    "C>B>A",  # 4
    "B>C>A",  # 5
    "B>A>C",  # 6
    "Cycle",  # 7
]

# Colors for segments 1-6 + cycle
SEGMENT_COLORS = {
    1: 'red',      # A>B>C
    2: 'orange',   # A>C>B
    3: 'green',    # C>A>B
    4: 'lime',     # C>B>A
    5: 'cyan',     # B>C>A
    6: 'blue',     # B>A>C
    7: 'purple',   # Cycle
    None: 'gray',  # Tie (not in 7-vector)
}

# =============================================================================
# Core math utilities
# =============================================================================

def profile_to_ranking_freq(profile):
    """
    Convert a (n_voters, 3) profile of permutations into a length-6 frequency vector
    in the canonical order [ABC, ACB, BAC, BCA, CAB, CBA].
    """
    freq = np.zeros(6, dtype=np.float64)
    for row in profile:
        idx = _RANKING_TUPLE_TO_IDX.get(tuple(int(x) for x in row))
        if idx is not None:
            freq[idx] += 1.0
    return freq


def pairwise_counts_from_freq(freq):
    """
    Given a length-6 ranking frequency vector, return (A_B, A_C, B_C, total).
    """
    f = np.asarray(freq, dtype=np.float64)
    if f.shape != (6,):
        raise ValueError("freq must be length 6")
    A_B, A_C, B_C = f @ PAIRWISE
    total = float(np.sum(f))
    return float(A_B), float(A_C), float(B_C), total


def pairwise_margins_from_freq(freq):
    """
    Compute normalized pairwise margins from a 6-ranking frequency vector.
    Returns (m_AB, m_AC, m_BC) each in [-1, 1].
    m_XY = 2 * p(X>Y) - 1
    """
    A_B, A_C, B_C, total = pairwise_counts_from_freq(freq)
    if total <= 0:
        return 0.0, 0.0, 0.0
    m_ab = 2.0 * (A_B / total) - 1.0
    m_ac = 2.0 * (A_C / total) - 1.0
    m_bc = 2.0 * (B_C / total) - 1.0
    return m_ab, m_ac, m_bc


def classify_majority_relation_from_freq(freq, eps=1e-12):
    """
    Classify the majority relation implied by a 6-ranking distribution.

    Returns: (segment, relation)
      - segment: int 1..6 when relation == 'transitive', 7 when 'cycle', None when 'tie'
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
        return 7, "cycle"

    # Otherwise transitive (one candidate has 2 wins)
    if not any(w == 2 for w in wins.values()):
        return 7, "cycle"  # fallback

    order = tuple(sorted(wins.keys(), key=lambda k: wins[k], reverse=True))
    # Type numbering:
    #  1: A>B>C
    #  2: A>C>B
    #  3: C>A>B
    #  4: C>B>A
    #  5: B>C>A
    #  6: B>A>C
    order_to_type = {
        ("A", "B", "C"): 1,
        ("A", "C", "B"): 2,
        ("C", "A", "B"): 3,
        ("C", "B", "A"): 4,
        ("B", "C", "A"): 5,
        ("B", "A", "C"): 6,
    }
    return order_to_type.get(order, 7), "transitive"


# =============================================================================
# Profile generation
# =============================================================================

def generate_profile_uniform(n_voters, seed=None):
    """Generate a profile where each voter picks a random strict ranking uniformly."""
    if seed is not None:
        np.random.seed(seed)
    base = np.array([0, 1, 2], dtype=np.int8)
    profile = np.empty((n_voters, 3), dtype=np.int8)
    for j in range(n_voters):
        profile[j] = np.random.permutation(base)
    return profile


def generate_profile_saari(n_voters, seed=None):
    """
    Generate a profile using Saari-style barycentric weighting.
    Sample a random point in the simplex, then generate rankings weighted by that point.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random barycentric coordinates
    x = np.random.random()
    y = np.random.random()
    if x + y > 1:
        x, y = 1 - x, 1 - y
    z = 1.0 - x - y
    
    # Weights for each ranking based on barycentric position
    w_abc = x * (1 - y * 0.5) * (1 - z * 0.5)
    w_acb = x * (1 - y) * z
    w_bac = y * (1 - z) * x
    w_bca = y * (1 - x * 0.5) * (1 - z * 0.5)
    w_cab = z * (1 - y) * x
    w_cba = z * (1 - x * 0.5) * (1 - y * 0.5)
    
    weights = np.array([w_abc, w_acb, w_bac, w_bca, w_cab, w_cba])
    weights = weights + 0.01  # small base probability
    weights = weights / np.sum(weights)
    
    counts = np.random.multinomial(n_voters, weights)
    rankings = _rankings_6()
    
    profile = []
    for i, count in enumerate(counts):
        for _ in range(count):
            profile.append(rankings[i])
    
    profile_array = np.array(profile, dtype=np.int8)
    np.random.shuffle(profile_array)
    return profile_array


def generate_profile_weighted(n_voters, seed=None):
    """
    Generate a profile with weighted first-choice based on random barycentric coords.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random barycentric coordinates
    x = np.random.random()
    y = np.random.random()
    if x + y > 1:
        x, y = 1 - x, 1 - y
    z = 1.0 - x - y
    
    probs = np.array([x, y, z])
    probs = probs / np.sum(probs)
    
    profile = np.empty((n_voters, 3), dtype=np.int8)
    for j in range(n_voters):
        first = np.random.choice(3, p=probs)
        remaining = [c for c in range(3) if c != first]
        rem_probs = probs[remaining]
        rem_probs = rem_probs / np.sum(rem_probs)
        second = np.random.choice(remaining, p=rem_probs)
        third = [c for c in remaining if c != second][0]
        profile[j] = np.array([first, second, third], dtype=np.int8)
    return profile


GENERATION_METHODS = {
    'saari': generate_profile_saari,
    'uniform': generate_profile_uniform,
    'weighted': generate_profile_weighted,
}


# =============================================================================
# Global state
# =============================================================================
points_data = []  # List of dicts with profile info
n_voters_sim = 101  # Odd to avoid ties
n_profiles_sim = 100
generation_method = 'saari'
selected_point_idx = None
selected_voting_rules = set(VOTING_RULES.keys())
display_type = 'all'  # 'only_cycles', 'cycles_gray', 'all'

# 7-vector: counts for segments 1-6 + cycle
segment_counts = np.zeros(7, dtype=np.int64)


# =============================================================================
# UI Setup
# =============================================================================

fig = plt.figure(figsize=(16, 10))
ax_3d = fig.add_axes([0.05, 0.30, 0.45, 0.65], projection='3d')
ax_stats = fig.add_axes([0.55, 0.30, 0.42, 0.65])
ax_stats.axis('off')


def draw_3d_space(ax, show_points=True):
    """Draw the 3D space with optional points."""
    ax.clear()
    
    # Set axis limits (margins are in [-1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    
    # Set axis labels
    ax.set_xlabel('m_AB (A vs B)', fontsize=10)
    ax.set_ylabel('m_AC (A vs C)', fontsize=10)
    ax.set_zlabel('m_BC (B vs C)', fontsize=10)
    ax.set_title("3D Majority Space", fontsize=14, weight='bold')
    
    # Draw coordinate planes at 0 (optional, can be commented out for clarity)
    # Draw planes at x=0, y=0, z=0 with low alpha
    plane_alpha = 0.1
    plane_color = 'gray'
    
    # Plane at x=0 (YZ plane)
    y_plane = np.linspace(-1, 1, 10)
    z_plane = np.linspace(-1, 1, 10)
    Y_plane, Z_plane = np.meshgrid(y_plane, z_plane)
    X_plane = np.zeros_like(Y_plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=plane_alpha, color=plane_color)
    
    # Plane at y=0 (XZ plane)
    x_plane = np.linspace(-1, 1, 10)
    z_plane = np.linspace(-1, 1, 10)
    X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
    Y_plane = np.zeros_like(X_plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=plane_alpha, color=plane_color)
    
    # Plane at z=0 (XY plane)
    x_plane = np.linspace(-1, 1, 10)
    y_plane = np.linspace(-1, 1, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.zeros_like(X_plane)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=plane_alpha, color=plane_color)
    
    # Draw cube outline (12 edges of a cube)
    # Define the 8 vertices of the cube
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
    ]
    
    # Define the 12 edges (pairs of vertex indices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                'k-', linewidth=0.5, alpha=0.3)
    
    # Draw points
    if show_points and points_data:
        # Group points by segment for better visualization
        points_by_segment = {}
        for i, point_data in enumerate(points_data):
            segment = point_data['segment']
            if segment not in points_by_segment:
                points_by_segment[segment] = []
            points_by_segment[segment].append((i, point_data))
        
        # Plot each segment
        for segment, point_list in points_by_segment.items():
            x_coords = []
            y_coords = []
            z_coords = []
            indices = []
            
            for idx, point_data in point_list:
                x, y, z = point_data['margins']  # m_AB, m_AC, m_BC
                
                # Filter/color based on display_type
                is_cycle = (segment == 7)
                
                if display_type == 'only_cycles':
                    if not is_cycle:
                        continue  # Skip non-cycle points
                elif display_type == 'cycles_gray':
                    if is_cycle:
                        pass  # Show cycles in their color
                    else:
                        continue  # Skip non-cycles (they'll be shown separately)
                
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                indices.append(idx)
            
            if len(x_coords) > 0:
                color = SEGMENT_COLORS.get(segment, 'gray')
                marker_size = 20 if len(points_data) <= 100 else 10
                ax.scatter(x_coords, y_coords, z_coords, 
                          c=color, s=marker_size, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # For cycles_gray mode, show non-cycles in gray
        if display_type == 'cycles_gray':
            x_coords = []
            y_coords = []
            z_coords = []
            for i, point_data in enumerate(points_data):
                segment = point_data['segment']
                is_cycle = (segment == 7)
                if not is_cycle:
                    x, y, z = point_data['margins']
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
            if len(x_coords) > 0:
                ax.scatter(x_coords, y_coords, z_coords, 
                          c='gray', s=20 if len(points_data) <= 100 else 10, 
                          alpha=0.4, edgecolors='black', linewidths=0.3)
    
    # Highlight selected point
    if show_points and selected_point_idx is not None and 0 <= selected_point_idx < len(points_data):
        sel = points_data[selected_point_idx]
        sx, sy, sz = sel['margins']
        ax.scatter([sx], [sy], [sz], s=300, c='magenta', marker='o', 
                   edgecolors='black', linewidths=2, alpha=0.9)


def update_statistics():
    """Update the statistics display."""
    global stats_text_full, segment_counts
    
    ax_stats.clear()
    ax_stats.axis('off')
    
    if not points_data:
        ax_stats.text(0.5, 0.5, 'No points generated yet.\nClick "Generate Points" to start.',
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        return
    
    # Compute segment counts
    segment_counts = np.zeros(7, dtype=np.int64)
    tie_count = 0
    
    for point_data in points_data:
        segment = point_data['segment']
        if segment is None:
            tie_count += 1
        elif 1 <= segment <= 7:
            segment_counts[segment - 1] += 1
    
    # Build statistics text
    lines = []
    lines.append("3D MAJORITY SPACE STATISTICS")
    lines.append("=" * 60)
    lines.append(f"Number of Profiles: {len(points_data)}")
    lines.append(f"Voters per Profile: {n_voters_sim}")
    lines.append(f"Generation Method: {generation_method}")
    lines.append("")
    lines.append("7-VECTOR (Segment Counts)")
    lines.append("-" * 60)
    
    total = len(points_data)
    for i in range(7):
        count = segment_counts[i]
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        label = SEGMENT_LABELS[i]
        lines.append(f"  {i+1}. {label:<10}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    if tie_count > 0:
        pct = 100 * tie_count / total if total > 0 else 0
        lines.append(f"  Ties:        {tie_count:>4} ({pct:>5.1f}%) [not in 7-vector]")
    
    lines.append("")
    lines.append("7-VECTOR (percentages):")
    pct_vec = 100 * segment_counts / total if total > 0 else np.zeros(7)
    lines.append(f"  [{', '.join(f'{p:.1f}' for p in pct_vec)}]")
    
    # Voting rule results
    lines.append("")
    lines.append("VOTING RULE RESULTS (across all profiles)")
    lines.append("-" * 60)
    
    rule_names = [name for name in VOTING_RULES.keys() if name in selected_voting_rules]
    for rule_name in rule_names:
        winner_counts = {'A': 0, 'B': 0, 'C': 0, None: 0}
        for point_data in points_data:
            outcomes = point_data.get('outcomes', {})
            winner = outcomes.get(rule_name)
            if winner in winner_counts:
                winner_counts[winner] += 1
            else:
                winner_counts[None] += 1
        
        display_name = rule_name.replace('_', ' ').title()
        lines.append(f"{display_name:<18}: A={winner_counts['A']}, B={winner_counts['B']}, C={winner_counts['C']}")
    
    stats_text_full = '\n'.join(lines)
    
    # Display text
    ax_stats.text(0.02, 0.98, stats_text_full, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_point_details(index):
    """Show details for a selected point."""
    ax_stats.clear()
    ax_stats.axis('off')
    
    if index is None or not (0 <= index < len(points_data)):
        ax_stats.text(0.5, 0.5, 'No point selected.',
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        return
    
    point_data = points_data[index]
    
    lines = []
    lines.append("POINT DETAILS (click empty space to deselect)")
    lines.append("=" * 60)
    lines.append(f"Point index: {index + 1} of {len(points_data)}")
    m_ab, m_ac, m_bc = point_data['margins']
    lines.append(f"3D Position: ({m_ab:.4f}, {m_ac:.4f}, {m_bc:.4f})")
    lines.append(f"Margins: m_AB={m_ab:.3f}, m_AC={m_ac:.3f}, m_BC={m_bc:.3f}")
    
    segment = point_data['segment']
    if segment is None:
        lines.append(f"Segment: Tie (pairwise tie exists)")
    elif segment == 7:
        lines.append(f"Segment: 7 (Condorcet cycle)")
    else:
        lines.append(f"Segment: {segment} ({SEGMENT_LABELS[segment-1]})")
    
    lines.append(f"Relation: {point_data['relation']}")
    lines.append(f"Rule conflict: {'YES' if point_data.get('has_conflict', False) else 'No'}")
    
    lines.append("")
    lines.append("Ranking distribution:")
    freq = point_data['freq']
    total = sum(freq)
    for i, count in enumerate(freq):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"  {RANKINGS_6_STR[i]}: {int(count):>4} ({pct:>5.1f}%)")
    
    lines.append("")
    lines.append("Winners by rule:")
    lines.append("-" * 60)
    outcomes = point_data.get('outcomes', {})
    for rule_name, winner in outcomes.items():
        display_name = rule_name.replace('_', ' ').title()
        lines.append(f"  {display_name:<20}: {winner}")
    
    text = '\n'.join(lines)
    ax_stats.text(0.02, 0.98, text, transform=ax_stats.transAxes,
                  fontsize=9, fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def update_display():
    """Update the main display."""
    draw_3d_space(ax_3d, show_points=True)
    
    if selected_point_idx is not None:
        update_point_details(selected_point_idx)
    else:
        update_statistics()
    
    fig.canvas.draw_idle()


# Initial draw
draw_3d_space(ax_3d)
update_statistics()


# =============================================================================
# UI Controls
# =============================================================================

ax_generate = fig.add_axes([0.05, 0.22, 0.12, 0.04])
button_generate = Button(ax_generate, 'Generate Points')

ax_clear = fig.add_axes([0.19, 0.22, 0.08, 0.04])
button_clear = Button(ax_clear, 'Clear All')

ax_voters = fig.add_axes([0.05, 0.15, 0.20, 0.03])
slider_voters = Slider(ax_voters, 'Voters', 11, 501, valinit=101, valstep=2)  # Odd to avoid ties

ax_profiles = fig.add_axes([0.05, 0.10, 0.20, 0.03])
slider_profiles = Slider(ax_profiles, 'Profiles', 10, 500, valinit=100, valstep=10)

ax_method = fig.add_axes([0.30, 0.08, 0.12, 0.12])
ax_method.set_title('Generation Method', fontsize=9)
radio_method = RadioButtons(ax_method, ['saari', 'uniform', 'weighted'], active=0)

ax_rules = fig.add_axes([0.45, 0.05, 0.25, 0.20])
ax_rules.set_title('Voting Rules', fontsize=9)
rule_names_list = list(VOTING_RULES.keys())
check_rules = CheckButtons(ax_rules, rule_names_list, [True] * len(rule_names_list))

ax_display = fig.add_axes([0.72, 0.05, 0.12, 0.12])
ax_display.set_title('Display types', fontsize=9)
radio_display = RadioButtons(ax_display, ['Only cycles', 'Cycles and normal in gray', 'All'], active=2)


# =============================================================================
# Callbacks
# =============================================================================

def on_generate(event):
    """Generate random profiles and plot them."""
    global points_data, selected_point_idx
    
    points_data = []
    selected_point_idx = None
    
    gen_func = GENERATION_METHODS.get(generation_method, generate_profile_saari)
    
    for i in range(n_profiles_sim):
        seed = np.random.randint(0, 2**31)
        profile = gen_func(n_voters_sim, seed=seed)
        
        # Compute frequency and margins
        freq = profile_to_ranking_freq(profile)
        m_ab, m_ac, m_bc = pairwise_margins_from_freq(freq)
        
        # Classify
        segment, relation = classify_majority_relation_from_freq(freq)
        
        # Compute voting rule outcomes
        outcomes = {}
        for rule_name in [n for n in VOTING_RULES.keys() if n in selected_voting_rules]:
            outcomes[rule_name] = VOTING_RULES[rule_name](profile)
        
        # Check for conflict
        winners = [w for w in outcomes.values() if w is not None]
        has_conflict = len(set(winners)) > 1
        
        points_data.append({
            'margins': (m_ab, m_ac, m_bc),  # 3D coordinates
            'freq': freq,
            'segment': segment,
            'relation': relation,
            'outcomes': outcomes,
            'has_conflict': has_conflict,
        })
    
    update_display()


def on_clear(event):
    """Clear all points."""
    global points_data, selected_point_idx
    points_data = []
    selected_point_idx = None
    update_display()


def on_voters_change(val):
    """Update number of voters."""
    global n_voters_sim
    n_voters_sim = int(val)


def on_profiles_change(val):
    """Update number of profiles."""
    global n_profiles_sim
    n_profiles_sim = int(val)


def on_method_change(label):
    """Change generation method."""
    global generation_method
    generation_method = label


def on_rules_change(label):
    """Toggle voting rules."""
    global selected_voting_rules
    if label in selected_voting_rules:
        selected_voting_rules.remove(label)
    else:
        selected_voting_rules.add(label)
    if selected_point_idx is not None:
        update_point_details(selected_point_idx)


def on_display_change(label):
    """Change display type."""
    global display_type
    # Map UI labels to internal values
    label_map = {
        'Only cycles': 'only_cycles',
        'Cycles and normal in gray': 'cycles_gray',
        'All': 'all'
    }
    display_type = label_map.get(label, 'all')
    update_display()


def on_click(event):
    """Handle click to select a point."""
    global selected_point_idx
    
    if event.inaxes != ax_3d:
        return
    if not points_data:
        return
    
    # For 3D plots, we need to project the click to 3D space
    # This is approximate - we use the current view's projection
    click_2d = np.array([event.xdata, event.ydata], dtype=np.float64)
    
    if click_2d[0] is None or click_2d[1] is None:
        selected_point_idx = None
        update_display()
        return
    
    # Get all point positions
    pts_3d = np.array([p['margins'] for p in points_data], dtype=np.float64)
    
    # Project to 2D using current view (approximate)
    # This is a simplified approach - in practice, 3D point selection is complex
    # We'll use the XY projection for simplicity
    pts_2d = pts_3d[:, :2]  # Use X and Y coordinates
    
    dists = np.sqrt(np.sum((pts_2d - click_2d) ** 2, axis=1))
    idx = int(np.argmin(dists))
    
    # Threshold for selection (in data coordinates)
    threshold = 0.15
    if dists[idx] <= threshold:
        selected_point_idx = idx
    else:
        selected_point_idx = None
    
    update_display()


# Connect callbacks
button_generate.on_clicked(on_generate)
button_clear.on_clicked(on_clear)
slider_voters.on_changed(on_voters_change)
slider_profiles.on_changed(on_profiles_change)
radio_method.on_clicked(on_method_change)
check_rules.on_clicked(on_rules_change)
radio_display.on_clicked(on_display_change)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

