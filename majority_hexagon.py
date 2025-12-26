"""
Hexagon Majority Geometry Simulator

Plots election profiles in Saari-style pairwise-majority geometry (hexagon),
classifying outcomes into 7 segments:
  - Segments 1-6: the 6 transitive majority orders
  - Segment 7: Condorcet paradox (strict cycle), both orientations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
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

# Colors for segments 1-6 (same as saari_triangle.py) + cycle
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
# Circle geometry with inscribed triangle
# =============================================================================
# We use a circle of radius 1, with 3 lines forming an equilateral triangle
# in the center. This creates 7 regions:
#   - 6 outer regions (transitive majority orders)
#   - 1 central triangular region (cycle) - but we draw 2 opposite triangles

CIRCLE_RADIUS = 1.0

# Equilateral triangle vertices (centered at origin)
TRIANGLE_VERTICES = np.array([
    [0.0, 1.0],                          # Y (top)
    [-np.sqrt(3)/2, -0.5],               # X (bottom-left)
    [np.sqrt(3)/2, -0.5]                 # Z (bottom-right)
], dtype=np.float64)

# Edge normals pointing "outside" the triangle
# N_AB: normal to edge between vertices 1 and 2 (XZ, bottom edge)
# N_BC: normal to edge between vertices 2 and 0 (YZ, right edge)
# N_CA: normal to edge between vertices 0 and 1 (XY, left edge)
N_AB = np.array([0.0, -1.0])            # normal to XZ (A ~ B)
N_BC = np.array([np.sqrt(3)/2, 0.5])    # normal to YZ (B ~ C)
N_CA = np.array([-np.sqrt(3)/2, 0.5])   # normal to XY (C ~ A)

# The 3 lines extend from triangle sides to circle edge
# Each line is defined by two triangle vertices; we extend it to the circle


def _line_circle_intersections(p1, p2, r=CIRCLE_RADIUS):
    """Find where line through p1, p2 intersects circle of radius r."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Parametric: (p1[0] + t*dx, p1[1] + t*dy)
    # Solve: (p1[0] + t*dx)^2 + (p1[1] + t*dy)^2 = r^2
    a = dx*dx + dy*dy
    b = 2 * (p1[0]*dx + p1[1]*dy)
    c = p1[0]*p1[0] + p1[1]*p1[1] - r*r
    
    disc = b*b - 4*a*c
    if disc < 0:
        return None, None
    
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    pt1 = (p1[0] + t1*dx, p1[1] + t1*dy)
    pt2 = (p1[0] + t2*dx, p1[1] + t2*dy)
    return pt1, pt2


def _line_offset_perpendicular(p1, p2, offset):
    """
    Create a line parallel to the line through p1 and p2, offset perpendicularly by 'offset'.
    Returns (new_p1, new_p2) where the new line is offset in the direction of the perpendicular.
    
    Positive offset moves in the direction of rotating the line vector 90° counterclockwise.
    """
    # Direction vector of the line
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    if length < 1e-10:
        # Degenerate line, return original
        return p1, p2
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Perpendicular vector (rotate 90° counterclockwise: (x, y) -> (-y, x))
    perp_x = -dy_norm
    perp_y = dx_norm
    
    # Offset both endpoints
    new_p1 = (p1[0] + offset * perp_x, p1[1] + offset * perp_y)
    new_p2 = (p2[0] + offset * perp_x, p2[1] + offset * perp_y)
    
    return new_p1, new_p2


def _line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two lines defined by (p1, p2) and (p3, p4).
    Returns (x, y) or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Line 1: (x1, y1) + t * (x2-x1, y2-y1)
    # Line 2: (x3, y3) + s * (x4-x3, y4-y3)
    
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    # Solve: x1 + t*dx1 = x3 + s*dx2, y1 + t*dy1 = y3 + s*dy2
    # Rearranging: t*dx1 - s*dx2 = x3 - x1, t*dy1 - s*dy2 = y3 - y1
    
    det = dx1 * (-dy2) - (-dx2) * dy1
    if abs(det) < 1e-10:
        return None  # Lines are parallel
    
    t = ((x3 - x1) * (-dy2) - (y3 - y1) * (-dx2)) / det
    
    x = x1 + t * dx1
    y = y1 + t * dy1
    
    return (x, y)


def _three_line_intersection(line1, line2, line3):
    """
    Find the best intersection point of three lines.
    Since three lines may not all intersect at one point, we:
    1. Try pairwise intersections
    2. If all three intersect at approximately the same point, return it
    3. Otherwise, return the average of the three pairwise intersections
    """
    p1, p2 = line1
    p3, p4 = line2
    p5, p6 = line3
    
    # Find all three pairwise intersections
    inter12 = _line_intersection(p1, p2, p3, p4)
    inter13 = _line_intersection(p1, p2, p5, p6)
    inter23 = _line_intersection(p3, p4, p5, p6)
    
    if inter12 is None or inter13 is None or inter23 is None:
        # If any are parallel, use the ones that work
        points = [p for p in [inter12, inter13, inter23] if p is not None]
        if len(points) == 0:
            return (0.0, 0.0)  # Fallback to origin
        if len(points) == 1:
            return points[0]
        # Average of available points
        avg_x = sum(p[0] for p in points) / len(points)
        avg_y = sum(p[1] for p in points) / len(points)
        return (avg_x, avg_y)
    
    # Check if all three intersections are close (within tolerance)
    dist12_13 = np.sqrt((inter12[0] - inter13[0])**2 + (inter12[1] - inter13[1])**2)
    dist12_23 = np.sqrt((inter12[0] - inter23[0])**2 + (inter12[1] - inter23[1])**2)
    dist13_23 = np.sqrt((inter13[0] - inter23[0])**2 + (inter13[1] - inter23[1])**2)
    
    tolerance = 0.01
    if dist12_13 < tolerance and dist12_23 < tolerance and dist13_23 < tolerance:
        # All three are close, return their average
        return inter12
    
    # Otherwise, return average of all three
    avg_x = (inter12[0] + inter13[0] + inter23[0]) / 3.0
    avg_y = (inter12[1] + inter13[1] + inter23[1]) / 3.0
    return (avg_x, avg_y)


def margins_to_2d(m_ab, m_ac, m_bc, scale=CIRCLE_RADIUS):
    """
    Map pairwise margins to 2D point using linear combination of edge normals.
    
    The triangle has 3 edges with normals:
    - N_AB: normal to bottom edge (A ~ B)
    - N_BC: normal to right edge (B ~ C)
    - N_CA: normal to left edge (C ~ A)
    
    Positive margin moves point "outside" along normal.
    """
    # Convert m_ac to m_ca (C vs A margin)
    m_ca = -m_ac
    
    # Linear combination of normals weighted by margins
    point = m_ab * N_AB + m_bc * N_BC + m_ca * N_CA
    
    # Optional: scale to fit visualization circle
    r = np.linalg.norm(point)
    if r > scale:
        point = point / r * scale
    
    return float(point[0]), float(point[1])


def _clip_line_to_circle(p1, p2, r=CIRCLE_RADIUS):
    """
    Clip a line segment to the circle boundary.
    Returns the clipped line segment as (p1_clipped, p2_clipped) or None if line doesn't intersect circle.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Parametric line: (x1, y1) + t * (dx, dy), t in [0, 1]
    dx = x2 - x1
    dy = y2 - y1
    
    # Solve intersection with circle: (x1 + t*dx)^2 + (y1 + t*dy)^2 = r^2
    a = dx*dx + dy*dy
    if a < 1e-10:
        return None  # Degenerate line
    
    b = 2 * (x1*dx + y1*dy)
    c = x1*x1 + y1*y1 - r*r
    
    disc = b*b - 4*a*c
    if disc < 0:
        return None  # Line doesn't intersect circle
    
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    # Find valid t values in [0, 1]
    valid_ts = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            valid_ts.append(t)
    
    if len(valid_ts) == 0:
        # Check if entire line is inside circle
        if x1*x1 + y1*y1 <= r*r and x2*x2 + y2*y2 <= r*r:
            return p1, p2
        return None
    
    if len(valid_ts) == 1:
        # One endpoint is inside, one outside
        t = valid_ts[0]
        p_intersect = (x1 + t*dx, y1 + t*dy)
        # Determine which endpoint is inside
        if x1*x1 + y1*y1 <= r*r:
            return p1, p_intersect
        else:
            return p_intersect, p2
    
    # Both intersections are on the segment
    t_min = min(valid_ts)
    t_max = max(valid_ts)
    p1_clip = (x1 + t_min*dx, y1 + t_min*dy)
    p2_clip = (x1 + t_max*dx, y1 + t_max*dy)
    return p1_clip, p2_clip


def draw_grid(ax):
    """
    Draw a grid relative to each of the three axes (triangle edges).
    Grid lines are drawn at margin intervals of 0.1 from -1.0 to 1.0.
    """
    # Get triangle vertices
    v0 = TRIANGLE_VERTICES[0]  # Top vertex
    v1 = TRIANGLE_VERTICES[1]  # Bottom-left
    v2 = TRIANGLE_VERTICES[2]  # Bottom-right
    
    # For grid lines, we'll draw lines perpendicular to normals at constant margin values
    # The margin values are already normalized, so we can use them directly
    
    # Grid step size in margin space
    step = 0.1
    
    # Generate margin values from -1.0 to 1.0 in steps of 0.1
    margin_values = np.arange(-1.0, 1.0 + step/2, step)
    
    # Colors for each axis (light, distinct colors)
    axis_colors = ['lightblue', 'lightgreen', 'lightcoral']  # B>A (horizontal), B>C (right), C>A (left)
    
    # Draw grid for each axis
    # Grid lines are perpendicular to normals, representing constant margin values
    
    # Axis 0: m_ab (N_AB = [0, -1], horizontal lines)
    for margin in margin_values:
        if abs(margin) < 0.01:  # Skip the zero line
            continue
        # Line perpendicular to N_AB: y = -margin (horizontal line)
        y_val = -margin
        if abs(y_val) <= CIRCLE_RADIUS:
            x_intersect = np.sqrt(CIRCLE_RADIUS**2 - y_val**2)
            ax.plot([-x_intersect, x_intersect], [y_val, y_val], 
                   color=axis_colors[0], linewidth=0.5, alpha=0.4, linestyle='--')
            if abs(margin) >= 0.2:
                ax.text(x_intersect * 0.8, y_val, f'{margin:.1f}', 
                       fontsize=6, color=axis_colors[0], alpha=0.6, ha='center', va='center')
    
    # Axis 1: m_bc (N_BC = [sqrt(3)/2, 0.5])
    for margin in margin_values:
        if abs(margin) < 0.01:
            continue
        # Line perpendicular to N_BC: (sqrt(3)/2)*x + 0.5*y = margin
        # Find two points on this line far from origin, then clip to circle
        n_bc_norm = np.linalg.norm(N_BC)
        c = margin * n_bc_norm
        perp_dir = np.array([-N_BC[1], N_BC[0]])  # Rotate 90° counterclockwise
        perp_dir = perp_dir / np.linalg.norm(perp_dir)
        closest = c * N_BC / (n_bc_norm**2)
        extend = 2.0 * CIRCLE_RADIUS
        p1 = tuple(closest + extend * perp_dir)
        p2 = tuple(closest - extend * perp_dir)
        circle_p1, circle_p2 = _line_circle_intersections(p1, p2)
        if circle_p1 is not None and circle_p2 is not None:
            ax.plot([circle_p1[0], circle_p2[0]], [circle_p1[1], circle_p2[1]], 
                   color=axis_colors[1], linewidth=0.5, alpha=0.4, linestyle='--')
            if abs(margin) >= 0.2:
                label_pos = circle_p1 if margin > 0 else circle_p2
                ax.text(label_pos[0], label_pos[1], f'{margin:.1f}', 
                       fontsize=6, color=axis_colors[1], alpha=0.6, ha='center', va='center')
    
    # Axis 2: m_ca (N_CA = [-sqrt(3)/2, 0.5])
    for margin in margin_values:
        if abs(margin) < 0.01:
            continue
        # Line perpendicular to N_CA: (-sqrt(3)/2)*x + 0.5*y = margin
        n_ca_norm = np.linalg.norm(N_CA)
        c = margin * n_ca_norm
        perp_dir = np.array([-N_CA[1], N_CA[0]])
        perp_dir = perp_dir / np.linalg.norm(perp_dir)
        closest = c * N_CA / (n_ca_norm**2)
        extend = 2.0 * CIRCLE_RADIUS
        p1 = tuple(closest + extend * perp_dir)
        p2 = tuple(closest - extend * perp_dir)
        circle_p1, circle_p2 = _line_circle_intersections(p1, p2)
        if circle_p1 is not None and circle_p2 is not None:
            ax.plot([circle_p1[0], circle_p2[0]], [circle_p1[1], circle_p2[1]], 
                   color=axis_colors[2], linewidth=0.5, alpha=0.4, linestyle='--')
            if abs(margin) >= 0.2:
                label_pos = circle_p1 if margin > 0 else circle_p2
                ax.text(label_pos[0], label_pos[1], f'{margin:.1f}', 
                       fontsize=6, color=axis_colors[2], alpha=0.6, ha='center', va='center')


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
# Profile generation (mirrors saari_triangle.py)
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
# Circle + Triangle geometry
# =============================================================================

def compute_triangle_lines():
    """
    Compute the 3 lines that form the central triangle, extended to the circle edge.
    Each line connects two vertices of the inner triangle, extended to circle boundary.
    """
    lines = []
    
    # Each edge of the triangle
    for i in range(3):
        p1 = TRIANGLE_VERTICES[i]
        p2 = TRIANGLE_VERTICES[(i + 1) % 3]
        
        # Extend this line to circle boundary
        circle_p1, circle_p2 = _line_circle_intersections(p1, p2)
        if circle_p1 is not None:
            lines.append((circle_p1, circle_p2))
    
    return lines


TRIANGLE_LINES = compute_triangle_lines()


# =============================================================================
# Global state
# =============================================================================
points_data = []  # List of dicts with profile info
current_view = 'hexagon'
n_voters_sim = 101  # Odd to avoid ties
n_profiles_sim = 100
generation_method = 'saari'
stats_text_full = ""
stats_scroll_pos = 0
stats_lines_per_page = 50
selected_point_idx = None
selected_voting_rules = set(VOTING_RULES.keys())
display_type = 'all'  # 'only_cycles', 'cycles_gray', 'all'

# 7-vector: counts for segments 1-6 + cycle
segment_counts = np.zeros(7, dtype=np.int64)


# =============================================================================
# UI Setup
# =============================================================================

fig = plt.figure(figsize=(16, 10))
ax_hexagon = fig.add_axes([0.05, 0.30, 0.45, 0.65])
ax_stats = fig.add_axes([0.55, 0.30, 0.42, 0.65])
ax_stats.axis('off')


def draw_hexagon(ax, show_points=True):
    """Draw the circle with central triangle and optional points."""
    ax.clear()
    ax.set_aspect('equal')
    
    # Draw circle outline
    circle = plt.Circle((0, 0), CIRCLE_RADIUS, fill=False,
                         edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    # Draw the 3 lines forming the central triangle (extended to circle)
    for (p1, p2) in TRIANGLE_LINES:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'k-', linewidth=1.5, alpha=0.7)
    
    # Draw the inner triangle (filled lightly to show cycle region)
    inner_triangle = Polygon(TRIANGLE_VERTICES, closed=True, 
                             facecolor='purple', edgecolor='purple',
                             alpha=0.15, linewidth=2)
    ax.add_patch(inner_triangle)
    
    # Also shade the opposite triangle (the "other" cycle orientation)
    # This is the triangle formed by reflecting through origin
    opposite_triangle_verts = -TRIANGLE_VERTICES
    opposite_triangle = Polygon(opposite_triangle_verts, closed=True,
                                facecolor='purple', edgecolor='purple',
                                alpha=0.15, linewidth=2)
    ax.add_patch(opposite_triangle)
    
    # Draw grid lines for each axis
    draw_grid(ax)
    
    # Label the 6 outer regions
    # Position labels at angles between the triangle lines, near circle edge
    region_labels = {
        1: "A>B>C",
        2: "A>C>B",
        3: "C>A>B",
        4: "C>B>A",
        5: "B>C>A",
        6: "B>A>C",
    }
    
    # Place labels in the 6 wedge regions (between triangle lines)
    # The triangle vertices are at 90°, 210°, 330°
    # The 6 regions are centered at angles: 30°, 90°+60°=150°, etc.
    # Actually, compute from the pure ranking positions
    label_radius = 0.7
    for idx in range(6):
        freq = np.zeros(6, dtype=np.float64)
        freq[idx] = 1.0
        m_ab, m_ac, m_bc = pairwise_margins_from_freq(freq)
        x, y = margins_to_2d(m_ab, m_ac, m_bc)
        
        # Normalize to label_radius
        r = np.sqrt(x*x + y*y)
        if r > 0.01:
            lx = x / r * label_radius
            ly = y / r * label_radius
        else:
            lx, ly = 0, 0
        
        segment = idx + 1
        ax.text(lx, ly, region_labels.get(segment, ""), fontsize=8, ha='center', va='center',
                color=SEGMENT_COLORS.get(segment, 'black'), weight='bold', alpha=0.8)
    
    # Label cycle regions (center)
    ax.text(0, 0.18, "Cycle", fontsize=9, ha='center', va='center',
            color='purple', weight='bold', alpha=0.9)
    ax.text(0, -0.18, "Cycle", fontsize=9, ha='center', va='center',
            color='purple', weight='bold', alpha=0.9)
    
    # Draw points
    if show_points and points_data:
        for i, point_data in enumerate(points_data):
            x, y = point_data['xy']
            segment = point_data['segment']
            has_conflict = point_data.get('has_conflict', False)
            
            # Filter/color based on display_type
            is_cycle = (segment == 7)
            
            if display_type == 'only_cycles':
                if not is_cycle:
                    continue  # Skip non-cycle points
                color = SEGMENT_COLORS.get(segment, 'gray')
            elif display_type == 'cycles_gray':
                if is_cycle:
                    color = SEGMENT_COLORS.get(segment, 'gray')
                else:
                    color = 'gray'  # Show non-cycles in gray
            else:  # display_type == 'all'
                color = SEGMENT_COLORS.get(segment, 'gray')
            
            marker_size = 10 if len(points_data) <= 100 else 6
            
            edge_width = 2 if has_conflict else 0
            edge_color = 'black' if has_conflict else 'none'
            
            ax.plot(x, y, 'o', color=color, markersize=marker_size,
                    markeredgecolor=edge_color, markeredgewidth=edge_width, alpha=0.8)
            
            # Add segment label
            if segment is not None:
                text_color = 'white' if color in ['blue', 'green', 'cyan', 'lime', 'red', 'purple'] else 'black'
                label = str(segment) if segment <= 6 else 'C'
                ax.text(x, y, label, ha='center', va='center',
                        fontsize=7, weight='bold', color=text_color)
    
    # Highlight selected point
    if show_points and selected_point_idx is not None and 0 <= selected_point_idx < len(points_data):
        sel = points_data[selected_point_idx]
        sx, sy = sel['xy']
        ax.plot(sx, sy, marker='o', markersize=18, markerfacecolor='none',
                markeredgecolor='magenta', markeredgewidth=3, zorder=10)
    
    # Set axis limits with padding (circle has radius 1)
    pad = 0.15
    ax.set_xlim(-CIRCLE_RADIUS - pad, CIRCLE_RADIUS + pad)
    ax.set_ylim(-CIRCLE_RADIUS - pad, CIRCLE_RADIUS + pad)
    
    ax.set_xlabel('Margin dimension 1', fontsize=10)
    ax.set_ylabel('Margin dimension 2', fontsize=10)
    ax.set_title("Majority Circle Geometry", fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)


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
    lines.append("MAJORITY HEXAGON STATISTICS")
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
    lines.append(f"2D Position: ({point_data['xy'][0]:.4f}, {point_data['xy'][1]:.4f})")
    lines.append(f"Margins: m_AB={point_data['margins'][0]:.3f}, m_AC={point_data['margins'][1]:.3f}, m_BC={point_data['margins'][2]:.3f}")
    
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
    draw_hexagon(ax_hexagon, show_points=True)
    
    if selected_point_idx is not None:
        update_point_details(selected_point_idx)
    else:
        update_statistics()
    
    fig.canvas.draw_idle()


# Initial draw
draw_hexagon(ax_hexagon)
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
        x, y = margins_to_2d(m_ab, m_ac, m_bc)
        
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
            'xy': (x, y),
            'margins': (m_ab, m_ac, m_bc),
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
    
    if event.inaxes != ax_hexagon:
        return
    if not points_data:
        return
    
    click_xy = np.array([event.xdata, event.ydata], dtype=np.float64)
    pts_xy = np.array([p['xy'] for p in points_data], dtype=np.float64)
    
    dists = np.sqrt(np.sum((pts_xy - click_xy) ** 2, axis=1))
    idx = int(np.argmin(dists))
    
    # Threshold for selection (in data coordinates)
    threshold = 0.08
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

