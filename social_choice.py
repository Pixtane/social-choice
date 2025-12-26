import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons, RangeSlider
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from voting_rules import VOTING_RULES, _get_positions

# Custom yellow colormap for noise regions (transparent to yellow/orange)
NOISE_CMAP = LinearSegmentedColormap.from_list('noise', [
    (1.0, 1.0, 1.0, 0.0),    # Transparent (no noise)
    (1.0, 1.0, 0.6, 0.3),    # Light yellow
    (1.0, 0.95, 0.3, 0.5),   # Yellow  
    (1.0, 0.8, 0.0, 0.7),    # Orange-yellow
    (1.0, 0.6, 0.0, 0.85),   # Orange (high noise)
])

# Generate all profiles at once using numpy (optimized)
def generate_profiles(n_voters, n_simulations):
    """Generate n_simulations profiles, each with n_voters random rankings."""
    base = np.array([0, 1, 2], dtype=np.int8)
    profiles = np.empty((n_simulations, n_voters, 3), dtype=np.int8)
    for i in range(n_simulations):
        for j in range(n_voters):
            profiles[i, j] = np.random.permutation(base)
    return profiles

# Vectorized pairwise counting
def count_pairwise_batch(profiles, a, b):
    """Count voters preferring candidate a over b for all profiles."""
    n_sims = profiles.shape[0]
    counts = np.empty(n_sims, dtype=np.int32)
    for i in range(n_sims):
        pos = _get_positions(profiles[i])
        counts[i] = np.sum(pos[:, a] < pos[:, b])
    return counts

# Run simulation (optimized)
def run_simulation(n_voters, n_simulations, voting_rule, compute_noise=False, noise_rules=None):
    """
    Run voting simulation.
    
    If compute_noise is True, compute winners for selected voting rules and classify
    each profile as signal-dominated (all rules agree) or noise-dominated (rules disagree).
    
    Returns:
        xs, ys, zs, colors, noise_mask, rule_contributions
        rule_contributions: dict mapping rule_name -> count of profiles where this rule disagreed
    """
    profiles = generate_profiles(n_voters, n_simulations)
    color_map = {'A': 0, 'B': 1, 'C': 2}
    
    xs = count_pairwise_batch(profiles, 0, 1)  # A > B
    ys = count_pairwise_batch(profiles, 1, 2)  # B > C
    zs = count_pairwise_batch(profiles, 2, 0)  # C > A
    
    colors = np.empty(n_simulations, dtype=np.int8)
    noise_mask = np.zeros(n_simulations, dtype=bool)  # False = signal, True = noise
    rule_contributions = {}  # rule_name -> disagreement count
    
    if compute_noise and noise_rules:
        # Get selected rules for noise calculation
        selected_rules = {name: VOTING_RULES[name] for name in noise_rules if name in VOTING_RULES}
        
        if len(selected_rules) < 2:
            # Need at least 2 rules to detect disagreement
            rule_func = VOTING_RULES[voting_rule]
            for i in range(n_simulations):
                winner = rule_func(profiles[i])
                colors[i] = color_map[winner]
            return xs, ys, zs, colors, noise_mask, rule_contributions
        
        rule_func = VOTING_RULES[voting_rule]
        rule_names = list(selected_rules.keys())
        
        # Initialize contribution counters
        for name in rule_names:
            rule_contributions[name] = 0
        
        for i in range(n_simulations):
            profile = profiles[i]
            
            # Get winner from selected rule for display color
            winner = rule_func(profile)
            colors[i] = color_map[winner]
            
            # Compute winners for all selected rules
            rule_winners = {}
            for name, rule in selected_rules.items():
                w = rule(profile)
                if w is not None:  # condorcet can return None
                    rule_winners[name] = w
            
            # Get unique winners
            winners_set = set(rule_winners.values())
            
            # Noise-dominated if rules disagree (more than one distinct winner)
            if len(winners_set) > 1:
                noise_mask[i] = True
                
                # Find the "majority" winner (most common among rules)
                winner_counts = {}
                for w in rule_winners.values():
                    winner_counts[w] = winner_counts.get(w, 0) + 1
                majority_winner = max(winner_counts.keys(), key=lambda w: winner_counts[w])
                
                # Count contribution: rules that picked a different winner than majority
                for name, w in rule_winners.items():
                    if w != majority_winner:
                        rule_contributions[name] += 1
    else:
        rule_func = VOTING_RULES[voting_rule]
        for i in range(n_simulations):
            winner = rule_func(profiles[i])
            colors[i] = color_map[winner]
    
    return xs, ys, zs, colors, noise_mask, rule_contributions

# Aggregate points by coordinate
def aggregate_points(xs, ys, zs, colors, use_3d, blend_mode, noise_mask=None, compute_noise=False):
    """
    Combine points at same coordinates. Returns aggregated coords, colors, sizes.
    
    If compute_noise is True, also returns noise_ratios for each aggregated point
    (fraction of profiles at that coordinate that are noise-dominated).
    """
    COLOR_RGB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.6, 0.0]])
    
    if use_3d:
        coords = np.column_stack([xs, ys, zs])
    else:
        coords = np.column_stack([xs, ys])
    
    # Find unique coordinates and group indices
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    n_unique = len(unique_coords)
    
    # Aggregate colors and counts
    agg_colors = np.zeros((n_unique, 3), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.int32)
    winner_counts = np.zeros((n_unique, 3), dtype=np.int32)
    noise_counts = np.zeros(n_unique, dtype=np.int32)  # Count of noise-dominated profiles
    
    for i, idx in enumerate(inverse):
        counts[idx] += 1
        winner_counts[idx, colors[i]] += 1
        agg_colors[idx] += COLOR_RGB[colors[i]]
        if compute_noise and noise_mask is not None and noise_mask[i]:
            noise_counts[idx] += 1
    
    if blend_mode:
        # Average color
        agg_colors = agg_colors / counts[:, np.newaxis]
    else:
        # Winner takes all - use color of most frequent winner
        dominant = np.argmax(winner_counts, axis=1)
        agg_colors = COLOR_RGB[dominant]
    
    # Scale point sizes by count (sqrt for visual balance)
    sizes = 5 + 20 * np.sqrt(counts)
    
    # Compute noise ratios (fraction of profiles that are noise-dominated)
    noise_ratios = noise_counts / counts if compute_noise else None
    
    if use_3d:
        return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], agg_colors, sizes, noise_ratios
    else:
        return unique_coords[:, 0], unique_coords[:, 1], None, agg_colors, sizes, noise_ratios

def build_noise_grid(xs, ys, zs, noise_mask, n_voters, use_3d):
    """
    Build a grid representation of noise data for region visualization.
    Returns grid coordinates and noise ratio grid.
    """
    if use_3d:
        # 3D grid
        grid_size = n_voters + 1
        noise_grid = np.full((grid_size, grid_size, grid_size), np.nan)
        count_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)
        
        for i in range(len(xs)):
            x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                if np.isnan(noise_grid[x, y, z]):
                    noise_grid[x, y, z] = 0
                count_grid[x, y, z] += 1
                if noise_mask[i]:
                    noise_grid[x, y, z] += 1
        
        # Convert to ratios
        valid = count_grid > 0
        noise_grid[valid] = noise_grid[valid] / count_grid[valid]
        
        return noise_grid, count_grid
    else:
        # 2D grid
        grid_size = n_voters + 1
        noise_grid = np.full((grid_size, grid_size), np.nan)
        count_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        for i in range(len(xs)):
            x, y = int(xs[i]), int(ys[i])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                if np.isnan(noise_grid[x, y]):
                    noise_grid[x, y] = 0
                count_grid[x, y] += 1
                if noise_mask[i]:
                    noise_grid[x, y] += 1
        
        # Convert to ratios
        valid = count_grid > 0
        noise_grid[valid] = noise_grid[valid] / count_grid[valid]
        
        return noise_grid, count_grid

def find_noise_boundaries_2d(noise_grid, threshold=0.0):
    """
    Find boundary segments between noise and signal regions in 2D.
    Returns list of line segments [(x1,y1,x2,y2), ...] for drawing borders.
    """
    rows, cols = noise_grid.shape
    segments = []
    
    # Check each cell edge
    for i in range(rows):
        for j in range(cols):
            is_noise = not np.isnan(noise_grid[i, j]) and noise_grid[i, j] > threshold
            
            # Check right neighbor
            if j + 1 < cols:
                neighbor_noise = not np.isnan(noise_grid[i, j+1]) and noise_grid[i, j+1] > threshold
                if is_noise != neighbor_noise:
                    # Boundary between (i,j) and (i,j+1) - vertical line at x=j+1
                    segments.append((j + 1, i, j + 1, i + 1))
            
            # Check bottom neighbor  
            if i + 1 < rows:
                neighbor_noise = not np.isnan(noise_grid[i+1, j]) and noise_grid[i+1, j] > threshold
                if is_noise != neighbor_noise:
                    # Boundary between (i,j) and (i+1,j) - horizontal line at y=i+1
                    segments.append((j, i + 1, j + 1, i + 1))
            
            # Add outer edges if this cell has data
            if not np.isnan(noise_grid[i, j]):
                # Left edge (j=0 or left neighbor has no data)
                if j == 0 or np.isnan(noise_grid[i, j-1]):
                    if is_noise:
                        segments.append((j, i, j, i + 1))
                # Top edge (i=0 or top neighbor has no data)
                if i == 0 or np.isnan(noise_grid[i-1, j]):
                    if is_noise:
                        segments.append((j, i, j + 1, i))
                # Right edge (j=cols-1 or right neighbor has no data)
                if j == cols - 1 or np.isnan(noise_grid[i, j+1]):
                    if is_noise:
                        segments.append((j + 1, i, j + 1, i + 1))
                # Bottom edge (i=rows-1 or bottom neighbor has no data)
                if i == rows - 1 or np.isnan(noise_grid[i+1, j]):
                    if is_noise:
                        segments.append((j, i + 1, j + 1, i + 1))
    
    return segments

def find_noise_boundaries_3d(noise_grid, threshold=0.0):
    """
    Find boundary faces between noise and signal regions in 3D.
    Returns list of face vertices for drawing as Poly3DCollection.
    """
    nx, ny, nz = noise_grid.shape
    faces = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                is_noise = not np.isnan(noise_grid[i, j, k]) and noise_grid[i, j, k] > threshold
                
                if not is_noise:
                    continue
                
                # Check each of 6 neighbors - add face if neighbor is signal or empty
                # +X face
                if i + 1 >= nx or np.isnan(noise_grid[i+1, j, k]) or noise_grid[i+1, j, k] <= threshold:
                    faces.append([
                        [i+1, j, k], [i+1, j+1, k], [i+1, j+1, k+1], [i+1, j, k+1]
                    ])
                # -X face
                if i - 1 < 0 or np.isnan(noise_grid[i-1, j, k]) or noise_grid[i-1, j, k] <= threshold:
                    faces.append([
                        [i, j, k], [i, j, k+1], [i, j+1, k+1], [i, j+1, k]
                    ])
                # +Y face
                if j + 1 >= ny or np.isnan(noise_grid[i, j+1, k]) or noise_grid[i, j+1, k] <= threshold:
                    faces.append([
                        [i, j+1, k], [i, j+1, k+1], [i+1, j+1, k+1], [i+1, j+1, k]
                    ])
                # -Y face
                if j - 1 < 0 or np.isnan(noise_grid[i, j-1, k]) or noise_grid[i, j-1, k] <= threshold:
                    faces.append([
                        [i, j, k], [i+1, j, k], [i+1, j, k+1], [i, j, k+1]
                    ])
                # +Z face
                if k + 1 >= nz or np.isnan(noise_grid[i, j, k+1]) or noise_grid[i, j, k+1] <= threshold:
                    faces.append([
                        [i, j, k+1], [i+1, j, k+1], [i+1, j+1, k+1], [i, j+1, k+1]
                    ])
                # -Z face
                if k - 1 < 0 or np.isnan(noise_grid[i, j, k-1]) or noise_grid[i, j, k-1] <= threshold:
                    faces.append([
                        [i, j, k], [i, j+1, k], [i+1, j+1, k], [i+1, j, k]
                    ])
    
    return faces

# Current settings
current_settings = {
    'n_voters': 100,
    'n_simulations': 1000,
    'voting_rule': 'plurality',
    'use_3d': False,
    'blend_mode': True,
    'compute_noise': False,
    'noise_rules': set(VOTING_RULES.keys()),  # All rules selected by default
    'noise_display': 'points'  # 'points' or 'region'
}

# Store simulation data
sim_data = {'xs': [], 'ys': [], 'zs': [], 'colors': [], 'noise_mask': [], 'rule_contributions': {}}

# Create figure (wider to accommodate noise rule selection)
fig = plt.figure(figsize=(16, 9))
ax = None
ax_contrib_text = None  # Axes for contribution table text

def create_axes():
    global ax, ax_contrib_text
    if ax is not None:
        ax.remove()
    if ax_contrib_text is not None:
        ax_contrib_text.remove()
        ax_contrib_text = None
    
    if current_settings['use_3d']:
        ax = fig.add_axes([0.05, 0.4, 0.4, 0.5], projection='3d')
    else:
        ax = fig.add_axes([0.05, 0.4, 0.4, 0.5])
    
    # Create contribution table axes only when noise mode is on
    if current_settings['compute_noise']:
        ax_contrib_text = fig.add_axes([0.47, 0.4, 0.18, 0.5])
        ax_contrib_text.axis('off')

def update_plot(rerun=True, update_zoom=True):
    global sim_data
    
    n = int(slider_voters.val)
    compute_noise = current_settings['compute_noise']
    noise_rules = current_settings['noise_rules']
    
    if rerun:
        xs, ys, zs, colors, noise_mask, rule_contributions = run_simulation(
            n, int(slider_sims.val), current_settings['voting_rule'], compute_noise, noise_rules
        )
        sim_data = {
            'xs': xs, 'ys': ys, 'zs': zs, 'colors': colors, 
            'noise_mask': noise_mask, 'rule_contributions': rule_contributions
        }
        if update_zoom:
            slider_zoom_x.set_val((0, n))
            slider_zoom_y.set_val((0, n))
            slider_zoom_z.set_val((0, n))
    
    create_axes()
    
    # Aggregate points
    agg_x, agg_y, agg_z, agg_c, sizes, noise_ratios = aggregate_points(
        sim_data['xs'], sim_data['ys'], sim_data['zs'], sim_data['colors'],
        current_settings['use_3d'], current_settings['blend_mode'],
        sim_data['noise_mask'], compute_noise
    )
    
    mode_str = "Blend" if current_settings['blend_mode'] else "Winner"
    
    xmin, xmax = slider_zoom_x.val
    ymin, ymax = slider_zoom_y.val
    zmin, zmax = slider_zoom_z.val
    
    if compute_noise and noise_ratios is not None:
        # Calculate noise statistics
        total_profiles = len(sim_data['noise_mask'])
        noise_count = np.sum(sim_data['noise_mask'])
        signal_count = total_profiles - noise_count
        noise_pct = 100 * noise_count / total_profiles if total_profiles > 0 else 0
        
        n_rules = len(noise_rules)
        display_mode = current_settings['noise_display']
        title = f"{current_settings['voting_rule'].replace('_', ' ').title()} [{mode_str}] | {display_mode.title()}\n"
        title += f"Signal: {signal_count} ({100-noise_pct:.1f}%) | Noise: {noise_count} ({noise_pct:.1f}%)"
        
        # Display contribution table
        if ax_contrib_text is not None and sim_data['rule_contributions']:
            contrib = sim_data['rule_contributions']
            
            # Build table text
            table_text = "NOISE CONTRIBUTION\n"
            table_text += "─" * 28 + "\n"
            table_text += f"{'Rule':<18} {'Count':>5} {'%':>4}\n"
            table_text += "─" * 28 + "\n"
            
            # Sort by contribution (descending)
            sorted_rules = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
            
            for rule_name, count in sorted_rules:
                pct = 100 * count / noise_count if noise_count > 0 else 0
                # Shorten rule name if needed
                display_name = rule_name.replace('_', ' ').title()
                if len(display_name) > 16:
                    display_name = display_name[:14] + '..'
                table_text += f"{display_name:<18} {count:>5} {pct:>3.0f}%\n"
            
            table_text += "─" * 28 + "\n"
            table_text += f"{'Total noise profiles:':<18} {noise_count:>5}\n"
            table_text += f"{'Rules used:':<18} {n_rules:>5}"
            
            ax_contrib_text.text(0.05, 0.95, table_text, transform=ax_contrib_text.transAxes,
                                fontsize=9, fontfamily='monospace', verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Separate signal-dominated and noise-dominated points
        signal_mask = noise_ratios == 0
        noise_mask_agg = noise_ratios > 0
        
        if display_mode == 'region':
            # Region-based visualization with yellow noise regions and borders
            noise_grid, count_grid = build_noise_grid(
                sim_data['xs'], sim_data['ys'], sim_data['zs'], 
                sim_data['noise_mask'], n, current_settings['use_3d']
            )
            
            if current_settings['use_3d']:
                # 3D region visualization
                # First plot all points with their colors
                ax.scatter(agg_x, agg_y, agg_z, c=agg_c, s=sizes, alpha=0.6)
                
                # Find and draw noise boundary faces
                faces = find_noise_boundaries_3d(noise_grid, threshold=0.0)
                if faces:
                    # Create face collection with yellow coloring
                    face_collection = Poly3DCollection(
                        faces, alpha=0.4, facecolor='gold', 
                        edgecolor='darkgoldenrod', linewidth=0.8
                    )
                    ax.add_collection3d(face_collection)
                
                ax.set_xlabel('A > B')
                ax.set_ylabel('B > C')
                ax.set_zlabel('C > A')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_zlim(zmin, zmax)
            else:
                # 2D region visualization
                # Create mesh grid for pcolormesh
                grid_x = np.arange(noise_grid.shape[1] + 1)
                grid_y = np.arange(noise_grid.shape[0] + 1)
                
                # Plot noise regions as filled cells with yellow colormap
                # Transpose for correct orientation (pcolormesh expects [y, x])
                masked_grid = np.ma.masked_invalid(noise_grid)
                ax.pcolormesh(grid_x, grid_y, masked_grid, cmap=NOISE_CMAP, 
                             vmin=0, vmax=1, shading='flat', zorder=1)
                
                # Plot all points on top
                ax.scatter(agg_x, agg_y, c=agg_c, s=sizes, alpha=0.7, zorder=3)
                
                # Draw boundary lines between noise and signal regions
                segments = find_noise_boundaries_2d(noise_grid, threshold=0.0)
                for x1, y1, x2, y2 in segments:
                    ax.plot([x1, x2], [y1, y2], color='darkgoldenrod', 
                           linewidth=2, zorder=2, solid_capstyle='round')
                
                ax.set_xlabel('Voters preferring A over B')
                ax.set_ylabel('Voters preferring B over C')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
        else:
            # Points-based visualization (original)
            if current_settings['use_3d']:
                # Plot signal-dominated points normally
                if np.any(signal_mask):
                    ax.scatter(agg_x[signal_mask], agg_y[signal_mask], agg_z[signal_mask], 
                              c=agg_c[signal_mask], s=sizes[signal_mask], alpha=0.8, label='Signal')
                
                # Plot noise-dominated points with black edge and reduced alpha
                if np.any(noise_mask_agg):
                    ax.scatter(agg_x[noise_mask_agg], agg_y[noise_mask_agg], agg_z[noise_mask_agg],
                              c=agg_c[noise_mask_agg], s=sizes[noise_mask_agg] * 1.2, 
                              alpha=0.5, edgecolors='black', linewidths=1.5, label='Noise')
                
                ax.set_xlabel('A > B')
                ax.set_ylabel('B > C')
                ax.set_zlabel('C > A')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_zlim(zmin, zmax)
            else:
                # Plot signal-dominated points normally
                if np.any(signal_mask):
                    ax.scatter(agg_x[signal_mask], agg_y[signal_mask], 
                              c=agg_c[signal_mask], s=sizes[signal_mask], alpha=0.8, label='Signal')
                
                # Plot noise-dominated points with black edge and visual distinction
                if np.any(noise_mask_agg):
                    ax.scatter(agg_x[noise_mask_agg], agg_y[noise_mask_agg],
                              c=agg_c[noise_mask_agg], s=sizes[noise_mask_agg] * 1.2, 
                              alpha=0.5, edgecolors='black', linewidths=1.5, label='Noise')
                
                ax.set_xlabel('Voters preferring A over B')
                ax.set_ylabel('Voters preferring B over C')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                
                # Add legend only if we have both types
                if np.any(signal_mask) and np.any(noise_mask_agg):
                    ax.legend(loc='upper right', fontsize=8)
    else:
        title = f"{current_settings['voting_rule'].replace('_', ' ').title()} [{mode_str}]\n(Red=A, Blue=B, Green=C) | {len(agg_x)} points"
        
        if current_settings['use_3d']:
            ax.scatter(agg_x, agg_y, agg_z, c=agg_c, s=sizes, alpha=0.7)
            ax.set_xlabel('A > B')
            ax.set_ylabel('B > C')
            ax.set_zlabel('C > A')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
        else:
            ax.scatter(agg_x, agg_y, c=agg_c, s=sizes, alpha=0.7)
            ax.set_xlabel('Voters preferring A over B')
            ax.set_ylabel('Voters preferring B over C')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    
    ax.set_title(title)
    fig.canvas.draw_idle()

# Slider for number of voters
ax_voters = fig.add_axes([0.08, 0.28, 0.35, 0.025])
slider_voters = Slider(ax_voters, 'Voters', 10, 500, valinit=100, valstep=10)

# Slider for number of simulations
ax_sims = fig.add_axes([0.08, 0.23, 0.35, 0.025])
slider_sims = Slider(ax_sims, 'Simulations', 100, 10000, valinit=1000, valstep=100)

# Zoom sliders (range sliders)
ax_zoom_x = fig.add_axes([0.08, 0.15, 0.35, 0.025])
slider_zoom_x = RangeSlider(ax_zoom_x, 'Zoom X', 0, 100, valinit=(0, 100))

ax_zoom_y = fig.add_axes([0.08, 0.10, 0.35, 0.025])
slider_zoom_y = RangeSlider(ax_zoom_y, 'Zoom Y', 0, 100, valinit=(0, 100))

ax_zoom_z = fig.add_axes([0.08, 0.05, 0.35, 0.025])
slider_zoom_z = RangeSlider(ax_zoom_z, 'Zoom Z (3D)', 0, 100, valinit=(0, 100))

# Radio buttons for voting rule selection (display rule)
ax_radio = fig.add_axes([0.68, 0.55, 0.14, 0.35])
ax_radio.set_title('Display Rule', fontsize=9)
radio = RadioButtons(ax_radio, list(VOTING_RULES.keys()), active=0)

# Noise rules selection checkboxes
ax_noise_rules = fig.add_axes([0.84, 0.55, 0.14, 0.35])
ax_noise_rules.set_title('Noise Rules', fontsize=9)
noise_rule_names = list(VOTING_RULES.keys())
check_noise_rules = CheckButtons(ax_noise_rules, noise_rule_names, [True] * len(noise_rule_names))

# Run button
ax_button = fig.add_axes([0.68, 0.42, 0.1, 0.05])
button = Button(ax_button, 'Run')

# 3D toggle checkbox
ax_check = fig.add_axes([0.80, 0.42, 0.12, 0.05])
check_3d = CheckButtons(ax_check, ['3D View'], [False])

# Reset zoom button
ax_reset = fig.add_axes([0.68, 0.34, 0.1, 0.05])
button_reset = Button(ax_reset, 'Reset Zoom')

# Color mode radio buttons
ax_color_mode = fig.add_axes([0.80, 0.26, 0.12, 0.12])
radio_color = RadioButtons(ax_color_mode, ['Blend', 'Winner'], active=0)

# Noise mode toggle checkbox
ax_noise = fig.add_axes([0.68, 0.26, 0.1, 0.05])
check_noise = CheckButtons(ax_noise, ['Noise Mode'], [False])

# Noise display mode radio buttons (Points vs Region)
ax_noise_display = fig.add_axes([0.68, 0.12, 0.1, 0.1])
ax_noise_display.set_title('Noise View', fontsize=8)
radio_noise_display = RadioButtons(ax_noise_display, ['Points', 'Region'], active=0)

# Debounce timer for settings changes (0.5 second delay)
DEBOUNCE_MS = 500  # milliseconds
debounce_timer = None
pending_update = {'rerun': False, 'update_zoom': False}

def debounced_update(rerun=True, update_zoom=True):
    """Schedule an update with debouncing. Resets timer if called again within DEBOUNCE_MS."""
    global debounce_timer, pending_update
    
    # Accumulate update flags (use most demanding option)
    pending_update['rerun'] = pending_update['rerun'] or rerun
    pending_update['update_zoom'] = pending_update['update_zoom'] or update_zoom
    
    # Stop existing timer if running
    if debounce_timer is not None:
        debounce_timer.stop()
    
    # Create and start new timer
    debounce_timer = fig.canvas.new_timer(interval=DEBOUNCE_MS)
    debounce_timer.add_callback(execute_pending_update)
    debounce_timer.single_shot = True
    debounce_timer.start()

def execute_pending_update():
    """Execute the pending update after debounce delay."""
    global pending_update, debounce_timer
    
    rerun = pending_update['rerun']
    update_zoom = pending_update['update_zoom']
    
    # Reset pending state
    pending_update = {'rerun': False, 'update_zoom': False}
    debounce_timer = None
    
    # Execute the actual update
    update_plot(rerun=rerun, update_zoom=update_zoom)

def on_radio_change(label):
    current_settings['voting_rule'] = label
    debounced_update(rerun=True, update_zoom=False)

def on_button_click(event):
    # Run button should execute immediately without debounce
    update_plot(rerun=True, update_zoom=False)

def on_3d_toggle(label):
    current_settings['use_3d'] = not current_settings['use_3d']
    debounced_update(rerun=False, update_zoom=False)

def on_zoom_change(val):
    debounced_update(rerun=False, update_zoom=False)

def on_reset_zoom(event):
    n = int(slider_voters.val)
    slider_zoom_x.set_val((0, n))
    slider_zoom_y.set_val((0, n))
    slider_zoom_z.set_val((0, n))
    debounced_update(rerun=False, update_zoom=False)

def on_voters_change(val):
    n = int(val)
    slider_zoom_x.valmin, slider_zoom_x.valmax = 0, n
    slider_zoom_y.valmin, slider_zoom_y.valmax = 0, n
    slider_zoom_z.valmin, slider_zoom_z.valmax = 0, n
    slider_zoom_x.ax.set_xlim(0, n)
    slider_zoom_y.ax.set_xlim(0, n)
    slider_zoom_z.ax.set_xlim(0, n)
    slider_zoom_x.set_val((0, n))
    slider_zoom_y.set_val((0, n))
    slider_zoom_z.set_val((0, n))
    debounced_update(rerun=True, update_zoom=True)

def on_color_mode_change(label):
    current_settings['blend_mode'] = (label == 'Blend')
    debounced_update(rerun=False, update_zoom=False)

def on_noise_toggle(label):
    current_settings['compute_noise'] = not current_settings['compute_noise']
    # Noise toggle requires rerun to recompute winners across all rules
    debounced_update(rerun=True, update_zoom=False)

def on_noise_rules_change(label):
    """Toggle a rule's inclusion in noise calculation."""
    if label in current_settings['noise_rules']:
        current_settings['noise_rules'].remove(label)
    else:
        current_settings['noise_rules'].add(label)
    
    # If noise mode is on, rerun with new rule set
    if current_settings['compute_noise']:
        debounced_update(rerun=True, update_zoom=False)

def on_noise_display_change(label):
    """Switch between 'points' and 'region' noise visualization modes."""
    current_settings['noise_display'] = label.lower()
    # Just redraw, no need to rerun simulation
    if current_settings['compute_noise']:
        debounced_update(rerun=False, update_zoom=False)

radio.on_clicked(on_radio_change)
button.on_clicked(on_button_click)
check_3d.on_clicked(on_3d_toggle)
slider_zoom_x.on_changed(on_zoom_change)
slider_zoom_y.on_changed(on_zoom_change)
slider_zoom_z.on_changed(on_zoom_change)
button_reset.on_clicked(on_reset_zoom)
slider_voters.on_changed(on_voters_change)
radio_color.on_clicked(on_color_mode_change)
check_noise.on_clicked(on_noise_toggle)
check_noise_rules.on_clicked(on_noise_rules_change)
radio_noise_display.on_clicked(on_noise_display_change)

# Initial plot
update_plot()
plt.show()
