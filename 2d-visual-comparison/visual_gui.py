"""
2D Visualization GUI for Voting Simulation.

Interactive matplotlib-based visualization showing:
- Voter and candidate positions in 2D space
- Distance metric used by each voter (color-coded)
- Voting lines from voters to their chosen candidates
- Distance values displayed on lines
- Configurable settings panel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from matplotlib.patches import Rectangle
import sys
import os

# Add parent directory to path to import simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator import (
    SimulationConfig, GeometryConfig, UtilityConfig,
    HeterogeneousDistanceConfig
)
from simulator.utility import UtilityComputer
from simulator.geometry import GeometryGenerator
from simulator.voting_rules import VotingRuleEngine, VotingRuleConfig, get_rule_type, RuleType
from simulator.utility import utilities_to_rankings


# Color mapping for distance metrics
METRIC_COLORS = {
    'l1': '#FF6B6B',      # Red
    'l2': '#4ECDC4',      # Teal
    'cosine': '#95E1D3',  # Light teal
    'chebyshev': '#F38181'  # Pink
}

METRIC_NAMES = {
    'l1': 'L1 (Manhattan)',
    'l2': 'L2 (Euclidean)',
    'cosine': 'Cosine',
    'chebyshev': 'Chebyshev'
}


class Visual2DGUI:
    """Interactive 2D visualization GUI for voting simulation."""
    
    def __init__(self):
        """Initialize the GUI."""
        # Default configuration
        self.config = SimulationConfig(
            n_profiles=1,
            n_voters=25,
            n_candidates=3,
            voting_rules=['plurality'],
            geometry=GeometryConfig(method='uniform', n_dim=2),
            utility=UtilityConfig(
                function='gaussian',
                distance_metric='l2',
                heterogeneous_distance=HeterogeneousDistanceConfig(enabled=False)
            ),
            rng_seed=42
        )
        
        # Current profile data
        self.voter_positions = None
        self.candidate_positions = None
        self.utilities = None
        self.rankings = None
        self.voter_metrics = None
        self.voting_result = None
        self.voter_choices = None  # Which candidate each voter votes for
        self.vote_counts = None  # Vote counts per candidate
        
        # Matplotlib setup
        self.fig = None
        self.ax = None
        self.stats_ax = None
        self.setup_ui()
        
        # Generate initial profile
        self.generate_profile()
        self.update_visualization()
    
    def setup_ui(self):
        """Set up the matplotlib UI with controls."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.suptitle('2D Voting Simulation Visualization', fontsize=16, fontweight='bold')
        
        # Main plot area (larger)
        self.ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Dimension 1', fontsize=12)
        self.ax.set_ylabel('Dimension 2', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Voter and Candidate Positions', fontsize=14)
        
        # Statistics panel (below main plot)
        self.stats_ax = plt.subplot2grid((4, 4), (3, 0), colspan=3, rowspan=1)
        self.stats_ax.axis('off')
        
        # Control panel area
        control_ax = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        control_ax.axis('off')
        
        # Create control widgets
        self.create_controls(control_ax)
    
    def create_controls(self, parent_ax):
        """Create control widgets."""
        # Geometry method
        geom_label_ax = plt.axes([0.78, 0.95, 0.20, 0.03])
        geom_label_ax.text(0.5, 0.5, 'Geometry Method', ha='center', va='center', 
                          fontsize=10, fontweight='bold', transform=geom_label_ax.transAxes)
        geom_label_ax.axis('off')
        
        geom_ax = plt.axes([0.78, 0.87, 0.20, 0.08])
        self.geom_radio = RadioButtons(geom_ax, ('uniform', 'clustered', 'polarized', '2d'),
                                       active=0)
        self.geom_radio.on_clicked(self.on_geometry_change)
        
        # Number of voters
        voters_label_ax = plt.axes([0.78, 0.80, 0.20, 0.03])
        voters_label_ax.text(0.5, 0.5, 'Number of Voters', ha='center', va='center',
                            fontsize=10, fontweight='bold', transform=voters_label_ax.transAxes)
        voters_label_ax.axis('off')
        
        voters_ax = plt.axes([0.78, 0.75, 0.20, 0.04])
        self.voters_slider = Slider(voters_ax, '', 5, 100, valinit=25, valstep=1)
        self.voters_slider.on_changed(self.on_voters_change)
        
        # Number of candidates
        cands_label_ax = plt.axes([0.78, 0.68, 0.20, 0.03])
        cands_label_ax.text(0.5, 0.5, 'Number of Candidates', ha='center', va='center',
                          fontsize=10, fontweight='bold', transform=cands_label_ax.transAxes)
        cands_label_ax.axis('off')
        
        cands_ax = plt.axes([0.78, 0.63, 0.20, 0.04])
        self.cands_slider = Slider(cands_ax, '', 2, 6, valinit=3, valstep=1)
        self.cands_slider.on_changed(self.on_candidates_change)
        
        # Voting rule
        rule_label_ax = plt.axes([0.78, 0.56, 0.20, 0.03])
        rule_label_ax.text(0.5, 0.5, 'Voting Rule', ha='center', va='center',
                           fontsize=10, fontweight='bold', transform=rule_label_ax.transAxes)
        rule_label_ax.axis('off')
        
        rule_ax = plt.axes([0.78, 0.45, 0.20, 0.10])
        self.rule_radio = RadioButtons(rule_ax, ('plurality', 'borda', 'irv', 'approval', 'star', 'schulze'),
                                      active=0)
        self.rule_radio.on_clicked(self.on_rule_change)
        
        # Distance metric (homogeneous)
        metric_label_ax = plt.axes([0.78, 0.38, 0.20, 0.03])
        metric_label_ax.text(0.5, 0.5, 'Distance Metric', ha='center', va='center',
                            fontsize=10, fontweight='bold', transform=metric_label_ax.transAxes)
        metric_label_ax.axis('off')
        
        metric_ax = plt.axes([0.78, 0.30, 0.20, 0.08])
        self.metric_radio = RadioButtons(metric_ax, ('l1', 'l2', 'cosine', 'chebyshev'),
                                         active=1)
        self.metric_radio.on_clicked(self.on_metric_change)
        
        # Heterogeneous distance toggle
        het_label_ax = plt.axes([0.78, 0.23, 0.20, 0.03])
        het_label_ax.text(0.5, 0.5, 'Heterogeneous Distance', ha='center', va='center',
                         fontsize=10, fontweight='bold', transform=het_label_ax.transAxes)
        het_label_ax.axis('off')
        
        het_ax = plt.axes([0.78, 0.15, 0.20, 0.08])
        self.het_radio = RadioButtons(het_ax, ('Disabled', 'Center-Extreme', 'Radial-Steps'),
                                     active=0)
        self.het_radio.on_clicked(self.on_heterogeneous_change)
        
        # Buttons
        regenerate_ax = plt.axes([0.78, 0.07, 0.09, 0.04])
        self.regenerate_btn = Button(regenerate_ax, 'Regenerate')
        self.regenerate_btn.on_clicked(self.on_regenerate)
        
        update_ax = plt.axes([0.89, 0.07, 0.09, 0.04])
        self.update_btn = Button(update_ax, 'Update')
        self.update_btn.on_clicked(self.on_update)
    
    def on_geometry_change(self, label):
        """Handle geometry method change."""
        self.config.geometry.method = label
        self.generate_profile()
        self.update_visualization()
    
    def on_voters_change(self, val):
        """Handle number of voters change."""
        self.config.n_voters = int(val)
        self.generate_profile()
        self.update_visualization()
    
    def on_candidates_change(self, val):
        """Handle number of candidates change."""
        self.config.n_candidates = int(val)
        self.generate_profile()
        self.update_visualization()
    
    def on_rule_change(self, label):
        """Handle voting rule change."""
        self.config.voting_rules = [label]
        self.update_visualization()
    
    def on_metric_change(self, label):
        """Handle distance metric change."""
        if not self.config.utility.heterogeneous_distance.enabled:
            self.config.utility.distance_metric = label
            self.generate_profile()
            self.update_visualization()
    
    def on_heterogeneous_change(self, label):
        """Handle heterogeneous distance toggle."""
        het_config = self.config.utility.heterogeneous_distance
        
        if label == 'Disabled':
            het_config.enabled = False
        elif label == 'Center-Extreme':
            het_config.enabled = True
            het_config.strategy = 'center_extreme'
            het_config.center_metric = 'l2'
            het_config.extreme_metric = 'cosine'
            het_config.extreme_threshold = 0.5
        elif label == 'Radial-Steps':
            het_config.enabled = True
            het_config.strategy = 'radial_steps'
            het_config.radial_metrics = ['l1', 'l2', 'chebyshev']
            het_config.radial_scaling = 'linear'
            het_config.scaling_parameter = 2.0
        
        self.generate_profile()
        self.update_visualization()
    
    def on_regenerate(self, event):
        """Regenerate profile with new random seed."""
        self.config.rng_seed = np.random.randint(0, 10000)
        self.generate_profile()
        self.update_visualization()
    
    def on_update(self, event):
        """Update visualization without regenerating."""
        self.update_visualization()
    
    def generate_profile(self):
        """Generate a new preference profile."""
        # Generate spatial positions
        rng = np.random.default_rng(self.config.rng_seed)
        geometry_gen = GeometryGenerator(self.config.geometry, rng)
        spatial_profile = geometry_gen.generate(1, self.config.n_voters, self.config.n_candidates)
        
        # Extract single profile
        self.voter_positions = spatial_profile.voter_positions[0]  # (n_voters, 2)
        self.candidate_positions = spatial_profile.candidate_positions[0]  # (n_candidates, 2)
        
        # Compute utilities
        utility_computer = UtilityComputer(self.config.utility)
        distances = utility_computer.compute_distances(
            self.voter_positions,
            self.candidate_positions
        )
        self.utilities = utility_computer.compute_utilities(distances, 2)
        self.rankings = utilities_to_rankings(self.utilities, self.config.epsilon)
        
        # Get voter metrics (for heterogeneous distance)
        self.voter_metrics = utility_computer.get_voter_metrics(self.voter_positions)
        if self.voter_metrics is None:
            # Homogeneous - all use same metric
            metric = self.config.utility.distance_metric
            self.voter_metrics = np.array([metric] * self.config.n_voters)
    
    def compute_voting(self):
        """Compute voting results and determine voter choices."""
        voting_engine = VotingRuleEngine(self.config.voting_rule_config)
        rule_name = self.config.voting_rules[0]
        rule_type = get_rule_type(rule_name)
        
        # Apply voting rule
        if rule_type == RuleType.CARDINAL:
            self.voting_result = voting_engine.apply_rule(rule_name, utilities=self.utilities)
        else:
            self.voting_result = voting_engine.apply_rule(rule_name, rankings=self.rankings)
        
        # Determine which candidate each voter "votes for"
        # For ordinal rules: first choice
        # For cardinal rules: depends on rule
        self.voter_choices = self._get_voter_choices(rule_name, rule_type)
        
        # Compute vote counts
        self.vote_counts = self._compute_vote_counts(rule_name, rule_type)
    
    def _get_voter_choices(self, rule_name: str, rule_type: RuleType) -> np.ndarray:
        """Get which candidate each voter votes for."""
        if rule_type == RuleType.ORDINAL:
            # For ordinal rules, voter votes for their first choice
            return self.rankings[:, 0]
        
        # Cardinal rules
        if rule_name == 'approval':
            # For approval, voter votes for all approved candidates
            # We'll show line to their top approved candidate
            policy = self.config.voting_rule_config.approval_policy
            k = self.config.voting_rule_config.approval_k
            
            if policy == 'top_k':
                # Top k candidates
                top_k_indices = np.argsort(-self.utilities, axis=1)[:, :k]
                return top_k_indices[:, 0]  # Show line to top approved
            elif policy == 'mean':
                # Above mean utility
                means = np.mean(self.utilities, axis=1, keepdims=True)
                approved = self.utilities > means
                # Get top approved candidate (fallback to top choice if none approved)
                approved_utilities = np.where(approved, self.utilities, -np.inf)
                choices = np.argmax(approved_utilities, axis=1)
                # If no approved candidates, use top choice
                no_approved = ~np.any(approved, axis=1)
                choices[no_approved] = np.argmax(self.utilities[no_approved], axis=1)
                return choices
            else:
                # Default to top choice
                return np.argmax(self.utilities, axis=1)
        
        elif rule_name in ['score', 'star']:
            # For score/star, voter votes for highest utility candidate
            return np.argmax(self.utilities, axis=1)
        
        elif rule_name == 'utilitarian':
            # Utilitarian: voter's preference is their top utility
            return np.argmax(self.utilities, axis=1)
        
        else:
            # Default: top utility candidate
            return np.argmax(self.utilities, axis=1)
    
    def _compute_vote_counts(self, rule_name: str, rule_type: RuleType) -> np.ndarray:
        """Compute vote counts for each candidate."""
        n_candidates = len(self.candidate_positions)
        vote_counts = np.zeros(n_candidates, dtype=int)
        
        if rule_type == RuleType.ORDINAL:
            # For ordinal rules, count first-choice votes
            first_choices = self.rankings[:, 0]
            for choice in first_choices:
                vote_counts[choice] += 1
        else:
            # For cardinal rules, depends on the rule
            if rule_name == 'approval':
                # Count approvals
                policy = self.config.voting_rule_config.approval_policy
                k = self.config.voting_rule_config.approval_k
                
                if policy == 'top_k':
                    # Count top k candidates
                    top_k_indices = np.argsort(-self.utilities, axis=1)[:, :k]
                    for voter_top_k in top_k_indices:
                        for cand_idx in voter_top_k:
                            vote_counts[cand_idx] += 1
                elif policy == 'mean':
                    # Count candidates above mean utility
                    means = np.mean(self.utilities, axis=1, keepdims=True)
                    approved = self.utilities > means
                    for voter_approved in approved:
                        for cand_idx, is_approved in enumerate(voter_approved):
                            if is_approved:
                                vote_counts[cand_idx] += 1
                else:
                    # Default: count top choice
                    top_choices = np.argmax(self.utilities, axis=1)
                    for choice in top_choices:
                        vote_counts[choice] += 1
            else:
                # For other cardinal rules, count top utility choice
                top_choices = np.argmax(self.utilities, axis=1)
                for choice in top_choices:
                    vote_counts[choice] += 1
        
        return vote_counts
    
    def compute_distances_to_choices(self) -> np.ndarray:
        """Compute distances from voters to their chosen candidates."""
        n_voters = len(self.voter_positions)
        distances = np.zeros(n_voters)
        
        utility_computer = UtilityComputer(self.config.utility)
        
        for i in range(n_voters):
            voter_pos = self.voter_positions[i:i+1]  # (1, 2)
            candidate_idx = self.voter_choices[i]
            candidate_pos = self.candidate_positions[candidate_idx:candidate_idx+1]  # (1, 2)
            
            # Get the metric this voter uses
            voter_metric = self.voter_metrics[i]
            
            # Compute distance using that metric
            if voter_metric == 'l2':
                dist = np.linalg.norm(voter_pos - candidate_pos)
            elif voter_metric == 'l1':
                dist = np.sum(np.abs(voter_pos - candidate_pos))
            elif voter_metric == 'cosine':
                v_norm = voter_pos / (np.linalg.norm(voter_pos) + 1e-10)
                c_norm = candidate_pos / (np.linalg.norm(candidate_pos) + 1e-10)
                similarity = np.sum(v_norm * c_norm)
                dist = 1.0 - similarity
            elif voter_metric == 'chebyshev':
                dist = np.max(np.abs(voter_pos - candidate_pos))
            else:
                dist = np.linalg.norm(voter_pos - candidate_pos)
            
            distances[i] = dist
        
        return distances
    
    def update_visualization(self):
        """Update the visualization."""
        # Compute voting
        self.compute_voting()
        
        # Clear axes
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Dimension 1', fontsize=12)
        self.ax.set_ylabel('Dimension 2', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        rule_name = self.config.voting_rules[0]
        winner = self.voting_result.winner if self.voting_result else None
        title = f'Voting: {rule_name.upper()}'
        if winner is not None:
            title += f' | Winner: Candidate {winner}'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Compute distances to chosen candidates
        choice_distances = self.compute_distances_to_choices()
        
        # Plot voting lines first (so they're behind markers)
        for i in range(len(self.voter_positions)):
            voter_pos = self.voter_positions[i]
            candidate_idx = self.voter_choices[i]
            candidate_pos = self.candidate_positions[candidate_idx]
            
            # Draw line with color matching voter's metric
            metric = self.voter_metrics[i]
            line_color = METRIC_COLORS.get(metric, 'gray')
            self.ax.plot([voter_pos[0], candidate_pos[0]], 
                        [voter_pos[1], candidate_pos[1]],
                        color=line_color, alpha=0.4, linewidth=1.0, zorder=1)
            
            # Add distance label at midpoint (slightly offset to avoid overlap)
            mid_x = (voter_pos[0] + candidate_pos[0]) / 2
            mid_y = (voter_pos[1] + candidate_pos[1]) / 2
            distance_val = choice_distances[i]
            
            # Offset label perpendicular to line to avoid overlap
            dx = candidate_pos[0] - voter_pos[0]
            dy = candidate_pos[1] - voter_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Perpendicular offset (rotate 90 degrees)
                offset_x = -dy / length * 0.02
                offset_y = dx / length * 0.02
            else:
                offset_x = offset_y = 0
            
            self.ax.text(mid_x + offset_x, mid_y + offset_y, f'{distance_val:.2f}', 
                        fontsize=7, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.8, edgecolor=line_color, linewidth=0.5),
                        zorder=3, color='black', weight='bold')
        
        # Plot candidates (larger, distinct markers)
        for i, pos in enumerate(self.candidate_positions):
            color = 'red' if (winner is not None and i == winner) else 'blue'
            marker = 's'  # square for candidates
            size = 200 if (winner is not None and i == winner) else 150
            self.ax.scatter(pos[0], pos[1], c=color, marker=marker, 
                           s=size, edgecolors='black', linewidths=2, 
                           zorder=5, label=f'Candidate {i}' if i < 3 else '')
        
        # Plot voters (colored by their distance metric)
        for i, pos in enumerate(self.voter_positions):
            metric = self.voter_metrics[i]
            color = METRIC_COLORS.get(metric, 'gray')
            self.ax.scatter(pos[0], pos[1], c=color, marker='o', 
                           s=80, edgecolors='black', linewidths=1, 
                           alpha=0.8, zorder=4)
        
        # Create legend for metrics
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=METRIC_COLORS['l1'], label='L1 (Manhattan)'),
            Patch(facecolor=METRIC_COLORS['l2'], label='L2 (Euclidean)'),
            Patch(facecolor=METRIC_COLORS['cosine'], label='Cosine'),
            Patch(facecolor=METRIC_COLORS['chebyshev'], label='Chebyshev'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Add info text
        info_text = f"Voters: {len(self.voter_positions)} | Candidates: {len(self.candidate_positions)}"
        if self.config.utility.heterogeneous_distance.enabled:
            metric_dist = UtilityComputer(self.config.utility).get_metric_distribution(
                self.voter_positions
            )
            info_text += f"\nHeterogeneous: {self.config.utility.heterogeneous_distance.strategy}"
        else:
            info_text += f"\nMetric: {self.config.utility.distance_metric.upper()}"
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Update statistics display
        self.update_statistics()
        
        self.fig.canvas.draw()
    
    def update_statistics(self):
        """Update the statistics panel with vote counts."""
        self.stats_ax.clear()
        self.stats_ax.axis('off')
        
        if self.vote_counts is None or self.voting_result is None:
            return
        
        rule_name = self.config.voting_rules[0]
        winner = self.voting_result.winner
        n_voters = len(self.voter_positions)
        
        # Title
        self.stats_ax.text(0.5, 0.95, 'Voting Statistics', 
                          ha='center', va='top', fontsize=14, fontweight='bold',
                          transform=self.stats_ax.transAxes)
        
        # Create table-like display
        y_start = 0.80
        y_spacing = 0.12
        x_left = 0.05
        x_mid = 0.50
        x_right = 0.95
        
        # Header
        self.stats_ax.text(x_left, y_start, 'Candidate', 
                          ha='left', va='top', fontsize=11, fontweight='bold',
                          transform=self.stats_ax.transAxes)
        self.stats_ax.text(x_mid, y_start, 'Votes', 
                          ha='center', va='top', fontsize=11, fontweight='bold',
                          transform=self.stats_ax.transAxes)
        self.stats_ax.text(x_right, y_start, 'Percentage', 
                          ha='right', va='top', fontsize=11, fontweight='bold',
                          transform=self.stats_ax.transAxes)
        
        # Draw separator line
        self.stats_ax.plot([x_left, x_right], [y_start - 0.02, y_start - 0.02],
                          'k-', linewidth=1, transform=self.stats_ax.transAxes)
        
        # Candidate rows
        for i in range(len(self.candidate_positions)):
            y_pos = y_start - 0.05 - (i * y_spacing)
            
            # Candidate name (highlight winner)
            cand_text = f'Candidate {i}'
            if winner is not None and i == winner:
                cand_text += ' (Winner)'
                text_color = 'red'
                font_weight = 'bold'
            else:
                text_color = 'black'
                font_weight = 'normal'
            
            self.stats_ax.text(x_left, y_pos, cand_text,
                             ha='left', va='top', fontsize=10, 
                             color=text_color, weight=font_weight,
                             transform=self.stats_ax.transAxes)
            
            # Vote count
            votes = self.vote_counts[i]
            self.stats_ax.text(x_mid, y_pos, str(votes),
                             ha='center', va='top', fontsize=10,
                             transform=self.stats_ax.transAxes)
            
            # Percentage
            if rule_name == 'approval':
                # For approval, show percentage of voters who approved this candidate
                percentage = (votes / n_voters * 100) if n_voters > 0 else 0
            else:
                # For other rules, show percentage of total votes
                total_votes = np.sum(self.vote_counts)
                percentage = (votes / total_votes * 100) if total_votes > 0 else 0
            self.stats_ax.text(x_right, y_pos, f'{percentage:.1f}%',
                             ha='right', va='top', fontsize=10,
                             transform=self.stats_ax.transAxes)
        
        # Total votes (for approval voting, may be > n_voters)
        if rule_name == 'approval':
            total_votes = np.sum(self.vote_counts)
            y_total = y_start - 0.05 - (len(self.candidate_positions) * y_spacing) - 0.05
            self.stats_ax.text(x_left, y_total, 'Total Approvals:',
                             ha='left', va='top', fontsize=9, style='italic',
                             transform=self.stats_ax.transAxes)
            self.stats_ax.text(x_mid, y_total, str(total_votes),
                             ha='center', va='top', fontsize=9, style='italic',
                             transform=self.stats_ax.transAxes)
        
        # Voting rule info
        rule_info_y = 0.15
        self.stats_ax.text(0.5, rule_info_y, f'Rule: {rule_name.upper()}',
                          ha='center', va='top', fontsize=9,
                          transform=self.stats_ax.transAxes,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    def show(self):
        """Show the GUI."""
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    gui = Visual2DGUI()
    gui.show()


if __name__ == "__main__":
    main()

