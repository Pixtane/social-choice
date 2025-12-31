"""
Enhanced visualization tools for heterogeneity research.

Provides comprehensive visualizations for:
- Spatial distribution of metrics
- Parameter sweep analysis
- Comparison between homogeneous and heterogeneous
- Statistical distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Color schemes
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
    'chebyshev': 'Chebyshev (L∞)'
}


class HeterogeneityVisualizer:
    """Main visualization class for heterogeneity analysis."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style."""
        try:
            plt.style.use(style)
        except OSError:
            # Fallback if style not available
            plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
    
    def plot_spatial_metric_distribution(
        self,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray,
        voter_metrics: np.ndarray,
        winners: Optional[Dict[str, int]] = None,
        title: str = "Spatial Distribution of Distance Metrics",
        save_path: Optional[str] = None
    ):
        """
        Plot 2D spatial distribution showing voters colored by their metric.
        
        Args:
            voter_positions: (n_voters, 2) array
            candidate_positions: (n_candidates, 2) array
            voter_metrics: (n_voters,) array of metric names
            winners: Dict mapping rule names to winner indices
            title: Plot title
            save_path: Optional path to save figure
        """
        if voter_positions.shape[1] != 2:
            raise ValueError("Spatial plot requires 2D positions")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot voters colored by metric
        unique_metrics = np.unique(voter_metrics)
        for metric in unique_metrics:
            mask = voter_metrics == metric
            positions = voter_positions[mask]
            ax.scatter(
                positions[:, 0], positions[:, 1],
                c=METRIC_COLORS.get(metric, 'gray'),
                label=METRIC_NAMES.get(metric, metric),
                s=100, alpha=0.7, edgecolors='black', linewidths=1
            )
        
        # Plot candidates
        for i, pos in enumerate(candidate_positions):
            # Determine color based on winners
            is_winner = False
            winner_rules = []
            if winners:
                for rule, winner_idx in winners.items():
                    if winner_idx == i:
                        is_winner = True
                        winner_rules.append(rule)
            
            color = 'red' if is_winner else 'blue'
            size = 300 if is_winner else 200
            marker = 's'  # square
            
            ax.scatter(
                pos[0], pos[1],
                c=color, marker=marker, s=size,
                edgecolors='black', linewidths=2,
                zorder=10
            )
            
            # Label candidate
            label = f"C{i}"
            if winner_rules:
                label += f" ({', '.join(winner_rules)})"
            ax.annotate(
                label, (pos[0], pos[1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold'
            )
        
        # Add metric distribution info
        metric_counts = {m: np.sum(voter_metrics == m) for m in unique_metrics}
        info_text = "Metric Distribution:\n"
        for metric, count in sorted(metric_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(voter_metrics)
            info_text += f"{METRIC_NAMES.get(metric, metric)}: {pct:.1f}%\n"
        
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9
        )
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_threshold_sweep(
        self,
        thresholds: np.ndarray,
        results: Dict[str, np.ndarray],
        metric_pair: str = "",
        title: str = "Threshold Sweep Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot how outcomes change with threshold parameter.
        
        Args:
            thresholds: Array of threshold values
            results: Dict mapping metric names to arrays of values
            metric_pair: Description of metric pair (e.g., "L2 + Cosine")
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('rule_disagreement', 'Rule Disagreement (%)'),
            ('vse_mean', 'VSE (Mean)'),
            ('winner_extremism', 'Winner Extremism'),
            ('condorcet_consistency', 'Condorcet Consistency (%)')
        ]
        
        for idx, (key, ylabel) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if key in results:
                values = results[key]
                ax.plot(thresholds, values, 'o-', linewidth=2, markersize=6)
                ax.fill_between(
                    thresholds, values,
                    alpha=0.2, interpolate=True
                )
            
            ax.set_xlabel('Threshold', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(thresholds.min(), thresholds.max())
        
        fig.suptitle(
            f"{title}\n{metric_pair}" if metric_pair else title,
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def plot_metric_pair_heatmap(
        self,
        metric_pairs: List[Tuple[str, str]],
        thresholds: np.ndarray,
        results: np.ndarray,  # (n_pairs, n_thresholds)
        metric_name: str = "Rule Disagreement (%)",
        title: str = "Metric Pair Comparison",
        save_path: Optional[str] = None
    ):
        """
        Create heatmap showing results for all metric pairs.
        
        Args:
            metric_pairs: List of (center_metric, extreme_metric) tuples
            thresholds: Array of threshold values
            results: 2D array of results
            metric_name: Name of the metric being visualized
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create labels for pairs
        pair_labels = [
            f"{METRIC_NAMES.get(c, c)} + {METRIC_NAMES.get(e, e)}"
            for c, e in metric_pairs
        ]
        
        # Create heatmap
        im = ax.imshow(
            results, aspect='auto', cmap='viridis',
            interpolation='nearest'
        )
        
        # Set ticks
        ax.set_xticks(np.arange(len(thresholds)))
        ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
        ax.set_yticks(np.arange(len(metric_pairs)))
        ax.set_yticklabels(pair_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, fontsize=11)
        
        # Add text annotations
        for i in range(len(metric_pairs)):
            for j in range(len(thresholds)):
                text = ax.text(
                    j, i, f"{results[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if results[i, j] > results.max() * 0.5 else "black",
                    fontsize=8
                )
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Metric Pair', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_comparison_distributions(
        self,
        homogeneous: Dict[str, np.ndarray],
        heterogeneous: Dict[str, np.ndarray],
        metric_names: List[str],
        title: str = "Homogeneous vs Heterogeneous Comparison",
        save_path: Optional[str] = None
    ):
        """
        Compare distributions between homogeneous and heterogeneous.
        
        Args:
            homogeneous: Dict mapping metric names to arrays of values
            heterogeneous: Dict mapping metric names to arrays of values
            metric_names: List of metric names to compare
            title: Plot title
            save_path: Optional path to save figure
        """
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            if metric_name in homogeneous and metric_name in heterogeneous:
                homo_data = homogeneous[metric_name]
                het_data = heterogeneous[metric_name]
                
                # Create violin plots
                parts = ax.violinplot(
                    [homo_data, het_data],
                    positions=[0, 1],
                    showmeans=True,
                    showmedians=True
                )
                
                # Color the violins
                for pc, color in zip(parts['bodies'], ['#4ECDC4', '#FF6B6B']):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Homogeneous', 'Heterogeneous'])
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                homo_mean = np.mean(homo_data)
                het_mean = np.mean(het_data)
                diff = het_mean - homo_mean
                pct_change = 100 * diff / homo_mean if homo_mean != 0 else 0
                
                stats_text = f"Δ = {diff:.3f}\n({pct_change:+.1f}%)"
                ax.text(
                    0.5, 0.95, stats_text,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9
                )
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def plot_rule_sensitivity(
        self,
        rules: List[str],
        disagreement_rates: np.ndarray,
        vse_changes: np.ndarray,
        title: str = "Voting Rule Sensitivity to Heterogeneity",
        save_path: Optional[str] = None
    ):
        """
        Plot how sensitive different voting rules are to heterogeneity.
        
        Args:
            rules: List of voting rule names
            disagreement_rates: Array of disagreement percentages
            vse_changes: Array of VSE changes (heterogeneous - homogeneous)
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Disagreement rates
        colors = ['red' if d > 50 else 'orange' if d > 30 else 'green' 
                 for d in disagreement_rates]
        bars1 = ax1.barh(rules, disagreement_rates, color=colors, alpha=0.7)
        ax1.set_xlabel('Disagreement Rate (%)', fontsize=11)
        ax1.set_title('Rule Disagreement', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.legend()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, disagreement_rates)):
            ax1.text(
                val + 1, i, f"{val:.1f}%",
                va='center', fontsize=9
            )
        
        # VSE changes
        colors2 = ['green' if v > 0 else 'red' for v in vse_changes]
        bars2 = ax2.barh(rules, vse_changes, color=colors2, alpha=0.7)
        ax2.set_xlabel('VSE Change (Heterogeneous - Homogeneous)', fontsize=11)
        ax2.set_title('VSE Impact', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, vse_changes)):
            ax2.text(
                val + (0.01 if val >= 0 else -0.01), i,
                f"{val:+.3f}",
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9
            )
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, (ax1, ax2)
    
    def plot_dimensionality_effects(
        self,
        dimensions: np.ndarray,
        results: Dict[str, np.ndarray],
        title: str = "Dimensionality Effects on Heterogeneity",
        save_path: Optional[str] = None
    ):
        """
        Plot how heterogeneity effects change with dimensionality.
        
        Args:
            dimensions: Array of dimension values
            results: Dict mapping metric names to arrays
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('rule_disagreement', 'Rule Disagreement (%)'),
            ('vse_difference', 'VSE Difference'),
            ('winner_extremism_change', 'Winner Extremism Change'),
            ('condorcet_consistency_change', 'Condorcet Consistency Change (%)')
        ]
        
        for idx, (key, ylabel) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if key in results:
                values = results[key]
                ax.plot(dimensions, values, 'o-', linewidth=2, markersize=8)
                ax.fill_between(
                    dimensions, values,
                    alpha=0.2, interpolate=True
                )
            
            ax.set_xlabel('Dimensions', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(dimensions.min() - 0.5, dimensions.max() + 0.5)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, axes


def create_summary_dashboard(
    results_data: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Create a comprehensive dashboard summarizing heterogeneity research.
    
    Args:
        results_data: Dictionary containing all experimental results
        save_path: Optional path to save figure
    """
    viz = HeterogeneityVisualizer()
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # This is a template - would need actual data to populate
    # Add various subplots based on available data
    
    plt.suptitle(
        "Heterogeneity in Distance Functions - Research Dashboard",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

