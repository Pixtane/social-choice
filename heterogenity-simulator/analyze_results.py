"""
Analysis script to process research results and generate findings.

This script analyzes the experimental results to:
1. Verify/correct original findings
2. Discover new phenomena
3. Generate statistical summaries
4. Create visualizations
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class ResultsAnalyzer:
    """Analyzer for research results."""

    def __init__(self, results_dir: str = "heterogenity-simulator/results"):
        self.results_dir = Path(results_dir)
        self.findings = {}

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def analyze_voter_scaling(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voter scaling effects."""
        analysis = {
            'stable': {},
            'changing': {},
            'trends': {}
        }

        if 'voter_counts' not in results:
            return analysis

        voter_counts = results['voter_counts']
        data = results.get('data', {})

        for rule in ['plurality', 'borda', 'irv']:
            disagreements = []
            valid_counts = []
            for n_voters in voter_counts:
                if str(n_voters) in data and rule in data[str(n_voters)]:
                    disagreements.append(data[str(n_voters)][rule]['disagreement_rate'])
                    valid_counts.append(n_voters)

            if len(disagreements) < 2:
                continue

            disagreements = np.array(disagreements)
            valid_counts = np.array(valid_counts)

            # Check if stable (low variance) or changing
            std_dev = np.std(disagreements)
            mean_val = np.mean(disagreements)
            cv = std_dev / (mean_val + 1e-10)  # Coefficient of variation

            # Check for trend
            if len(disagreements) >= 3:
                # Linear regression
                x = valid_counts
                y = disagreements
                slope = np.polyfit(x, y, 1)[0]

                analysis['trends'][rule] = {
                    'slope': float(slope),
                    'mean': float(mean_val),
                    'std': float(std_dev),
                    'cv': float(cv),
                    'min': float(np.min(disagreements)),
                    'max': float(np.max(disagreements)),
                    'range': float(np.max(disagreements) - np.min(disagreements))
                }

                if cv < 0.05:  # Very stable
                    analysis['stable'][rule] = analysis['trends'][rule]
                elif abs(slope) > 0.1:  # Significant trend
                    analysis['changing'][rule] = analysis['trends'][rule]

        return analysis

    def analyze_threshold_sweep(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threshold sweep for phase transitions."""
        analysis = {}

        thresholds = np.array(results['thresholds'])
        data = results['data']

        for rule in ['plurality', 'borda', 'irv']:
            disagreements = []
            for t in thresholds:
                key = f"{t:.2f}"
                if key in data and rule in data[key]:
                    disagreements.append(data[key][rule]['disagreement_rate'])

            if len(disagreements) < 3:
                continue

            disagreements = np.array(disagreements)

            # Find critical points
            d1 = np.diff(disagreements)
            d2 = np.diff(d1)

            # Inflection points
            inflection_indices = []
            for i in range(1, len(d2)):
                if d2[i-1] * d2[i] < 0:
                    inflection_indices.append(i+1)

            # Max curvature
            curvature = np.abs(d2)
            max_curvature_idx = np.argmax(curvature) if len(curvature) > 0 else None

            # Max jump
            max_jump = np.max(np.abs(d1))
            jump_idx = np.argmax(np.abs(d1))

            # Peak and valley
            peak_idx = np.argmax(disagreements)
            valley_idx = np.argmin(disagreements)

            analysis[rule] = {
                'max_disagreement': float(np.max(disagreements)),
                'max_disagreement_threshold': float(thresholds[peak_idx]),
                'min_disagreement': float(np.min(disagreements)),
                'min_disagreement_threshold': float(thresholds[valley_idx]),
                'range': float(np.max(disagreements) - np.min(disagreements)),
                'inflection_points': [float(thresholds[i]) for i in inflection_indices if i < len(thresholds)],
                'max_curvature_threshold': float(thresholds[max_curvature_idx + 2]) if max_curvature_idx is not None and max_curvature_idx + 2 < len(thresholds) else None,
                'max_jump': float(max_jump),
                'max_jump_threshold': float(thresholds[jump_idx]) if jump_idx < len(thresholds) else None,
                'mean': float(np.mean(disagreements)),
                'std': float(np.std(disagreements))
            }

        return analysis

    def analyze_dimensional_scaling(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dimensional scaling effects."""
        analysis = {}

        dimensions = results['dimensions']
        data = results['data']

        for rule in ['plurality', 'borda', 'irv']:
            disagreements = []
            for d in dimensions:
                if str(d) in data and rule in data[str(d)]:
                    disagreements.append(data[str(d)][rule]['disagreement_rate'])

            if len(disagreements) < 2:
                continue

            disagreements = np.array(disagreements)
            dims = np.array(dimensions[:len(disagreements)])

            # Find peak
            peak_idx = np.argmax(disagreements)
            peak_dim = dims[peak_idx]
            peak_value = disagreements[peak_idx]

            # Fit power law for dimensions <= peak
            pre_peak_mask = dims <= peak_dim
            if np.sum(pre_peak_mask) >= 2:
                pre_peak_dims = dims[pre_peak_mask]
                pre_peak_disagreements = disagreements[pre_peak_mask]

                # Log-log fit
                log_dims = np.log(pre_peak_dims + 1e-10)
                log_disagreements = np.log(pre_peak_disagreements + 1e-10)

                # Linear fit in log space
                coeffs = np.polyfit(log_dims, log_disagreements, 1)
                alpha = coeffs[0]  # Power law exponent
                intercept = coeffs[1]

                # R-squared
                log_pred = np.polyval(coeffs, log_dims)
                ss_res = np.sum((log_disagreements - log_pred)**2)
                ss_tot = np.sum((log_disagreements - np.mean(log_disagreements))**2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            else:
                alpha = None
                r_squared = None

            # Post-peak slope
            post_peak_mask = dims > peak_dim
            if np.sum(post_peak_mask) >= 2:
                post_peak_dims = dims[post_peak_mask]
                post_peak_disagreements = disagreements[post_peak_mask]
                beta = np.polyfit(post_peak_dims, post_peak_disagreements, 1)[0]
            else:
                beta = None

            analysis[rule] = {
                'peak_dimension': int(peak_dim),
                'peak_disagreement': float(peak_value),
                'scaling_exponent': float(alpha) if alpha is not None else None,
                'r_squared': float(r_squared) if r_squared is not None else None,
                'post_peak_slope': float(beta) if beta is not None else None,
                'min_dimension': int(dims[0]),
                'max_dimension': int(dims[-1]),
                'min_disagreement': float(disagreements[0]),
                'max_disagreement': float(disagreements[-1])
            }

        return analysis

    def analyze_metric_pairs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metric pair interactions and asymmetry."""
        analysis = {
            'asymmetries': {},
            'hierarchy': {},
            'interaction_strengths': {}
        }

        pairs = results['pairs']

        for rule in ['plurality', 'borda', 'irv']:
            rule_asymmetries = {}
            rule_strengths = {}

            for pair_name, pair_data in pairs.items():
                if rule not in pair_data:
                    continue

                data = pair_data[rule]
                asymmetry = data.get('asymmetry', 0.0)
                disagreement_ab = data.get('disagreement_ab', 0.0)
                disagreement_ba = data.get('disagreement_ba', 0.0)

                # Average disagreement (strength)
                avg_disagreement = (disagreement_ab + disagreement_ba) / 2

                rule_asymmetries[pair_name] = asymmetry
                rule_strengths[pair_name] = avg_disagreement

            analysis['asymmetries'][rule] = rule_asymmetries
            analysis['interaction_strengths'][rule] = rule_strengths

            # Hierarchy (sorted by strength)
            sorted_pairs = sorted(rule_strengths.items(), key=lambda x: x[1], reverse=True)
            analysis['hierarchy'][rule] = [
                {'pair': pair, 'strength': strength}
                for pair, strength in sorted_pairs
            ]

        return analysis

    def analyze_condorcet_paradox(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Condorcet cycle and efficiency paradox."""
        analysis = {}

        # Look for threshold sweep or metric pairs results
        if 'data' in results:
            # Threshold sweep format
            data = results['data']
            thresholds = results.get('thresholds', [])

            for rule in ['plurality', 'borda', 'irv']:
                cycle_deltas = []
                efficiency_deltas = []

                for t in thresholds:
                    key = f"{t:.2f}"
                    if key in data and rule in data[key]:
                        cycle_deltas.append(data[key][rule].get('cycle_rate_delta', 0.0))
                        efficiency_deltas.append(data[key][rule].get('condorcet_efficiency_delta', 0.0))

                if cycle_deltas and efficiency_deltas:
                    analysis[rule] = {
                        'mean_cycle_delta': float(np.mean(cycle_deltas)),
                        'mean_efficiency_delta': float(np.mean(efficiency_deltas)),
                        'paradox_coefficient': float(np.mean(efficiency_deltas) / (np.mean(cycle_deltas) + 1e-10))
                    }

        return analysis

    def generate_full_analysis(self) -> Dict[str, Any]:
        """Generate complete analysis from all result files."""
        analysis = {
            'voter_scaling': {},
            'threshold_sweep': {},
            'dimensional_scaling': {},
            'metric_pairs': {},
            'condorcet_paradox': {}
        }

        # Find result files - prioritize L1-cosine (shows real effects) over L2-cosine (methodology issue)
        result_files = list(self.results_dir.glob("*.json"))

        # Sort to prioritize L1-cosine files
        result_files.sort(key=lambda x: ('l1_cosine' in x.name, x.name))

        for filepath in result_files:
            filename = filepath.name

            # Skip old verification files
            if 'verification_' in filename and '2025' in filename:
                continue

            try:
                results = self.load_results(filename)

                if 'voter_scaling' in filename and 'l1_cosine' in filename:
                    analysis['voter_scaling'] = self.analyze_voter_scaling(results)
                elif 'threshold_sweep' in filename and 'l1_cosine' in filename:
                    analysis['threshold_sweep'] = self.analyze_threshold_sweep(results)
                elif 'dimensional_scaling' in filename and 'l1_cosine' in filename:
                    analysis['dimensional_scaling'] = self.analyze_dimensional_scaling(results)
                elif 'metric_pairs' in filename and not analysis.get('metric_pairs'):
                    analysis['metric_pairs'] = self.analyze_metric_pairs(results)
                    analysis['condorcet_paradox'] = self.analyze_condorcet_paradox(results)
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                continue

        return analysis

    def save_analysis(self, analysis: Dict[str, Any], output_file: str = "analysis.json"):
        """Save analysis results."""
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {output_path}")


if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analysis = analyzer.generate_full_analysis()
    analyzer.save_analysis(analysis, "full_analysis.json")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(json.dumps(analysis, indent=2))
