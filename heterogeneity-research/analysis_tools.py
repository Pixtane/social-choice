"""
Analysis and interpretation tools for heterogeneity research.

Provides functions to:
- Load and parse test results
- Compute summary statistics
- Generate insights and interpretations
- Create comparison reports
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import pandas as pd


class HeterogeneityAnalyzer:
    """Analyzer for heterogeneity research results."""
    
    def __init__(self, results_dir: str = "heterogeneity-research/results"):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)
        self.loaded_results = {}
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.loaded_results[filename] = data
        return data
    
    def analyze_baseline_characterization(
        self,
        results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze baseline characterization results.
        
        Returns insights about homogeneous metric performance.
        """
        if results is None:
            results = self.loaded_results.get('baseline_characterization.json')
            if results is None:
                results = self.load_results('baseline_characterization.json')
        
        analysis = {
            'metric_rankings': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Rank metrics by VSE for each rule
        metrics = list(results.keys())
        rules = set()
        for metric_data in results.values():
            rules.update(metric_data.keys())
        
        for rule in rules:
            metric_vse = {}
            for metric in metrics:
                if rule in results[metric]:
                    metric_vse[metric] = results[metric][rule]['vse_mean']
            
            # Sort by VSE (descending)
            ranked = sorted(metric_vse.items(), key=lambda x: -x[1])
            analysis['metric_rankings'][rule] = ranked
        
        # Key findings
        # Find best overall metric
        overall_scores = defaultdict(float)
        for rule, rankings in analysis['metric_rankings'].items():
            for rank, (metric, vse) in enumerate(rankings):
                overall_scores[metric] += (len(rankings) - rank) * vse
        
        best_metric = max(overall_scores.items(), key=lambda x: x[1])[0]
        analysis['key_findings'].append(
            f"Best overall metric: {best_metric.upper()} "
            f"(weighted score: {overall_scores[best_metric]:.3f})"
        )
        
        # Find metric with lowest cycle rate
        cycle_rates = {}
        for metric in metrics:
            if 'schulze' in results[metric]:
                cycle_rates[metric] = results[metric]['schulze'].get('cycle_percentage', 0)
        
        if cycle_rates:
            lowest_cycle = min(cycle_rates.items(), key=lambda x: x[1])
            analysis['key_findings'].append(
                f"Lowest cycle rate: {lowest_cycle[0].upper()} "
                f"({lowest_cycle[1]:.2f}%)"
            )
        
        return analysis
    
    def analyze_threshold_sweep(
        self,
        results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze threshold sweep results.
        
        Identifies optimal thresholds and patterns.
        """
        if results is None:
            # Try to find threshold sweep file
            files = list(self.results_dir.glob("threshold_sweep_*.json"))
            if files:
                results = self.load_results(files[0].name)
            else:
                raise ValueError("No threshold sweep results found")
        
        analysis = {
            'optimal_thresholds': {},
            'threshold_effects': {},
            'key_findings': []
        }
        
        thresholds = np.array(results['thresholds'])
        rules = set()
        for threshold_data in results['results'].values():
            rules.update(threshold_data.keys())
        
        for rule in rules:
            disagreements = []
            vse_diffs = []
            
            for threshold in thresholds:
                key = f"{threshold:.1f}"
                if key in results['results'] and rule in results['results'][key]:
                    data = results['results'][key][rule]
                    disagreements.append(data.get('disagreement_rate', 0))
                    vse_diffs.append(data.get('vse_difference', 0))
            
            if disagreements:
                # Find threshold with maximum disagreement
                max_idx = np.argmax(disagreements)
                optimal_threshold = thresholds[max_idx]
                
                analysis['optimal_thresholds'][rule] = {
                    'threshold': float(optimal_threshold),
                    'max_disagreement': float(disagreements[max_idx]),
                    'vse_difference_at_max': float(vse_diffs[max_idx])
                }
                
                # Analyze trend
                if len(disagreements) > 2:
                    trend = 'increasing' if disagreements[-1] > disagreements[0] else 'decreasing'
                    analysis['threshold_effects'][rule] = trend
        
        # Key findings
        avg_optimal = np.mean([
            data['threshold'] 
            for data in analysis['optimal_thresholds'].values()
        ])
        analysis['key_findings'].append(
            f"Average optimal threshold: {avg_optimal:.2f}"
        )
        
        return analysis
    
    def compare_homogeneous_vs_heterogeneous(
        self,
        homo_results: Dict[str, Any],
        het_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare homogeneous and heterogeneous results.
        
        Args:
            homo_results: Results from homogeneous simulation
            het_results: Results from heterogeneous simulation
        """
        comparison = {
            'rule_disagreement': {},
            'vse_changes': {},
            'key_insights': []
        }
        
        # Extract rule names
        rules = set()
        if 'rule_results' in homo_results:
            rules.update(homo_results['rule_results'].keys())
        if 'rule_results' in het_results:
            rules.update(het_results['rule_results'].keys())
        
        for rule in rules:
            if rule in homo_results.get('rule_results', {}) and \
               rule in het_results.get('rule_results', {}):
                
                homo_vse = homo_results['rule_results'][rule]['aggregate_metrics']['vse_mean']
                het_vse = het_results['rule_results'][rule]['aggregate_metrics']['vse_mean']
                
                comparison['vse_changes'][rule] = {
                    'homogeneous': homo_vse,
                    'heterogeneous': het_vse,
                    'difference': het_vse - homo_vse,
                    'percent_change': 100 * (het_vse - homo_vse) / homo_vse if homo_vse != 0 else 0
                }
        
        # Key insights
        improvements = [
            (rule, data['difference'])
            for rule, data in comparison['vse_changes'].items()
            if data['difference'] > 0
        ]
        
        if improvements:
            best_improvement = max(improvements, key=lambda x: x[1])
            comparison['key_insights'].append(
                f"Best improvement: {best_improvement[0]} "
                f"(+{best_improvement[1]:.3f} VSE)"
            )
        
        return comparison
    
    def generate_summary_report(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            output_file: Optional path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HETEROGENEITY RESEARCH - SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Baseline analysis
        try:
            baseline = self.analyze_baseline_characterization()
            report_lines.append("## BASELINE CHARACTERIZATION")
            report_lines.append("")
            report_lines.append("### Metric Rankings by Rule:")
            for rule, rankings in baseline['metric_rankings'].items():
                report_lines.append(f"\n{rule.upper()}:")
                for rank, (metric, vse) in enumerate(rankings, 1):
                    report_lines.append(f"  {rank}. {metric.upper()}: VSE = {vse:.4f}")
            
            report_lines.append("\n### Key Findings:")
            for finding in baseline['key_findings']:
                report_lines.append(f"  - {finding}")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"Baseline analysis unavailable: {e}")
            report_lines.append("")
        
        # Threshold sweep analysis
        try:
            threshold = self.analyze_threshold_sweep()
            report_lines.append("## THRESHOLD SWEEP ANALYSIS")
            report_lines.append("")
            report_lines.append("### Optimal Thresholds by Rule:")
            for rule, data in threshold['optimal_thresholds'].items():
                report_lines.append(
                    f"  {rule.upper()}: threshold={data['threshold']:.2f}, "
                    f"disagreement={data['max_disagreement']:.1f}%"
                )
            
            report_lines.append("\n### Key Findings:")
            for finding in threshold['key_findings']:
                report_lines.append(f"  - {finding}")
            report_lines.append("")
        except Exception as e:
            report_lines.append(f"Threshold analysis unavailable: {e}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def identify_research_insights(
        self,
        results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate research insights from results.
        
        Returns list of insight strings.
        """
        insights = []
        
        # Check for high disagreement rates
        if 'threshold_sweep' in results:
            max_disagreement = 0
            for threshold_data in results['threshold_sweep']['results'].values():
                for rule_data in threshold_data.values():
                    if isinstance(rule_data, dict):
                        disagreement = rule_data.get('disagreement_rate', 0)
                        max_disagreement = max(max_disagreement, disagreement)
            
            if max_disagreement > 50:
                insights.append(
                    f"Heterogeneity creates significant effects: "
                    f"up to {max_disagreement:.1f}% rule disagreement"
                )
        
        # Check for VSE improvements
        if 'threshold_sweep' in results:
            best_vse_improvement = -float('inf')
            for threshold_data in results['threshold_sweep']['results'].values():
                for rule_data in threshold_data.values():
                    if isinstance(rule_data, dict):
                        vse_diff = rule_data.get('vse_difference', 0)
                        best_vse_improvement = max(best_vse_improvement, vse_diff)
            
            if best_vse_improvement > 0:
                insights.append(
                    f"Heterogeneity can improve outcomes: "
                    f"best VSE improvement of {best_vse_improvement:.4f}"
                )
        
        return insights


def main():
    """Example usage of analyzer."""
    analyzer = HeterogeneityAnalyzer()
    
    # Generate summary report
    report = analyzer.generate_summary_report(
        output_file="heterogeneity-research/summary_report.md"
    )
    print(report)


if __name__ == '__main__':
    main()

