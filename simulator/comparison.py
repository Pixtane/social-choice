"""
Comparison module for analyzing multiple experiments.

Provides functionality to:
- Load and compare multiple experiments
- Generate comparison tables in different modes
- Compute differential statistics
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime

from .storage import list_experiments, load_experiment, InputStorage, ResultStorage


@dataclass
class ExperimentSummary:
    """Summary of a single experiment for comparison."""
    
    # Identification
    experiment_id: str
    filename: str
    inputs_path: str
    results_path: Optional[str] = None
    
    # Basic parameters
    n_profiles: int = 0
    n_voters: int = 0
    n_candidates: int = 0
    
    # Configuration
    geometry_method: str = ""
    utility_function: str = ""
    distance_metric: str = ""
    manipulation_enabled: bool = False
    heterogeneous_distance_enabled: bool = False
    
    # Voting rules evaluated
    voting_rules: List[str] = field(default_factory=list)
    
    # Per-rule aggregate metrics
    rule_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timing
    compute_time: float = 0.0
    created_at: str = ""
    
    @classmethod
    def from_experiment(cls, inputs_path: str, results_path: Optional[str] = None) -> 'ExperimentSummary':
        """Load experiment and create summary."""
        # Load inputs
        data = InputStorage.load(inputs_path)
        metadata = data.get('metadata', {})
        config = data.get('config', {})
        
        # Try to get voting rules from config
        voting_rules = metadata.get('voting_rules', [])
        if not voting_rules and isinstance(config, dict):
            voting_rules = config.get('voting_rules', [])
        
        summary = cls(
            experiment_id=metadata.get('experiment_id', 'unknown'),
            filename=Path(inputs_path).stem,
            inputs_path=inputs_path,
            results_path=results_path,
            n_profiles=metadata.get('n_profiles', 0),
            n_voters=metadata.get('n_voters', 0),
            n_candidates=metadata.get('n_candidates', 0),
            geometry_method=metadata.get('geometry_method', '') or config.get('geometry', {}).get('method', ''),
            utility_function=metadata.get('utility_function', '') or config.get('utility', {}).get('function', ''),
            distance_metric=metadata.get('utility_distance_metric', '') or config.get('utility', {}).get('distance_metric', ''),
            manipulation_enabled=metadata.get('manipulation_enabled', False),
            heterogeneous_distance_enabled=metadata.get('heterogeneous_distance_enabled', False),
            voting_rules=voting_rules,
            created_at=metadata.get('created_at', ''),
        )
        
        # Load results if available
        if results_path:
            try:
                results_data = ResultStorage.load_as_dict(results_path)
                summary._parse_results(results_data)
            except Exception:
                pass
        
        # Also try to find results file if not provided
        if not results_path:
            base = Path(inputs_path).stem
            results_dir = Path(inputs_path).parent.parent / 'results'
            
            for ext in ['.parquet', '.csv', '.json']:
                candidate = results_dir / f"{base}{ext}"
                if candidate.exists():
                    try:
                        results_data = ResultStorage.load_as_dict(str(candidate))
                        summary.results_path = str(candidate)
                        summary._parse_results(results_data)
                        break
                    except Exception:
                        pass
        
        return summary
    
    def _parse_results(self, results_data: Dict[str, Any]) -> None:
        """Parse results data to extract per-rule metrics."""
        # First, infer voting rules from column names if not already set
        if not self.voting_rules:
            # Common rule names to look for
            known_rules = ['plurality', 'borda', 'irv', 'copeland', 'minimax', 
                          'approval', 'star', 'score', 'kemeny', 'ranked_pairs',
                          'schulze', 'black', 'bucklin', 'coombs', 'random']
            detected_rules = set()
            
            for key in results_data.keys():
                for rule in known_rules:
                    if key.startswith(f'{rule}_'):
                        detected_rules.add(rule)
            
            self.voting_rules = list(detected_rules)
        
        # Parse per-rule metrics
        for key, values in results_data.items():
            if key.startswith('agg_') or key.startswith('config_'):
                continue
            
            # Look for per-rule columns like 'plurality_vse', 'borda_winner'
            for rule in self.voting_rules:
                if key.startswith(f'{rule}_'):
                    metric_name = key[len(rule)+1:]
                    
                    if rule not in self.rule_metrics:
                        self.rule_metrics[rule] = {}
                    
                    # Compute aggregate from list
                    if isinstance(values, list) and len(values) > 0:
                        try:
                            # Filter numeric values
                            float_values = []
                            for v in values:
                                if v is not None and v != '':
                                    try:
                                        float_values.append(float(v))
                                    except (ValueError, TypeError):
                                        pass
                            
                            if float_values:
                                self.rule_metrics[rule][f'{metric_name}_mean'] = np.mean(float_values)
                                self.rule_metrics[rule][f'{metric_name}_std'] = np.std(float_values)
                        except Exception:
                            pass
                    break  # Found the rule, no need to check others
        
        # Also try to get direct aggregate columns
        for key, values in results_data.items():
            if key.startswith('agg_'):
                metric_name = key[4:]  # Remove 'agg_' prefix
                # This is a global aggregate, not per-rule
                if isinstance(values, list) and len(values) > 0:
                    try:
                        # Just take first value (should all be same for aggregate)
                        val = float(values[0])
                        self.rule_metrics.setdefault('_global', {})[metric_name] = val
                    except (ValueError, TypeError):
                        pass
        
        # Get compute time
        if 'compute_time_seconds' in results_data:
            times = results_data['compute_time_seconds']
            if isinstance(times, list) and len(times) > 0:
                try:
                    self.compute_time = float(times[0])
                except (ValueError, TypeError):
                    pass


@dataclass
class ComparisonResult:
    """Result of comparing multiple experiments."""
    
    experiments: List[ExperimentSummary]
    mode: str
    
    # Comparison tables (different views)
    by_experiment: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_rule: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_metric: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Differential analysis
    differences: Dict[str, Any] = field(default_factory=dict)


class ExperimentComparator:
    """
    Compare multiple experiments.
    
    Supports multiple comparison modes:
    - by_experiment: Each row is an experiment, columns are metrics
    - by_rule: Each row is a voting rule, columns are experiments
    - by_metric: Each row is a metric, columns are experiments/rules
    - differential: Show differences between experiments
    """
    
    def __init__(self, experiments: List[ExperimentSummary]):
        """
        Initialize comparator.
        
        Args:
            experiments: List of experiment summaries to compare
        """
        self.experiments = experiments
    
    @classmethod
    def from_paths(cls, paths: List[str]) -> 'ExperimentComparator':
        """
        Create comparator from file paths.
        
        Args:
            paths: List of input file paths
            
        Returns:
            ExperimentComparator instance
        """
        summaries = []
        
        for path in paths:
            # Find matching results file
            results_path = None
            inputs_path = path
            
            if path.endswith('.npz'):
                # Look for results
                base = Path(path).stem
                results_dir = Path(path).parent.parent / 'results'
                
                for ext in ['.parquet', '.csv', '.json']:
                    candidate = results_dir / f"{base}{ext}"
                    if candidate.exists():
                        results_path = str(candidate)
                        break
            
            summary = ExperimentSummary.from_experiment(inputs_path, results_path)
            summaries.append(summary)
        
        return cls(summaries)
    
    @classmethod
    def from_experiment_ids(
        cls, 
        experiment_ids: List[str],
        base_dir: Optional[str] = None
    ) -> 'ExperimentComparator':
        """
        Create comparator from experiment IDs.
        
        Args:
            experiment_ids: List of experiment IDs (partial match supported)
            base_dir: Base directory for experiments
            
        Returns:
            ExperimentComparator instance
        """
        all_experiments = list_experiments(base_dir)
        
        summaries = []
        for exp_id in experiment_ids:
            # Find matching experiment
            for exp in all_experiments:
                if exp_id in exp.get('experiment_id', '') or exp_id in exp.get('filename', ''):
                    summary = ExperimentSummary.from_experiment(
                        exp['inputs_path'],
                        exp.get('results_path')
                    )
                    summaries.append(summary)
                    break
        
        return cls(summaries)
    
    def compare(
        self,
        mode: Literal['by_experiment', 'by_rule', 'by_metric', 'differential'] = 'by_experiment',
        metrics: Optional[List[str]] = None,
        rules: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare experiments.
        
        Args:
            mode: Comparison mode
            metrics: Specific metrics to compare (default: all)
            rules: Specific rules to compare (default: all)
            
        Returns:
            ComparisonResult with comparison tables
        """
        result = ComparisonResult(
            experiments=self.experiments,
            mode=mode
        )
        
        # Get all rules across experiments
        all_rules = set()
        for exp in self.experiments:
            all_rules.update(exp.voting_rules)
        
        if rules:
            all_rules = all_rules.intersection(rules)
        
        all_rules = sorted(all_rules)
        
        # Get all metrics
        all_metrics = set()
        for exp in self.experiments:
            for rule_metrics in exp.rule_metrics.values():
                all_metrics.update(rule_metrics.keys())
        
        if metrics:
            all_metrics = all_metrics.intersection(metrics)
        
        all_metrics = sorted(all_metrics)
        
        # Build comparison tables
        if mode == 'by_experiment':
            result.by_experiment = self._compare_by_experiment(all_rules, all_metrics)
        elif mode == 'by_rule':
            result.by_rule = self._compare_by_rule(all_rules, all_metrics)
        elif mode == 'by_metric':
            result.by_metric = self._compare_by_metric(all_rules, all_metrics)
        elif mode == 'differential':
            result.differences = self._compute_differential(all_rules, all_metrics)
        
        return result
    
    def _compare_by_experiment(
        self,
        rules: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare with experiments as rows."""
        table = {}
        
        for exp in self.experiments:
            row = {
                'id': exp.experiment_id[:8],
                'profiles': exp.n_profiles,
                'voters': exp.n_voters,
                'candidates': exp.n_candidates,
                'geometry': exp.geometry_method or '-',
                'rules': ', '.join(exp.voting_rules[:3]) + ('...' if len(exp.voting_rules) > 3 else ''),
            }
            
            # Add per-rule metrics (focus on VSE which is most important)
            for rule in rules:
                if rule in exp.rule_metrics:
                    rule_data = exp.rule_metrics[rule]
                    # Add VSE if available
                    if 'vse_mean' in rule_data:
                        row[f'{rule}_vse'] = rule_data['vse_mean']
                    # Add winner rank if available
                    if 'winner_rank_mean' in rule_data:
                        row[f'{rule}_rank'] = rule_data['winner_rank_mean']
            
            table[exp.filename] = row
        
        return table
    
    def _compare_by_rule(
        self,
        rules: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare with rules as rows."""
        table = {}
        
        for rule in rules:
            row = {'rule': rule}
            
            for exp in self.experiments:
                if rule in exp.rule_metrics:
                    for metric_name, value in exp.rule_metrics[rule].items():
                        if metric_name.endswith('_mean'):
                            clean_name = metric_name[:-5]
                            row[f'{exp.experiment_id[:8]}_{clean_name}'] = value
            
            table[rule] = row
        
        return table
    
    def _compare_by_metric(
        self,
        rules: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare with metrics as rows."""
        table = {}
        
        # Get unique metric base names
        metric_bases = set()
        for m in metrics:
            if m.endswith('_mean'):
                metric_bases.add(m[:-5])
            elif m.endswith('_std'):
                metric_bases.add(m[:-4])
        
        for metric_base in sorted(metric_bases):
            row = {'metric': metric_base}
            
            for exp in self.experiments:
                for rule in rules:
                    if rule in exp.rule_metrics:
                        rule_metrics = exp.rule_metrics[rule]
                        mean_key = f'{metric_base}_mean'
                        if mean_key in rule_metrics:
                            row[f'{exp.experiment_id[:8]}_{rule}'] = rule_metrics[mean_key]
            
            table[metric_base] = row
        
        return table
    
    def _compute_differential(
        self,
        rules: List[str],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute differences between experiments."""
        if len(self.experiments) < 2:
            return {'error': 'Need at least 2 experiments for differential'}
        
        # Use first experiment as baseline
        baseline = self.experiments[0]
        differences = {
            'baseline': baseline.experiment_id,
            'comparisons': {}
        }
        
        for exp in self.experiments[1:]:
            exp_diff = {}
            
            for rule in rules:
                if rule in baseline.rule_metrics and rule in exp.rule_metrics:
                    rule_diff = {}
                    
                    for metric_name in baseline.rule_metrics[rule]:
                        if metric_name in exp.rule_metrics[rule]:
                            baseline_val = baseline.rule_metrics[rule][metric_name]
                            exp_val = exp.rule_metrics[rule][metric_name]
                            
                            diff = exp_val - baseline_val
                            pct_diff = (diff / baseline_val * 100) if baseline_val != 0 else 0
                            
                            rule_diff[metric_name] = {
                                'baseline': baseline_val,
                                'comparison': exp_val,
                                'difference': diff,
                                'percent_change': pct_diff
                            }
                    
                    exp_diff[rule] = rule_diff
            
            differences['comparisons'][exp.experiment_id] = exp_diff
        
        return differences
    
    def format_table(
        self,
        result: ComparisonResult,
        format: Literal['text', 'markdown', 'csv'] = 'text'
    ) -> str:
        """
        Format comparison result as table string.
        
        Args:
            result: ComparisonResult from compare()
            format: Output format
            
        Returns:
            Formatted table string
        """
        if result.mode == 'by_experiment':
            data = result.by_experiment
        elif result.mode == 'by_rule':
            data = result.by_rule
        elif result.mode == 'by_metric':
            data = result.by_metric
        else:
            return self._format_differential(result.differences, format)
        
        if not data:
            return "No data to display"
        
        # Get all columns
        all_cols = set()
        for row in data.values():
            all_cols.update(row.keys())
        
        # Order columns nicely
        cols = sorted(all_cols)
        
        # Format based on type
        if format == 'csv':
            return self._format_csv(data, cols)
        elif format == 'markdown':
            return self._format_markdown(data, cols)
        else:
            return self._format_text(data, cols)
    
    def _format_text(self, data: Dict, cols: List[str]) -> str:
        """Format as plain text table."""
        # Calculate column widths
        widths = {col: len(col) for col in cols}
        widths['_key'] = max(len(k) for k in data.keys())
        
        for key, row in data.items():
            for col in cols:
                val = row.get(col, '')
                val_str = self._format_value(val)
                widths[col] = max(widths[col], len(val_str))
        
        # Build header
        lines = []
        header = "| " + " | ".join(
            [f"{'Experiment':<{widths['_key']}}"] +
            [f"{col:<{widths[col]}}" for col in cols]
        ) + " |"
        
        sep = "|-" + "-|-".join(
            ["-" * widths['_key']] +
            ["-" * widths[col] for col in cols]
        ) + "-|"
        
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        
        # Build rows
        for key, row in data.items():
            row_str = "| " + " | ".join(
                [f"{key:<{widths['_key']}}"] +
                [f"{self._format_value(row.get(col, '')):<{widths[col]}}" for col in cols]
            ) + " |"
            lines.append(row_str)
        
        lines.append(sep)
        
        return "\n".join(lines)
    
    def _format_markdown(self, data: Dict, cols: List[str]) -> str:
        """Format as Markdown table."""
        lines = []
        
        # Header
        header = "| Experiment | " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["-" * 12 for _ in range(len(cols) + 1)]) + "|"
        
        lines.append(header)
        lines.append(sep)
        
        # Rows
        for key, row in data.items():
            row_str = f"| {key} | " + " | ".join(
                self._format_value(row.get(col, '')) for col in cols
            ) + " |"
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def _format_csv(self, data: Dict, cols: List[str]) -> str:
        """Format as CSV."""
        lines = []
        
        # Header
        lines.append("experiment," + ",".join(cols))
        
        # Rows
        for key, row in data.items():
            values = [self._format_value(row.get(col, '')) for col in cols]
            lines.append(f"{key}," + ",".join(values))
        
        return "\n".join(lines)
    
    def _format_value(self, val) -> str:
        """Format a single value."""
        if val is None or val == '':
            return '-'
        elif isinstance(val, float):
            return f"{val:.4f}"
        elif isinstance(val, bool):
            return "Yes" if val else "No"
        else:
            return str(val)
    
    def _format_differential(self, differences: Dict, format: str) -> str:
        """Format differential comparison."""
        if 'error' in differences:
            return differences['error']
        
        lines = []
        lines.append(f"Baseline: {differences['baseline']}")
        lines.append("")
        
        for exp_id, exp_diff in differences.get('comparisons', {}).items():
            lines.append(f"Comparison: {exp_id}")
            lines.append("-" * 60)
            
            for rule, metrics in exp_diff.items():
                lines.append(f"  {rule}:")
                for metric_name, values in metrics.items():
                    if metric_name.endswith('_mean'):
                        clean_name = metric_name[:-5]
                        diff = values['difference']
                        pct = values['percent_change']
                        sign = "+" if diff > 0 else ""
                        lines.append(f"    {clean_name}: {sign}{diff:.4f} ({sign}{pct:.1f}%)")
            
            lines.append("")
        
        return "\n".join(lines)


def compare_experiments(
    experiment_paths: Optional[List[str]] = None,
    experiment_ids: Optional[List[str]] = None,
    mode: str = 'by_experiment',
    output_format: str = 'text',
    base_dir: Optional[str] = None
) -> str:
    """
    Convenience function to compare experiments.
    
    Args:
        experiment_paths: List of input file paths
        experiment_ids: List of experiment IDs (alternative to paths)
        mode: Comparison mode
        output_format: Output format (text, markdown, csv)
        base_dir: Base directory for experiments
        
    Returns:
        Formatted comparison string
    """
    if experiment_paths:
        comparator = ExperimentComparator.from_paths(experiment_paths)
    elif experiment_ids:
        comparator = ExperimentComparator.from_experiment_ids(experiment_ids, base_dir)
    else:
        raise ValueError("Must provide either experiment_paths or experiment_ids")
    
    result = comparator.compare(mode=mode)
    return comparator.format_table(result, format=output_format)


def get_comparison_modes() -> Dict[str, str]:
    """Get available comparison modes."""
    return {
        'by_experiment': 'Rows=experiments, Cols=metrics',
        'by_rule': 'Rows=voting rules, Cols=experiments',
        'by_metric': 'Rows=metrics, Cols=experiment/rule combinations',
        'differential': 'Show differences from first experiment as baseline',
    }


# Module exports
__all__ = [
    'ExperimentSummary',
    'ComparisonResult',
    'ExperimentComparator',
    'compare_experiments',
    'get_comparison_modes',
]

