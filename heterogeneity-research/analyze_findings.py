"""
Deep analysis of research results to identify novel phenomena.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Simple sigmoid fit without scipy
def fit_sigmoid(x, y):
    """Simple sigmoid fitting using least squares."""
    try:
        # Normalize x to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        # Initial guess: a=max, b=10, c=0.5, d=min
        a_init = np.max(y)
        d_init = np.min(y)
        c_init = 0.5
        b_init = 10.0
        
        # Simple grid search for b and c
        best_error = float('inf')
        best_params = None
        
        for b in [5, 10, 15, 20]:
            for c in np.linspace(0.2, 0.8, 10):
                try:
                    sigmoid = a_init / (1 + np.exp(-b * (x_norm - c))) + d_init
                    error = np.sum((y - sigmoid)**2)
                    if error < best_error:
                        best_error = error
                        best_params = (a_init, b, c, d_init)
                except:
                    continue
        
        if best_params:
            return True, best_params
        else:
            return False, None
    except:
        return False, None


def analyze_threshold_nonlinearity(data_file: str) -> Dict:
    """Analyze threshold data for non-linear patterns and phase transitions."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    thresholds = np.array(data['thresholds'])
    analysis = {}
    
    for rule in ['plurality', 'borda', 'irv']:
        disagreements = []
        vse_diffs = []
        
        for t in thresholds:
            key = f"{t:.2f}"
            if key in data['data'] and rule in data['data'][key]:
                disagreements.append(data['data'][key][rule]['disagreement_rate'])
                vse_diffs.append(data['data'][key][rule]['vse_difference'])
        
        disagreements = np.array(disagreements)
        vse_diffs = np.array(vse_diffs)
        
        # Find critical points (local maxima/minima)
        d1 = np.diff(disagreements)
        d2 = np.diff(d1)
        
        # Inflection points (where second derivative changes sign)
        inflection_indices = []
        for i in range(1, len(d2)):
            if d2[i-1] * d2[i] < 0:
                inflection_indices.append(i+1)
        
        # Find regions of high curvature
        curvature = np.abs(d2)
        max_curvature_idx = np.argmax(curvature) if len(curvature) > 0 else None
        
        # Test for phase transition (sudden jump)
        max_jump = np.max(np.abs(np.diff(disagreements)))
        jump_idx = np.argmax(np.abs(np.diff(disagreements)))
        
        # Fit sigmoid to see if there's a smooth transition
        sigmoid_fit, sigmoid_params = fit_sigmoid(thresholds, disagreements)
        if sigmoid_fit:
            sigmoid_center = sigmoid_params[2]  # Center of sigmoid
            sigmoid_slope = sigmoid_params[1]   # Steepness
        else:
            sigmoid_center = None
            sigmoid_slope = None
        
        analysis[rule] = {
            'max_disagreement': float(np.max(disagreements)),
            'max_disagreement_threshold': float(thresholds[np.argmax(disagreements)]),
            'min_disagreement': float(np.min(disagreements)),
            'min_disagreement_threshold': float(thresholds[np.argmax(disagreements)]),
            'inflection_points': [float(thresholds[i]) for i in inflection_indices],
            'max_curvature_threshold': float(thresholds[max_curvature_idx + 2]) if max_curvature_idx is not None else None,
            'max_jump': float(max_jump),
            'jump_threshold': float(thresholds[jump_idx]),
            'sigmoid_fit': sigmoid_fit,
            'sigmoid_center': float(sigmoid_center) if sigmoid_center is not None else None,
            'sigmoid_slope': float(sigmoid_slope) if sigmoid_slope is not None else None,
            'variance': float(np.var(disagreements)),
            'range': float(np.max(disagreements) - np.min(disagreements))
        }
    
    return analysis


def analyze_metric_asymmetry(data_file: str) -> Dict:
    """Analyze asymmetry in metric pair interactions."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    matrix = data['interaction_matrix']
    asymmetry_analysis = {}
    
    for rule in ['plurality', 'borda', 'irv']:
        asymmetries = []
        
        metrics = ['l1', 'l2', 'cosine', 'chebyshev']
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i == j:
                    continue
                
                pair1 = f"{m1}_{m2}"
                pair2 = f"{m2}_{m1}"
                
                if pair1 in matrix and pair2 in matrix:
                    if rule in matrix[pair1] and rule in matrix[pair2]:
                        d1 = matrix[pair1][rule]['disagreement_rate']
                        d2 = matrix[pair2][rule]['disagreement_rate']
                        
                        asymmetry = abs(d1 - d2)
                        relative_asymmetry = asymmetry / max(d1, d2) if max(d1, d2) > 0 else 0
                        
                        asymmetries.append({
                            'pair1': pair1,
                            'pair2': pair2,
                            'disagreement1': d1,
                            'disagreement2': d2,
                            'absolute_asymmetry': asymmetry,
                            'relative_asymmetry': relative_asymmetry
                        })
        
        if asymmetries:
            asymmetries.sort(key=lambda x: -x['absolute_asymmetry'])
            asymmetry_analysis[rule] = {
                'max_asymmetry': asymmetries[0],
                'mean_asymmetry': float(np.mean([a['absolute_asymmetry'] for a in asymmetries])),
                'std_asymmetry': float(np.std([a['absolute_asymmetry'] for a in asymmetries])),
                'top_asymmetries': asymmetries[:5]
            }
    
    return asymmetry_analysis


def analyze_dimensional_scaling(data_file: str) -> Dict:
    """Analyze how effects scale with dimensionality."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    dimensions = np.array(data['dimensions'])
    analysis = {}
    
    for rule in ['plurality', 'borda', 'irv']:
        disagreements = []
        
        for dim in dimensions:
            if str(dim) in data['data'] and rule in data['data'][str(dim)]:
                disagreements.append(data['data'][str(dim)][rule]['disagreement_rate'])
        
        disagreements = np.array(disagreements)
        
        # Find peak
        peak_idx = np.argmax(disagreements)
        peak_dim = int(dimensions[peak_idx])
        
        # Test power law: D ~ d^alpha for d <= peak
        log_dims = np.log(dimensions[:peak_idx+1] + 1)
        log_disagreements = np.log(disagreements[:peak_idx+1] + 1)
        
        if len(log_dims) > 1:
            coeffs = np.polyfit(log_dims, log_disagreements, 1)
            alpha = coeffs[0]
            r_squared = np.corrcoef(log_dims, log_disagreements)[0,1]**2
        else:
            alpha = 0.0
            r_squared = 0.0
        
        # Test plateau after peak
        if peak_idx < len(disagreements) - 1:
            post_peak = disagreements[peak_idx+1:]
            plateau_slope = np.polyfit(range(len(post_peak)), post_peak, 1)[0] if len(post_peak) > 1 else 0.0
        else:
            plateau_slope = 0.0
        
        analysis[rule] = {
            'peak_dimension': peak_dim,
            'peak_disagreement': float(disagreements[peak_idx]),
            'scaling_exponent': float(alpha),
            'scaling_r_squared': float(r_squared),
            'plateau_slope': float(plateau_slope),
            'initial_disagreement': float(disagreements[0]),
            'final_disagreement': float(disagreements[-1]),
            'total_change': float(disagreements[-1] - disagreements[0])
        }
    
    return analysis


def analyze_preference_structure(data_file: str) -> Dict:
    """Analyze changes in preference structure."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    analysis = {}
    
    for rule in ['plurality', 'borda', 'irv']:
        cycle_changes = []
        condorcet_changes = []
        
        for threshold in data['thresholds']:
            key = f"{threshold}"
            if key in data['data'] and rule in data['data'][key]:
                cycle_changes.append(data['data'][key][rule]['cycle_rate_change'])
                condorcet_changes.append(data['data'][key][rule]['condorcet_efficiency_change'])
        
        analysis[rule] = {
            'mean_cycle_change': float(np.mean(cycle_changes)),
            'std_cycle_change': float(np.std(cycle_changes)),
            'mean_condorcet_change': float(np.mean(condorcet_changes)),
            'std_condorcet_change': float(np.std(condorcet_changes)),
            'max_cycle_increase': float(np.max(cycle_changes)),
            'max_condorcet_change': float(np.max(np.abs(condorcet_changes)))
        }
    
    return analysis


if __name__ == '__main__':
    results_dir = Path("heterogeneity-research/results")
    
    print("Analyzing threshold non-linearity...")
    threshold_analysis = analyze_threshold_nonlinearity(
        results_dir / "threshold_nonlinearity.json"
    )
    
    print("Analyzing metric asymmetry...")
    asymmetry_analysis = analyze_metric_asymmetry(
        results_dir / "metric_interaction_matrix.json"
    )
    
    print("Analyzing dimensional scaling...")
    scaling_analysis = analyze_dimensional_scaling(
        results_dir / "dimensional_scaling.json"
    )
    
    print("Analyzing preference structure...")
    preference_analysis = analyze_preference_structure(
        results_dir / "preference_structure_changes.json"
    )
    
    # Save comprehensive analysis
    comprehensive = {
        'threshold_nonlinearity': threshold_analysis,
        'metric_asymmetry': asymmetry_analysis,
        'dimensional_scaling': scaling_analysis,
        'preference_structure': preference_analysis
    }
    
    with open(results_dir / "comprehensive_analysis.json", 'w') as f:
        json.dump(comprehensive, f, indent=2)
    
    print(f"\nComprehensive analysis saved to: {results_dir / 'comprehensive_analysis.json'}")

