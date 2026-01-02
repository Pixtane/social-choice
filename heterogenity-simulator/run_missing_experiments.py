"""
Run missing experiments with corrected methodology.

The issue: When center_metric = L2 and homogeneous baseline = L2, 
center voters use L2 in both cases, so disagreement is 0.

Solution: Re-run experiments comparing to a different baseline,
or use metric pairs that show real effects.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from research_suite import ResearchConfig, HeterogeneityResearcher

def main():
    print("=" * 80)
    print("RUNNING MISSING EXPERIMENTS WITH CORRECTED METHODOLOGY")
    print("=" * 80)
    print("\nIssue: L2 center + cosine extreme vs L2 homogeneous shows 0 disagreement")
    print("because center voters use L2 in both cases.")
    print("\nSolution: Re-run with L1-cosine pair (both different from each other)")
    print("=" * 80)
    
    config = ResearchConfig(
        base_n_profiles=200,
        base_n_voters=100,
        voter_scaling_range=[10, 25, 50, 100, 200, 300, 400, 500],
        verification_n_voters=500,
        verification_n_profiles=200
    )
    
    researcher = HeterogeneityResearcher(config)
    
    # Re-run key experiments with L1-cosine (both metrics different)
    print("\n" + "=" * 80)
    print("RE-RUNNING: VOTER SCALING (L1-COSINE)")
    print("=" * 80)
    voter_scaling = researcher.experiment_voter_scaling(
        center_metric='l1',
        extreme_metric='cosine',
        threshold=0.5,
        dimension=2
    )
    
    print("\n" + "=" * 80)
    print("RE-RUNNING: THRESHOLD SWEEP (L1-COSINE)")
    print("=" * 80)
    threshold_sweep = researcher.experiment_threshold_sweep(
        center_metric='l1',
        extreme_metric='cosine',
        dimension=2,
        n_voters=100
    )
    
    print("\n" + "=" * 80)
    print("RE-RUNNING: DIMENSIONAL SCALING (L1-COSINE)")
    print("=" * 80)
    dimensional_scaling = researcher.experiment_dimensional_scaling(
        center_metric='l1',
        extreme_metric='cosine',
        threshold=0.5,
        n_voters=100
    )
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nThese experiments use L1-cosine pair where both metrics differ,")
    print("so we should see real heterogeneity effects.")

if __name__ == "__main__":
    main()




