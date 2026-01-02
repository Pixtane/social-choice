"""
Quick test to verify the research pipeline works.

Runs a minimal experiment to check everything is functioning.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from research_suite import ResearchConfig, HeterogeneityResearcher

def main():
    print("=" * 80)
    print("QUICK TEST: Verifying Research Pipeline")
    print("=" * 80)
    
    # Minimal config for testing
    config = ResearchConfig(
        base_n_profiles=10,  # Very small for testing
        base_n_voters=25,
        voter_scaling_range=[25, 50],  # Just 2 points
        verification_n_voters=50,
        verification_n_profiles=10,
        thresholds=[0.3, 0.5, 0.7],  # Just 3 thresholds
        dimensions=[2],  # Just 2D
        metrics=['l2', 'cosine'],  # Just 2 metrics
        voting_rules=['plurality']  # Just one rule
    )
    
    researcher = HeterogeneityResearcher(config)
    
    print("\nRunning minimal voter scaling test...")
    result = researcher.experiment_voter_scaling(
        center_metric='l2',
        extreme_metric='cosine',
        threshold=0.5,
        dimension=2
    )
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nIf you see this, the pipeline is working!")
    print("You can now run the full research suite with:")
    print("  python run_research.py")
    print("\nOr run research_suite.py directly for the full automated suite.")

if __name__ == "__main__":
    main()





