"""
Run research experiments in stages with progress saving.

This allows running experiments incrementally and resuming if interrupted.
"""

import json
from pathlib import Path
from research_suite import ResearchConfig, HeterogeneityResearcher
import sys

def main():
    config = ResearchConfig(
        base_n_profiles=200,
        base_n_voters=100,
        voter_scaling_range=[10, 25, 50, 100, 200, 300, 400, 500],
        verification_n_voters=500,
        verification_n_profiles=200
    )
    
    researcher = HeterogeneityResearcher(config)
    
    print("=" * 80)
    print("HETEROGENEITY RESEARCH SUITE")
    print("=" * 80)
    print("\nThis will run comprehensive experiments. Each phase may take 30-60 minutes.")
    print("Results are saved incrementally, so you can stop and resume.")
    print("\nPhases:")
    print("1. Voter scaling (8 voter counts × 200 profiles × 2 configs = ~3200 simulations)")
    print("2. Threshold sweep (19 thresholds × 200 profiles × 2 configs = ~7600 simulations)")
    print("3. Dimensional scaling (7 dimensions × 200 profiles × 2 configs = ~2800 simulations)")
    print("4. Metric pairs (12 pairs × 200 profiles × 3 configs = ~7200 simulations)")
    print("5. Final verification (12 pairs × 200 profiles × 3 configs = ~7200 simulations)")
    print("\nTotal: ~28,000 simulations")
    print("\nEstimated time: 4-8 hours depending on hardware")
    print()
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        return
    
    # Run full suite
    results = researcher.run_full_research_suite()
    
    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python analyze_results.py")
    print("2. Run: python generate_findings.py")
    print("3. Review: FINDINGS-2.md and METHODOLOGY.md")

if __name__ == "__main__":
    main()




