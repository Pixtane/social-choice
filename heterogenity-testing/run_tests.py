"""
Main script to run all heterogeneity tests and generate analysis.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from test_runner import HeterogeneityTestRunner
import json


def main():
    """Run all tests and save results."""
    print("Starting heterogeneity testing suite...")
    print("This may take a while (15 tests × 100 runs each)\n")
    
    runner = HeterogeneityTestRunner(n_runs=100)
    results = runner.run_all_tests()
    
    # Save results to JSON
    results_dir = os.path.dirname(__file__)
    results_file = os.path.join(results_dir, 'results.json')
    
    # Convert to serializable format
    serializable_results = {}
    for key, result in results.items():
        serializable_results[key] = {
            'test_name': result.test_name,
            'metrics': result.metrics,
            'compute_time': result.compute_time,
            'config_summary': result.config_summary
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"All tests completed!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")
    
    # Generate analysis
    print("\nGenerating analysis...")
    from generate_analysis import generate_analysis
    generate_analysis(results_file)
    
    print("\nAnalysis complete! Check HETEROGENITY_ANALYSIS.md")


if __name__ == "__main__":
    main()







