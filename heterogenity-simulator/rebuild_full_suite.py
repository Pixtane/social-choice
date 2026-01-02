"""
Rebuild full_research_suite.json from individual result files.
"""

import json
from pathlib import Path
from fix_serialization import make_serializable


def main():
    results_dir = Path("heterogenity-simulator/results")

    # Load individual result files
    experiments = {}

    # Voter scaling
    voter_file = results_dir / "voter_scaling_l2_cosine_d2.json"
    if voter_file.exists():
        with open(voter_file, 'r', encoding='utf-8') as f:
            experiments['voter_scaling'] = json.load(f)
        print("Loaded voter_scaling")

    # Threshold sweep
    threshold_file = results_dir / "threshold_sweep_l2_cosine_d2_v100.json"
    if threshold_file.exists():
        with open(threshold_file, 'r', encoding='utf-8') as f:
            experiments['threshold_sweep'] = json.load(f)
        print("Loaded threshold_sweep")

    # Dimensional scaling
    dim_file = results_dir / "dimensional_scaling_l2_cosine_v100.json"
    if dim_file.exists():
        with open(dim_file, 'r', encoding='utf-8') as f:
            experiments['dimensional_scaling'] = json.load(f)
        print("Loaded dimensional_scaling")

    # Metric pairs (100 voters)
    pairs_100_file = results_dir / "metric_pairs_d2_v100.json"
    if pairs_100_file.exists():
        with open(pairs_100_file, 'r', encoding='utf-8') as f:
            experiments['metric_pairs'] = json.load(f)
        print("Loaded metric_pairs (100 voters)")

    # Metric pairs (500 voters - verification)
    pairs_500_file = results_dir / "metric_pairs_d2_v500.json"
    if pairs_500_file.exists():
        with open(pairs_500_file, 'r', encoding='utf-8') as f:
            experiments['verification_500_voters'] = json.load(f)
        print("Loaded verification_500_voters")

    # Create full suite structure
    full_suite = {
        'config': {
            'base_n_profiles': 200,
            'base_n_voters': 100,
            'base_n_candidates': 5,
            'voter_scaling_range': [10, 25, 50, 100, 200, 300, 400, 500],
            'verification_n_voters': 500,
            'verification_n_profiles': 200,
            'dimensions': [1, 2, 3, 4, 5, 7, 10],
            'thresholds': [round(0.05 + i * 0.05, 2) for i in range(19)],
            'metrics': ['l1', 'l2', 'cosine', 'chebyshev'],
            'voting_rules': ['plurality', 'borda', 'irv']
        },
        'timestamp': '2025-01-01T00:00:00',
        'experiments': experiments
    }

    # Make serializable
    full_suite = make_serializable(full_suite)

    # Save
    output_file = results_dir / "full_research_suite.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_suite, f, indent=2, ensure_ascii=False)

    print(f"\nRebuilt full_research_suite.json with {len(experiments)} experiments")


if __name__ == "__main__":
    main()
