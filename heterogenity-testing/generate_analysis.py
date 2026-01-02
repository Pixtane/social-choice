"""
Generate analysis report from test results.
"""

import json
import os
from typing import Dict, Any


def generate_analysis(results_file: str):
    """Generate analysis markdown file from results."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate markdown
    md_content = []
    md_content.append("# Heterogeneity Testing Analysis")
    md_content.append("")
    md_content.append("This document contains analysis of 15 heterogeneity experiments.")
    md_content.append("")
    
    # Test descriptions
    test_descriptions = {
        'test_1': "Test 1: Vary L2/Cosine Fraction",
        'test_2': "Test 2: Extreme Voters Use L∞",
        'test_3': "Test 3: Distance Rule Depends on Radius",
        'test_4': "Test 4: Random Distance Function Per Voter",
        'test_5': "Test 5: Utility Nonlinearity",
        'test_6': "Test 6: Strategic Misreporting",
        'test_7': "Test 7: Candidate Clustering",
        'test_8': "Test 8: Thresholds",
        'test_9': "Test 9: Dimensionality Sweep",
        'test_10': "Test 10: Candidate Count Sweep",
        'test_11': "Test 11: Outlier Voters",
        'test_12': "Test 12: Noise in Voter Perception",
        'test_13': "Test 13: Hybrid Distance Switching by Candidate Location",
        'test_14': "Test 14: Incremental Heterogeneity Sweep",
        'test_15': "Test 15: Saturated Utility + Heterogeneous Distance",
    }
    
    # Process each test
    for test_key in sorted(results.keys()):
        test_data = results[test_key]
        test_name = test_data.get('test_name', test_key)
        metrics = test_data.get('metrics', {})
        config = test_data.get('config_summary', {})
        
        md_content.append(f"## {test_descriptions.get(test_key, test_name)}")
        md_content.append("")
        
        # Configuration summary
        if config:
            md_content.append("### Configuration")
            md_content.append("")
            for key, value in config.items():
                md_content.append(f"- **{key}**: {value}")
            md_content.append("")
        
        # Metrics
        md_content.append("### Key Metrics")
        md_content.append("")
        
        # Rule disagreement
        if 'rule_disagreement_pct' in metrics:
            md_content.append(f"- **Rule Disagreement**: {metrics['rule_disagreement_pct']:.2f}%")
            md_content.append("")
        
        # Per-rule metrics
        if 'average_distance_to_ideal' in metrics:
            md_content.append("#### Average Distance to Ideal")
            md_content.append("")
            for rule, stats in metrics['average_distance_to_ideal'].items():
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                md_content.append(f"- **{rule}**: {mean:.4f} ± {std:.4f}")
            md_content.append("")
        
        if 'winner_extremism' in metrics:
            md_content.append("#### Winner Extremism")
            md_content.append("")
            for rule, stats in metrics['winner_extremism'].items():
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                md_content.append(f"- **{rule}**: {mean:.4f} ± {std:.4f}")
            md_content.append("")
        
        if 'worst_off_distance' in metrics:
            md_content.append("#### Worst-Off Distance")
            md_content.append("")
            for rule, stats in metrics['worst_off_distance'].items():
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                md_content.append(f"- **{rule}**: {mean:.4f} ± {std:.4f}")
            md_content.append("")
        
        if 'condorcet_consistency' in metrics:
            md_content.append("#### Condorcet Consistency")
            md_content.append("")
            for rule, value in metrics['condorcet_consistency'].items():
                md_content.append(f"- **{rule}**: {value*100:.2f}%")
            md_content.append("")
        
        md_content.append("---")
        md_content.append("")
    
    # Summary section
    md_content.append("## Summary")
    md_content.append("")
    md_content.append("### Key Findings")
    md_content.append("")
    md_content.append("1. **Heterogeneity Impact**: Different distance metrics lead to different outcomes")
    md_content.append("2. **Rule Sensitivity**: Voting rules respond differently to heterogeneity")
    md_content.append("3. **Extremism Effects**: Extreme voters using different metrics affect outcomes")
    md_content.append("")
    
    # Write to file
    output_file = os.path.join(os.path.dirname(results_file), 'HETEROGENITY_ANALYSIS.md')
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_content))
    
    print(f"Analysis written to: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = os.path.join(os.path.dirname(__file__), 'results.json')
    
    generate_analysis(results_file)







