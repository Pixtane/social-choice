"""
Quick verification script with reduced sample sizes for faster testing.

This version uses fewer profiles to verify findings more quickly,
useful for development and debugging.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from verify_findings import FindingsVerifier

if __name__ == '__main__':
    # Quick version with reduced sample sizes
    print("=" * 80)
    print("QUICK VERIFICATION (Reduced Sample Sizes)")
    print("=" * 80)
    print("Note: Results may be less accurate than full verification")
    print("=" * 80)
    print()
    
    verifier = FindingsVerifier(
        n_profiles=50,  # Reduced from 200
        n_voters=100,
        n_candidates=5
    )
    
    verifier.run_all_verifications()






