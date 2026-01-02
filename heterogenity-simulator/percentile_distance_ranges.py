"""
Research: Distance ranges for percentile bands of voters sorted by centrality.

For each dimension, divides voters into 10% percentile bands (0-10%, 10-20%, ..., 90-100%)
based on their centrality, and reports the range of raw L2 distances from center
for each band.
"""

import numpy as np
import json
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simulator.heterogeneous_distance import compute_voter_centrality


def compute_percentile_distance_ranges(
    dimensions: list[int],
    n_profiles: int = 500,
    n_voters: int = 200,
    position_min: float = -1.0,
    position_max: float = 1.0,
    rng_seed: int = 42
) -> dict:
    """
    Compute distance ranges for 10% percentile bands of voters.

    Returns a dict with structure:
    {
        "by_dimension": {
            "1": {
                "percentile_bands": {
                    "0-10%": {"min_dist": ..., "max_dist": ..., "mean_dist": ..., "median_dist": ...},
                    "10-20%": {...},
                    ...
                    "90-100%": {...}
                }
            },
            ...
        }
    }
    """
    rng = np.random.default_rng(rng_seed)
    results = {
        "config": {
            "position_min": position_min,
            "position_max": position_max,
            "n_profiles": n_profiles,
            "n_voters": n_voters,
            "percentile_step": 0.1,
            "dimensions": dimensions,
            "rng_seed": rng_seed
        },
        "by_dimension": {}
    }

    # Define percentile bands (0-10%, 10-20%, ..., 90-100%)
    percentile_bands = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
    ]
    band_labels = [
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
    ]

    for dim in dimensions:
        print(f"Processing dimension {dim}...", flush=True)

        # Storage: for each percentile band, collect raw distances and normalized centrality
        band_distances = {label: [] for label in band_labels}
        band_centralities = {label: [] for label in band_labels}

        center = np.full(dim, (position_min + position_max) / 2.0)
        max_distance = np.sqrt(dim) * (position_max - position_min) / 2.0

        for profile_idx in range(n_profiles):
            # Generate random voter positions
            voter_positions = rng.uniform(
                low=position_min,
                high=position_max,
                size=(n_voters, dim)
            )

            # Compute raw L2 distances from center
            raw_distances = np.linalg.norm(voter_positions - center, axis=-1)

            # Compute normalized centrality (0=center, 1=corner)
            centrality = compute_voter_centrality(
                voter_positions,
                center=center,
                position_min=position_min,
                position_max=position_max
            )

            # Sort voters by centrality (ascending: most central first)
            sort_order = np.argsort(centrality, kind='mergesort')
            sorted_centrality = centrality[sort_order]
            sorted_distances = raw_distances[sort_order]

            # Assign each voter to a percentile band
            for i, voter_idx in enumerate(sort_order):
                # Percentile rank (0.0 = most central, 1.0 = most extreme)
                percentile_rank = i / n_voters

                # Find which band this voter belongs to
                for band_idx, (low, high) in enumerate(percentile_bands):
                    if low <= percentile_rank < high or (band_idx == len(percentile_bands) - 1 and percentile_rank == 1.0):
                        band_label = band_labels[band_idx]
                        band_distances[band_label].append(raw_distances[voter_idx])
                        band_centralities[band_label].append(centrality[voter_idx])
                        break

        # Compute statistics for each band
        band_stats = {}
        for label in band_labels:
            distances = np.array(band_distances[label])
            centralities = np.array(band_centralities[label])
            if len(distances) > 0:
                band_stats[label] = {
                    "n_voters": int(len(distances)),
                    # Raw L2 distances
                    "raw_distance": {
                        "min": float(np.min(distances)),
                        "max": float(np.max(distances)),
                        "mean": float(np.mean(distances)),
                        "median": float(np.median(distances)),
                        "std": float(np.std(distances)),
                        "p05": float(np.percentile(distances, 5)),
                        "p95": float(np.percentile(distances, 95))
                    },
                    # Normalized centrality (0-1, used for radius threshold)
                    "normalized_centrality": {
                        "min": float(np.min(centralities)),
                        "max": float(np.max(centralities)),
                        "mean": float(np.mean(centralities)),
                        "median": float(np.median(centralities)),
                        "std": float(np.std(centralities)),
                        "p05": float(np.percentile(centralities, 5)),
                        "p95": float(np.percentile(centralities, 95))
                    }
                }
            else:
                band_stats[label] = {
                    "n_voters": 0,
                    "raw_distance": {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "median": None,
                        "std": None,
                        "p05": None,
                        "p95": None
                    },
                    "normalized_centrality": {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "median": None,
                        "std": None,
                        "p05": None,
                        "p95": None
                    }
                }

        results["by_dimension"][str(dim)] = {
            "percentile_bands": band_stats,
            "max_possible_distance": float(max_distance),
            "note": "Percentile bands are based on normalized centrality (0=center, 1=corner). Each band includes both raw L2 distances and normalized centrality values (0-1, used for radius threshold calculation)."
        }

    return results


def main():
    dimensions = [1, 2, 3, 5, 10, 20, 50, 100, 200]

    print("=" * 80)
    print("PERCENTILE DISTANCE RANGE ANALYSIS")
    print("=" * 80)
    print(f"Dimensions: {dimensions}")
    print(f"Profiles per dimension: 500")
    print(f"Voters per profile: 200")
    print("=" * 80)
    print()

    start_time = time.perf_counter()
    results = compute_percentile_distance_ranges(
        dimensions=dimensions,
        n_profiles=500,
        n_voters=200,
        position_min=-1.0,
        position_max=1.0,
        rng_seed=42
    )
    elapsed_time = time.perf_counter() - start_time

    # Save results
    output_dir = Path("heterogenity-simulator/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "percentile_distance_ranges.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total computation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Print summary table for raw distances
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Raw L2 Distance by Percentile Band")
    print("=" * 80)
    print(f"{'Dimension':<12}", end="")
    for label in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]:
        print(f"{label:>10}", end="")
    print()
    print("-" * 80)

    for dim in dimensions:
        dim_str = str(dim)
        print(f"{dim_str:<12}", end="")
        for label in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                      "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]:
            stats = results["by_dimension"][dim_str]["percentile_bands"][label]
            mean_dist = stats.get("raw_distance", {}).get("mean")
            if mean_dist is not None:
                print(f"{mean_dist:>10.4f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("=" * 80)

    # Print summary table for normalized centrality
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Normalized Centrality (0-1) by Percentile Band")
    print("=" * 80)
    print(f"{'Dimension':<12}", end="")
    for label in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                  "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]:
        print(f"{label:>10}", end="")
    print()
    print("-" * 80)

    for dim in dimensions:
        dim_str = str(dim)
        print(f"{dim_str:<12}", end="")
        for label in ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                      "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]:
            stats = results["by_dimension"][dim_str]["percentile_bands"][label]
            mean_cent = stats.get("normalized_centrality", {}).get("mean")
            if mean_cent is not None:
                print(f"{mean_cent:>10.4f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
