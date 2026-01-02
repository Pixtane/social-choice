"""
Centrality / effective-radius concentration analysis vs dimension.

This is a geometry-only diagnostic (no voting rules) that uses the same
hypercube sampling and the same normalized L2 centrality definition used by
the heterogeneous distance strategies.

Key idea:
  If X_i ~ Uniform(position_min, position_max) with center at midpoint,
  then for large d:
    (1/d) * sum_i X_i^2 -> E[X_1^2]
  so ||X|| / (sqrt(d) * (range/2)) -> sqrt(E[X_1^2]) / (range/2).

For position_min=-1, position_max=1 => range/2=1 and E[X_1^2]=1/3,
so normalized centrality concentrates near sqrt(1/3) ~= 0.57735.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add repo root to path (match other scripts in this folder)
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Reuse the exact normalization used in the simulator
from simulator.heterogeneous_distance import compute_voter_centrality


@dataclass
class ConcentrationConfig:
    position_min: float = -1.0
    position_max: float = 1.0
    # Profiles correspond to repeated draws of a whole electorate (N voters)
    n_profiles: int = 500
    n_voters: int = 200
    # Percentile cutoff t used to define effective radius in percentile mode
    threshold: float = 0.5
    # Dimensions to test (includes 200 by default)
    dimensions: List[int] = None
    # Histogram settings for centrality in [0, 1]
    hist_bins: int = 50
    # Output
    output_dir: str = "heterogenity-simulator/results"
    output_file: str = "centrality_concentration_report.json"
    rng_seed: int = 42

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = [1, 2, 3, 5, 10, 20, 50, 100, 200]


def _summary_stats(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qv = np.quantile(x, qs).tolist()
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "quantiles": {f"p{int(q*100):02d}": float(v) for q, v in zip(qs, qv)},
    }


def _histogram(x: np.ndarray, bins: int) -> Dict[str, Any]:
    counts, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    return {
        "bins": int(bins),
        "edges": [float(e) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


def _effective_radius_per_profile(
    centrality: np.ndarray,
    t: float,
) -> float:
    """
    Match the simulator's percentile cutoff definition:
      K = floor(t * N)
      r_eff = 0 if K==0 else c_sorted[K-1]
    """
    n = int(centrality.shape[0])
    k = int(np.floor(float(t) * n))
    if k <= 0:
        return 0.0
    order = np.argsort(centrality, kind="mergesort")
    return float(centrality[order[k - 1]])


def run_concentration_report(config: ConcentrationConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(config.rng_seed)

    # Theory target for normalized centrality under Uniform(-1,1)^d with center at 0.
    # For general [a,b], the centered coordinate U ~ Uniform(-(b-a)/2, (b-a)/2),
    # so normalized centrality concentrates near sqrt(E[U^2]) / ((b-a)/2) = sqrt(1/3).
    sqrt_one_third = float(np.sqrt(1.0 / 3.0))

    out: Dict[str, Any] = {
        "config": asdict(config),
        "theory": {
            "sqrt_one_third": sqrt_one_third,
            "note": "For uniform hypercube sampling with normalized L2 centrality, centrality concentrates near sqrt(1/3) as dimension increases.",
        },
        "by_dimension": {},
    }

    for d in config.dimensions:
        d = int(d)
        center = np.full(d, (config.position_min + config.position_max) / 2.0, dtype=float)

        # Sample (profiles, voters, dim)
        positions = rng.uniform(
            low=config.position_min,
            high=config.position_max,
            size=(config.n_profiles, config.n_voters, d),
        )

        # Centrality per voter (profiles, voters)
        centralities = np.stack(
            [
                compute_voter_centrality(
                    positions[i],
                    position_min=config.position_min,
                    position_max=config.position_max,
                )
                for i in range(config.n_profiles)
            ],
            axis=0,
        )

        # Raw L2 distances from center (profiles, voters)
        distances = np.linalg.norm(positions - center, axis=-1)

        # Distribution over voters (flatten)
        flat = centralities.reshape(-1)
        centrality_stats = _summary_stats(flat)
        centrality_hist = _histogram(flat, bins=config.hist_bins)

        # "Maximum distance from center in a profile" (raw units, not normalized)
        max_dist_per_profile = np.max(distances, axis=1)  # (profiles,)
        max_dist_stats = _summary_stats(max_dist_per_profile)

        # Distribution over profiles of the cutoff (effective radius)
        r_eff = np.asarray(
            [
                _effective_radius_per_profile(centralities[i], config.threshold)
                for i in range(config.n_profiles)
            ],
            dtype=float,
        )
        eff_stats = _summary_stats(r_eff)
        eff_hist = _histogram(r_eff, bins=config.hist_bins)

        out["by_dimension"][str(d)] = {
            "centrality": {
                "stats": centrality_stats,
                "hist": centrality_hist,
            },
            "max_distance_from_center_per_profile": {
                "units": "raw_l2",
                "stats": max_dist_stats,
            },
            "effective_radius_percentile_mode": {
                "threshold": float(config.threshold),
                "stats": eff_stats,
                "hist": eff_hist,
            },
        }

    return out


def main(argv: Optional[List[str]] = None) -> None:
    # CLI kept intentionally minimal: edit config in code if needed.
    config = ConcentrationConfig()
    report = run_concentration_report(config)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.output_file
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
