import numpy as np
import time

def compute_average_distance(dimension, n_points=100000, n_sample_pairs=100000):
    """
    Sample n_points uniformly from [-1,1]^d and compute average normalized distance.
    Uses random sampling of pairs to avoid computing all C(n_points, 2) pairs.
    """
    print(f"Dimension {dimension:3d}: Sampling {n_points:,} points...", end=" ", flush=True)
    start_time = time.time()

    # Sample points uniformly from [-1,1]^d
    points = np.random.uniform(-1, 1, size=(n_points, dimension))

    # Sample random pairs (without replacement if possible, but with replacement is fine for large n)
    # We'll sample n_sample_pairs random pairs
    indices_i = np.random.randint(0, n_points, size=n_sample_pairs)
    indices_j = np.random.randint(0, n_points, size=n_sample_pairs)

    # Avoid self-pairs
    mask = indices_i != indices_j
    indices_i = indices_i[mask]
    indices_j = indices_j[mask]

    # Compute distances for sampled pairs
    distances = np.linalg.norm(points[indices_i] - points[indices_j], axis=1)

    # Normalize by sqrt(d)
    normalized_distances = distances / np.sqrt(dimension)

    # Compute statistics
    mean_dist = np.mean(normalized_distances)
    std_dist = np.std(normalized_distances)

    elapsed = time.time() - start_time
    print(f"Mean: {mean_dist:.6f}, Std: {std_dist:.6f} (took {elapsed:.2f}s)")

    return mean_dist, std_dist

# Dimensions to test
dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200]
n_points = 100000
n_sample_pairs = 100000  # Sample this many pairs for each dimension

print("="*80)
print(f"Research: Average Normalized Distance Between Points")
print(f"Configuration: {n_points:,} points per dimension, {n_sample_pairs:,} sampled pairs")
print("="*80)
print()

results = []
theoretical_mean = np.sqrt(2/3)  # Theoretical value: sqrt(2/3) ≈ 0.816497

for d in dims:
    mean_dist, std_dist = compute_average_distance(d, n_points, n_sample_pairs)
    deviation = abs(mean_dist - theoretical_mean)
    results.append({
        'dimension': d,
        'mean': mean_dist,
        'std': std_dist,
        'deviation': deviation
    })

print()
print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'Dimension':<12} {'Mean Distance':<18} {'Std Dev':<18} {'Deviation from Theory':<25}")
print(f"{'':<12} {'(normalized)':<18} {'(normalized)':<18} {'(theory = 0.816497)':<25}")
print("-"*80)

for r in results:
    print(f"{r['dimension']:<12} {r['mean']:<18.6f} {r['std']:<18.6f} {r['deviation']:<25.6f}")

print()
print(f"Theoretical mean: {theoretical_mean:.6f}")
print("="*80)

# Additional analysis
print()
print("Convergence Analysis:")
print(f"  Mean distance at dimension 1: {results[0]['mean']:.6f}")
print(f"  Mean distance at dimension 200: {results[-1]['mean']:.6f}")
print(f"  Convergence to theory: {abs(results[-1]['mean'] - theoretical_mean):.6f}")
print()
print("Standard deviation trend:")
print(f"  Std at dimension 1: {results[0]['std']:.6f}")
print(f"  Std at dimension 200: {results[-1]['std']:.6f}")
print(f"  Reduction factor: {results[0]['std'] / results[-1]['std']:.2f}x")
