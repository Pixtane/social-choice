import numpy as np
import matplotlib.pyplot as plt

def sample_distances(dimension, n_pairs=10000):
    """Sample n_pairs of random points from [-1,1]^d and compute distances."""
    distances = []
    for _ in range(n_pairs):
        x = np.random.uniform(-1, 1, dimension)
        y = np.random.uniform(-1, 1, dimension)
        dist = np.linalg.norm(x - y)
        distances.append(dist)
    return np.array(distances)

# Dimensions to test
dims = [1, 2, 3, 5, 10, 20, 50, 100]
n_pairs = 10000

# Compute distances for each dimension
results = {}
for d in dims:
    print(f"Computing distances for dimension {d}...")
    distances = sample_distances(d, n_pairs)
    results[d] = distances

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, d in enumerate(dims):
    ax = axes[idx]
    distances = results[d]

    # Normalize by sqrt(d) for comparison
    normalized_distances = distances / np.sqrt(d)

    # Create histogram
    ax.hist(normalized_distances, bins=50, density=True, alpha=0.7,
            color=plt.cm.viridis(idx / (len(dims) - 1)), edgecolor='black', linewidth=0.5)

    # Add mean line
    mean_dist = np.mean(normalized_distances)
    std_dist = np.std(normalized_distances)
    ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.3f}')
    ax.axvline(mean_dist - std_dist, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(mean_dist + std_dist, color='orange', linestyle=':', linewidth=1, alpha=0.7)

    # Analytical mean (approximation: sqrt(2/3)*(1-7/(40*d)))
    analytical_mean = np.sqrt(2/3) * (1 - 7/(40*d))
    ax.axvline(analytical_mean, color='blue', linestyle='--', linewidth=1,
               alpha=0.5, label=f'Analytical: {analytical_mean:.3f}')

    ax.set_title(f'Dimension {d}\nStd: {std_dist:.4f}', fontsize=10)
    ax.set_xlabel('Normalized Distance / sqrt(d)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Proposition 2.2: Distance Uniformity - Distribution Concentrates as Dimension Increases',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distance_uniformity_visualization.png', dpi=300, bbox_inches='tight')
print("Saved visualization to distance_uniformity_visualization.png")

# Create a second plot showing variance convergence
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Coefficient of variation (std/mean) vs dimension
means = [np.mean(results[d] / np.sqrt(d)) for d in dims]
stds = [np.std(results[d] / np.sqrt(d)) for d in dims]
coeff_vars = [std/mean for std, mean in zip(stds, means)]

ax1.plot(dims, coeff_vars, 'o-', linewidth=2, markersize=8, color='purple')
ax1.set_xlabel('Dimension', fontsize=12)
ax1.set_ylabel('Coefficient of Variation (σ/μ)', fontsize=12)
ax1.set_title('Distance Distribution Variance Shrinks with Dimension', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Standard deviation vs dimension
ax2.plot(dims, stds, 'o-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Dimension', fontsize=12)
ax2.set_ylabel('Standard Deviation (normalized)', fontsize=12)
ax2.set_title('Standard Deviation Decreases with Dimension', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('distance_variance_convergence.png', dpi=300, bbox_inches='tight')
print("Saved variance convergence plot to distance_variance_convergence.png")

# Print statistics
print("\n" + "="*60)
print("Distance Statistics (normalized by sqrt(d)):")
print("="*60)
print(f"{'Dimension':<12} {'Mean':<12} {'Std':<12} {'CV (std/mean)':<12}")
print("-"*60)
for d in dims:
    normalized = results[d] / np.sqrt(d)
    mean = np.mean(normalized)
    std = np.std(normalized)
    cv = std / mean
    print(f"{d:<12} {mean:<12.6f} {std:<12.6f} {cv:<12.6f}")

plt.show()
