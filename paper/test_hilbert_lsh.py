#!/usr/bin/env python3
"""
Test Hilbert Curve and Z-order (Morton) LSH vs Bit Slicing

Space-filling curves preserve locality when mapping high-D → 1D.
This could provide better semantic grouping than random hyperplanes.

Approaches:
1. Z-order (Morton codes): Easy to compute, interleaves bits
2. Hilbert curve: Better locality preservation, more complex

For 384D embeddings, we'll project down to manageable dimension first.
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

# ============================================================================
# Z-order (Morton code) encoding
# ============================================================================
def morton_encode(coords, bits_per_dim=2):
    """
    Encode n-dimensional coordinates into Morton (Z-order) code

    Interleaves bits from each dimension to preserve locality
    """
    n_dims = len(coords)
    morton = 0

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            # Extract bit from this dimension
            bit_val = (coords[dim] >> bit) & 1
            # Position in interleaved sequence
            position = bit * n_dims + dim
            morton |= (bit_val << position)

    return morton

def lsh_morton(embeddings, num_bits=8, seed=12345):
    """
    LSH using Morton (Z-order) codes

    Approach:
    1. Project to num_bits dimensions using random projection
    2. Quantize each dimension to 2-bit values (4 levels)
    3. Interleave bits using Morton encoding
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Random projection to num_bits dimensions
    projection = np.random.randn(d, num_bits)
    projection = projection / (np.linalg.norm(projection, axis=0, keepdims=True) + 1e-10)

    projected = embeddings @ projection  # (n, num_bits)

    # Quantize to 2 bits per dimension (4 levels: 0,1,2,3)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    ranges = maxs - mins + 1e-10
    quantized = ((projected - mins) / ranges * 4).astype(int)
    quantized = np.clip(quantized, 0, 3)

    # Morton encode
    hash_values = []
    for coords in quantized:
        morton = morton_encode(coords, bits_per_dim=2)
        hash_values.append(morton)

    return np.array(hash_values)

# ============================================================================
# Hilbert curve encoding (compact version)
# ============================================================================
def gray_code(n):
    """Convert binary to Gray code"""
    return n ^ (n >> 1)

def inverse_gray_code(g):
    """Convert Gray code to binary"""
    n = g
    while g > 0:
        g >>= 1
        n ^= g
    return n

def hilbert_encode_2d(x, y, order):
    """
    Encode 2D coordinates to Hilbert curve index

    Simple implementation for low orders
    """
    n = 1 << order  # 2^order

    # Clip to range
    x = max(0, min(n-1, x))
    y = max(0, min(n-1, y))

    index = 0
    s = n // 2

    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0

        index += s * s * ((3 * rx) ^ ry)

        # Rotate coordinates
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x

        s //= 2

    return index

def lsh_hilbert(embeddings, num_bits=8, seed=12345):
    """
    LSH using Hilbert curve

    Approach:
    1. Project to 2D using random projection (pairs of dimensions)
    2. Quantize to grid
    3. Map to Hilbert curve positions
    4. Use curve position as hash

    We'll do multiple 2D Hilbert curves and combine them
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # How many 2D projections do we need?
    # Each 2D Hilbert with order=2 gives us 4 bits
    # For 8-bit hash, we need 2 projections
    n_projections = num_bits // 4
    if num_bits % 4 != 0:
        n_projections += 1

    order = 2  # 2^2 = 4x4 grid per projection

    hash_values = []

    for emb in embeddings:
        hash_val = 0

        for proj_idx in range(n_projections):
            # Random 2D projection
            np.random.seed(seed + proj_idx)
            proj_matrix = np.random.randn(d, 2)
            proj_matrix = proj_matrix / (np.linalg.norm(proj_matrix, axis=0, keepdims=True) + 1e-10)

            # Project embedding
            projected = emb @ proj_matrix  # (2,)

            # Quantize to grid
            # Map to [0, 2^order - 1]
            grid_size = 1 << order

            x = int((projected[0] + 3) / 6 * grid_size)  # Assume embeddings in roughly [-3, 3]
            y = int((projected[1] + 3) / 6 * grid_size)

            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))

            # Hilbert encode
            hilbert_idx = hilbert_encode_2d(x, y, order)

            # Add to hash (shift by 4 bits per projection)
            hash_val |= (hilbert_idx << (proj_idx * 4))

        hash_values.append(hash_val)

    return np.array(hash_values)

# ============================================================================
# Reference techniques
# ============================================================================
def lsh_bit_slicing(embeddings, num_bits=16, seed=12345):
    """Bit slicing (current winner)"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

def lsh_random(embeddings, num_bits=8, seed=12345):
    """Random hyperplanes (baseline)"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

# ============================================================================
# Evaluation
# ============================================================================
def compute_intra_bucket_similarity(embeddings):
    """Compute average pairwise cosine similarity within bucket"""
    if len(embeddings) < 2:
        return None

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
            similarities.append(cos_sim)

    return np.mean(similarities)

def evaluate_technique(embeddings, hash_func, name, use_refinement=False):
    """Evaluate LSH technique on embeddings"""
    start = time.perf_counter()

    if use_refinement:
        # Hierarchical refinement with bit slicing
        if name == "Bit Slicing":
            hash_16bit = hash_func(embeddings, num_bits=16)
            hash_8bit = hash_16bit & 0xFF
        else:
            # For space-filling curves, compute at both levels
            hash_8bit = hash_func(embeddings, num_bits=8)
            hash_16bit = hash_func(embeddings, num_bits=16)
    else:
        hash_8bit = hash_func(embeddings, num_bits=8)

    time_hash = time.perf_counter() - start

    # Group into buckets
    buckets = {}
    for i, hash_val in enumerate(hash_8bit):
        if hash_val not in buckets:
            buckets[hash_val] = []
        buckets[hash_val].append(i)

    # Find dense buckets
    dense_buckets = {bid: indices for bid, indices in buckets.items() if len(indices) >= 5}

    # Compute coherence
    coherences = []
    for indices in dense_buckets.values():
        bucket_embeddings = embeddings[indices]
        coh = compute_intra_bucket_similarity(bucket_embeddings)
        if coh is not None:
            coherences.append(coh)

    avg_coherence = np.mean(coherences) if len(coherences) > 0 else 0.0

    # Hierarchical refinement if enabled
    refined_coherence = avg_coherence
    if use_refinement:
        refined_coherences = []
        for indices in dense_buckets.values():
            if len(indices) < 6:
                continue

            # Extract level 1 hash
            if name == "Bit Slicing":
                level_1_hash = (hash_16bit[indices] >> 8) & 0x0F
            else:
                # For space-filling curves, compute new hash
                sub_embeddings = embeddings[indices]
                sub_hash = hash_func(sub_embeddings, num_bits=4)
                level_1_hash = sub_hash

            # Group into sub-buckets
            sub_buckets = {}
            for i, sub_hash_val in enumerate(level_1_hash):
                if sub_hash_val not in sub_buckets:
                    sub_buckets[sub_hash_val] = []
                sub_buckets[sub_hash_val].append(indices[i])

            # Compute coherence for sub-buckets
            for sub_indices in sub_buckets.values():
                if len(sub_indices) >= 2:
                    sub_embeddings = embeddings[sub_indices]
                    sub_coh = compute_intra_bucket_similarity(sub_embeddings)
                    if sub_coh is not None:
                        refined_coherences.append(sub_coh)

        refined_coherence = np.mean(refined_coherences) if len(refined_coherences) > 0 else avg_coherence

    return {
        'name': name,
        'num_buckets': len(buckets),
        'num_dense': len(dense_buckets),
        'avg_coherence': avg_coherence,
        'refined_coherence': refined_coherence,
        'improvement': refined_coherence - avg_coherence,
        'time_ms': time_hash * 1000
    }

# ============================================================================
# Main Experiment
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("SPACE-FILLING CURVE LSH TEST")
print("=" * 70)
print()

# Load Twitter data
embeddings_file = TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet"
df_embeddings = pl.read_parquet(embeddings_file)
all_embeddings = np.array([row for row in df_embeddings['embedding']])

print(f"Loaded {len(all_embeddings)} embeddings (dim={all_embeddings.shape[1]})")
print()

# Test techniques
techniques = [
    (lsh_random, "Random (baseline)"),
    (lsh_bit_slicing, "Bit Slicing"),
    (lsh_morton, "Z-order (Morton)"),
    (lsh_hilbert, "Hilbert Curve"),
]

results = []

print("Testing techniques (with hierarchical refinement)...")
print()

for hash_func, name in techniques:
    try:
        result = evaluate_technique(all_embeddings, hash_func, name, use_refinement=True)
        results.append(result)
        print(f"✓ {name:<25s} coherence: {result['avg_coherence']:.3f} → {result['refined_coherence']:.3f} (+{result['improvement']:.3f})")
    except Exception as e:
        print(f"✗ {name:<25s} FAILED: {e}")
        import traceback
        traceback.print_exc()

print()

# ============================================================================
# Results
# ============================================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"{'Technique':<25s} {'Buckets':<10s} {'Dense':<8s} {'Initial':<10s} {'Refined':<10s} {'Δ':<10s} {'Time':<10s}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<25s} {r['num_buckets']:<10d} {r['num_dense']:<8d} "
          f"{r['avg_coherence']:<10.3f} {r['refined_coherence']:<10.3f} "
          f"{r['improvement']:>+8.3f}  {r['time_ms']:>7.2f}ms")

print()

# Compare against baseline and bit slicing
baseline = [r for r in results if r['name'] == "Random (baseline)"][0]
bit_slicing = [r for r in results if r['name'] == "Bit Slicing"][0]

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()

print("vs Bit Slicing (current winner):")
for r in results:
    if r['name'] in ["Random (baseline)", "Bit Slicing"]:
        continue

    coherence_diff = r['refined_coherence'] - bit_slicing['refined_coherence']
    speed_ratio = r['time_ms'] / bit_slicing['time_ms']

    print(f"  {r['name']:<20s} coherence {coherence_diff:+.3f}  "
          f"speed {speed_ratio:.1f}× {'slower' if speed_ratio > 1 else 'faster'}")

print()

# Find best
best = max(results, key=lambda x: x['refined_coherence'])

if best['name'] == "Bit Slicing":
    print("✓ BIT SLICING REMAINS THE WINNER")
    print(f"  → {best['refined_coherence']:.3f} refined coherence")
    print("  → Space-filling curves didn't improve performance")
else:
    print(f"✓ NEW WINNER: {best['name']}")
    print(f"  → {best['refined_coherence']:.3f} refined coherence")
    print(f"  → {best['refined_coherence'] - bit_slicing['refined_coherence']:+.3f} better than bit slicing")

print()
print("=" * 70)
