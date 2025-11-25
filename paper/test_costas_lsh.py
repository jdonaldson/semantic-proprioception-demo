#!/usr/bin/env python3
"""
Test Costas Array LSH vs Bit Slicing

Costas arrays have optimal autocorrelation properties - all displacement
vectors between pairs are distinct. This could translate to better-separated
LSH buckets.

Approach: Use Costas array to define structured rotation angles for hyperplanes.
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

def generate_costas_welch(n):
    """
    Generate Costas array using Welch construction

    For prime n, uses primitive root to construct permutation
    with optimal autocorrelation properties.
    """
    # Find smallest prime >= n
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    p = n
    while not is_prime(p):
        p += 1

    # Find primitive root modulo p
    def primitive_root(p):
        """Find smallest primitive root of prime p"""
        if p == 2:
            return 1
        # Try small values
        for g in range(2, p):
            # Check if g is primitive root
            powers = set()
            for i in range(1, p):
                powers.add(pow(g, i, p))
            if len(powers) == p - 1:
                return g
        return 2  # Fallback

    g = primitive_root(p)

    # Welch construction: costas[i] = g^i mod p
    costas = []
    for i in range(1, min(n+1, p)):
        costas.append(pow(g, i, p) % n)

    # Pad if needed
    while len(costas) < n:
        costas.append(len(costas) % n)

    return np.array(costas[:n])

def lsh_costas(embeddings, num_bits=8, seed=12345):
    """
    LSH using Costas array to define structured hyperplane rotations

    Approach:
    1. Generate Costas array of size num_bits
    2. Use array to define rotation angles in 2D subspaces
    3. Apply structured rotations to random base directions
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Generate Costas array
    costas = generate_costas_welch(num_bits)

    # Map Costas positions to angles: position / num_bits * 2π
    angles = (costas / num_bits) * 2 * np.pi

    # Generate base random directions
    base_hyperplanes = np.random.randn(num_bits, d)

    # Apply structured rotations based on Costas angles
    # Rotate in pairs of dimensions using Costas-defined angles
    hyperplanes = base_hyperplanes.copy()
    for i in range(num_bits):
        angle = angles[i]
        # Apply rotation in first two dimensions
        if d >= 2:
            # 2D rotation matrix
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            hyperplanes[i, :2] = rot @ hyperplanes[i, :2]

    # Normalize
    hyperplanes = hyperplanes / (np.linalg.norm(hyperplanes, axis=1, keepdims=True) + 1e-10)

    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

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
        hash_16bit = hash_func(embeddings, num_bits=16)
        hash_8bit = hash_16bit & 0xFF
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

            # Extract level 1 hash (bits 8-11)
            if name == "Bit Slicing":
                level_1_hash = (hash_16bit[indices] >> 8) & 0x0F
            elif name == "Costas Array":
                # For Costas, compute new hash at higher level
                sub_embeddings = embeddings[indices]
                sub_hash = hash_func(sub_embeddings, num_bits=4)
                level_1_hash = sub_hash
            else:
                # For other techniques, compute new hash
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
print("COSTAS ARRAY LSH TEST")
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
    (lsh_costas, "Costas Array"),
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

# Compare against baseline
baseline = [r for r in results if r['name'] == "Random (baseline)"][0]
bit_slicing = [r for r in results if r['name'] == "Bit Slicing"][0]
costas = [r for r in results if r['name'] == "Costas Array"][0]

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()

print("vs Random baseline:")
print(f"  Bit Slicing:  coherence {bit_slicing['refined_coherence'] - baseline['refined_coherence']:+.3f}  "
      f"speed {baseline['time_ms'] / bit_slicing['time_ms']:.1f}× faster")
print(f"  Costas Array: coherence {costas['refined_coherence'] - baseline['refined_coherence']:+.3f}  "
      f"speed {baseline['time_ms'] / costas['time_ms']:.1f}× faster")
print()

print("Costas vs Bit Slicing:")
coherence_diff = costas['refined_coherence'] - bit_slicing['refined_coherence']
speed_ratio = costas['time_ms'] / bit_slicing['time_ms']
print(f"  Coherence: {coherence_diff:+.3f}")
print(f"  Speed: {speed_ratio:.1f}× {'slower' if speed_ratio > 1 else 'faster'}")
print()

if coherence_diff > 0.01:
    print("✓ COSTAS ARRAY WINS")
    print(f"  → {costas['refined_coherence']:.3f} refined coherence")
    print(f"  → {coherence_diff:+.3f} better than bit slicing")
elif coherence_diff < -0.01:
    print("✗ COSTAS ARRAY LOSES")
    print(f"  → {coherence_diff:+.3f} worse than bit slicing")
    print("  → Bit slicing remains the winner")
else:
    print("⚠️  SIMILAR PERFORMANCE")
    print(f"  → Only {coherence_diff:+.3f} difference")
    print("  → No significant advantage")

print()
print("=" * 70)
