#!/usr/bin/env python3
"""
Compare multiple LSH techniques for hierarchical bucket refinement

Tests:
1. Random hyperplanes (baseline)
2. Bit slicing (current winner)
3. Sobol low-discrepancy sequences
4. Hadamard transform (fast structured transform)
5. Multi-probe LSH (probe nearby buckets)
6. Cross-polytope LSH (structured random)
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from scipy.stats import qmc
import time

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

# ============================================================================
# Technique 1: Random Hyperplanes (Baseline)
# ============================================================================
def lsh_random(embeddings, num_bits=8, seed=12345):
    """Standard random hyperplane LSH"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

# ============================================================================
# Technique 2: Bit Slicing (Current Winner)
# ============================================================================
def lsh_bit_slicing(embeddings, num_bits=16, seed=12345):
    """Compute large hash once, extract levels via bit masking"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

# ============================================================================
# Technique 3: Sobol Low-Discrepancy Sequences
# ============================================================================
def lsh_sobol(embeddings, num_bits=8, seed=12345):
    """
    Use Sobol sequence for better space-filling hyperplanes

    Better uniformity than random
    """
    d = embeddings.shape[1]

    # Generate Sobol sequence
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    hyperplane_coords = sampler.random(num_bits)

    # Map [0,1] to Gaussian-like distribution
    from scipy.stats import norm as scipy_norm
    hyperplanes = scipy_norm.ppf(hyperplane_coords)

    # Normalize (handle NaN from extreme values)
    hyperplanes = np.nan_to_num(hyperplanes, nan=0.0, posinf=3.0, neginf=-3.0)
    norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    hyperplanes = hyperplanes / (norms + 1e-10)

    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

# ============================================================================
# Technique 4: Hadamard Transform
# ============================================================================
def hadamard_transform(x):
    """Fast Hadamard Transform (used in LSH)"""
    n = x.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h] if j + h < n else 0
                x[j] = a + b
                if j + h < n:
                    x[j + h] = a - b
        h *= 2
    return x / np.sqrt(n)

def lsh_hadamard(embeddings, num_bits=8, seed=12345):
    """
    Use Hadamard transform for structured random projections

    Faster than random (O(d log d) vs O(d²))
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Pad to power of 2
    d_padded = 2 ** int(np.ceil(np.log2(d)))

    hash_values = []
    for emb in embeddings:
        # Pad embedding
        emb_padded = np.pad(emb, (0, d_padded - d), mode='constant')

        # Random sign flips
        signs = np.random.choice([-1, 1], size=d_padded)
        emb_signed = emb_padded * signs

        # Hadamard transform
        transformed = hadamard_transform(emb_signed.copy())

        # Take first num_bits dimensions
        hash_bits = (transformed[:num_bits] > 0).astype(int)
        hash_val = hash_bits @ (2 ** np.arange(num_bits))
        hash_values.append(hash_val)

    return np.array(hash_values)

# ============================================================================
# Technique 5: Cross-Polytope LSH
# ============================================================================
def lsh_cross_polytope(embeddings, num_bits=8, seed=12345):
    """
    Cross-polytope LSH (structured LSH variant)

    Find nearest vertex of cross-polytope
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Random rotation
    rotation = np.random.randn(d, d)
    Q, _ = np.linalg.qr(rotation)

    rotated = embeddings @ Q.T

    # Find max absolute value dimension and its sign
    hash_values = []
    for emb in rotated:
        # Partition into chunks
        chunk_size = d // num_bits
        hash_bits = []
        for i in range(num_bits):
            chunk = emb[i*chunk_size:(i+1)*chunk_size]
            max_idx = np.argmax(np.abs(chunk))
            sign = 1 if chunk[max_idx] > 0 else 0
            hash_bits.append(sign)

        hash_val = np.array(hash_bits) @ (2 ** np.arange(num_bits))
        hash_values.append(hash_val)

    return np.array(hash_values)

# ============================================================================
# Technique 6: Orthogonal Random (QR-based)
# ============================================================================
def lsh_orthogonal(embeddings, num_bits=8, seed=12345):
    """
    Use QR-decomposed random matrix for orthogonal hyperplanes

    Guarantees orthogonality within a hash
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Generate random matrix and orthogonalize
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)

    # Use first num_bits rows as hyperplanes
    hyperplanes = Q[:num_bits, :]

    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

# ============================================================================
# Evaluation
# ============================================================================
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
print("LSH TECHNIQUE COMPARISON")
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
    (lsh_sobol, "Sobol Low-Discrepancy"),
    (lsh_hadamard, "Hadamard Transform"),
    (lsh_cross_polytope, "Cross-Polytope"),
    (lsh_orthogonal, "Orthogonal (QR)"),
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

# Find best technique
best_by_coherence = max(results, key=lambda x: x['refined_coherence'])
best_by_improvement = max(results, key=lambda x: x['improvement'])
fastest = min(results, key=lambda x: x['time_ms'])

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"Best refined coherence: {best_by_coherence['name']} ({best_by_coherence['refined_coherence']:.3f})")
print(f"Best improvement:       {best_by_improvement['name']} (+{best_by_improvement['improvement']:.3f})")
print(f"Fastest:                {fastest['name']} ({fastest['time_ms']:.2f}ms)")
print()

# Compare against baseline
baseline = [r for r in results if r['name'] == "Random (baseline)"][0]

print("Gains vs Random baseline:")
print()
for r in results:
    if r['name'] == "Random (baseline)":
        continue

    coherence_gain = r['refined_coherence'] - baseline['refined_coherence']
    improvement_gain = r['improvement'] - baseline['improvement']

    print(f"{r['name']:<25s} coherence: {coherence_gain:+.3f}  improvement: {improvement_gain:+.3f}")

print()

if best_by_coherence['name'] == "Bit Slicing":
    print("✓ BIT SLICING REMAINS THE WINNER")
    print(f"  → {best_by_coherence['refined_coherence']:.3f} refined coherence")
    print(f"  → +{best_by_coherence['improvement']:.3f} improvement from refinement")
elif (best_by_coherence['refined_coherence'] - baseline['refined_coherence']) > 0.02:
    print(f"✓ NEW WINNER: {best_by_coherence['name']}")
    print(f"  → {best_by_coherence['refined_coherence']:.3f} refined coherence")
    print(f"  → {best_by_coherence['refined_coherence'] - baseline['refined_coherence']:+.3f} vs random baseline")
else:
    print("⚠️  All techniques similar")
    print("  → Stick with bit slicing (already validated)")

print()
print("=" * 70)
