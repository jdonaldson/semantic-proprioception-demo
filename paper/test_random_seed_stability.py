#!/usr/bin/env python3
"""
Test random seed stability for LSH

Question: Does seed choice matter for bit slicing performance?
- If yes: We could optimize seed selection
- If no: Bit slicing's advantage is structural, not lucky randomness

We'll test:
1. Multiple random seeds with bit slicing
2. Multiple random seeds with separate hashes (baseline)
3. Variance analysis
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

def lsh_bit_slicing(embeddings, num_bits=16, seed=12345):
    """Bit slicing (current winner)"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

def lsh_random(embeddings, num_bits=8, seed=12345):
    """Random hyperplanes"""
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

def evaluate_seed(embeddings, seed, use_bit_slicing=True):
    """Evaluate single seed"""
    # Compute hash
    if use_bit_slicing:
        hash_16bit = lsh_bit_slicing(embeddings, num_bits=16, seed=seed)
        hash_8bit = hash_16bit & 0xFF
    else:
        hash_8bit = lsh_random(embeddings, num_bits=8, seed=seed)
        hash_16bit = lsh_random(embeddings, num_bits=16, seed=seed)

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

    # Hierarchical refinement
    refined_coherences = []
    for indices in dense_buckets.values():
        if len(indices) < 6:
            continue

        # Extract level 1 hash
        if use_bit_slicing:
            level_1_hash = (hash_16bit[indices] >> 8) & 0x0F
        else:
            # Separate hash with different seed
            sub_embeddings = embeddings[indices]
            sub_hash = lsh_random(sub_embeddings, num_bits=4, seed=seed+1000)
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
        'seed': seed,
        'num_buckets': len(buckets),
        'num_dense': len(dense_buckets),
        'avg_coherence': avg_coherence,
        'refined_coherence': refined_coherence,
        'improvement': refined_coherence - avg_coherence,
    }

# ============================================================================
# Main Experiment
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("RANDOM SEED STABILITY TEST")
print("=" * 70)
print()

# Load Twitter data
embeddings_file = TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet"
df_embeddings = pl.read_parquet(embeddings_file)
all_embeddings = np.array([row for row in df_embeddings['embedding']])

print(f"Loaded {len(all_embeddings)} embeddings (dim={all_embeddings.shape[1]})")
print()

# Test multiple seeds
test_seeds = [
    12345,   # Our standard seed
    42,      # Common test seed
    2024,    # Year-based
    99999,   # Large number
    7,       # Lucky number
    54321,   # Reverse of standard
    31415,   # Pi approximation
    27182,   # e approximation
    10007,   # Prime
    65536,   # Power of 2
]

print(f"Testing {len(test_seeds)} different random seeds...")
print()

# Test bit slicing
print("BIT SLICING:")
print("-" * 70)
bit_slicing_results = []

for seed in test_seeds:
    result = evaluate_seed(all_embeddings, seed, use_bit_slicing=True)
    bit_slicing_results.append(result)
    print(f"  Seed {seed:>6d}: initial {result['avg_coherence']:.3f} → refined {result['refined_coherence']:.3f} (+{result['improvement']:.3f})")

print()

# Test separate hashes
print("SEPARATE HASHES (baseline):")
print("-" * 70)
separate_results = []

for seed in test_seeds:
    result = evaluate_seed(all_embeddings, seed, use_bit_slicing=False)
    separate_results.append(result)
    print(f"  Seed {seed:>6d}: initial {result['avg_coherence']:.3f} → refined {result['refined_coherence']:.3f} (+{result['improvement']:.3f})")

print()

# ============================================================================
# Statistical Analysis
# ============================================================================
print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)
print()

bit_slice_coherences = [r['refined_coherence'] for r in bit_slicing_results]
separate_coherences = [r['refined_coherence'] for r in separate_results]

bit_slice_improvements = [r['improvement'] for r in bit_slicing_results]
separate_improvements = [r['improvement'] for r in separate_results]

print("Refined Coherence:")
print(f"  Bit Slicing:    mean={np.mean(bit_slice_coherences):.3f}  std={np.std(bit_slice_coherences):.4f}  "
      f"range=[{np.min(bit_slice_coherences):.3f}, {np.max(bit_slice_coherences):.3f}]")
print(f"  Separate Hash:  mean={np.mean(separate_coherences):.3f}  std={np.std(separate_coherences):.4f}  "
      f"range=[{np.min(separate_coherences):.3f}, {np.max(separate_coherences):.3f}]")
print()

print("Improvement from Refinement:")
print(f"  Bit Slicing:    mean={np.mean(bit_slice_improvements):.3f}  std={np.std(bit_slice_improvements):.4f}  "
      f"range=[{np.min(bit_slice_improvements):.3f}, {np.max(bit_slice_improvements):.3f}]")
print(f"  Separate Hash:  mean={np.mean(separate_improvements):.3f}  std={np.std(separate_improvements):.4f}  "
      f"range=[{np.min(separate_improvements):.3f}, {np.max(separate_improvements):.3f}]")
print()

# Find best and worst seeds
best_bit_slice = max(bit_slicing_results, key=lambda x: x['refined_coherence'])
worst_bit_slice = min(bit_slicing_results, key=lambda x: x['refined_coherence'])

best_separate = max(separate_results, key=lambda x: x['refined_coherence'])
worst_separate = min(separate_results, key=lambda x: x['refined_coherence'])

print("Best Seeds:")
print(f"  Bit Slicing:   seed={best_bit_slice['seed']:>6d}  coherence={best_bit_slice['refined_coherence']:.3f}")
print(f"  Separate Hash: seed={best_separate['seed']:>6d}  coherence={best_separate['refined_coherence']:.3f}")
print()

print("Worst Seeds:")
print(f"  Bit Slicing:   seed={worst_bit_slice['seed']:>6d}  coherence={worst_bit_slice['refined_coherence']:.3f}")
print(f"  Separate Hash: seed={worst_separate['seed']:>6d}  coherence={worst_separate['refined_coherence']:.3f}")
print()

# Coefficient of variation (std/mean)
cv_bit_slice = np.std(bit_slice_coherences) / np.mean(bit_slice_coherences)
cv_separate = np.std(separate_coherences) / np.mean(separate_coherences)

print("Stability (Coefficient of Variation = std/mean, lower is more stable):")
print(f"  Bit Slicing:   CV={cv_bit_slice:.4f}")
print(f"  Separate Hash: CV={cv_separate:.4f}")
print()

# ============================================================================
# Verdict
# ============================================================================
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()

mean_diff = np.mean(bit_slice_coherences) - np.mean(separate_coherences)
best_diff = best_bit_slice['refined_coherence'] - best_separate['refined_coherence']

print(f"Average advantage: Bit slicing {mean_diff:+.3f} better than separate hashes")
print(f"Best case:         Bit slicing {best_diff:+.3f} better at best seeds")
print()

if np.std(bit_slice_coherences) < 0.005:
    print("✓ BIT SLICING IS HIGHLY STABLE")
    print(f"  → Std dev = {np.std(bit_slice_coherences):.4f} (very low variance)")
    print("  → Seed choice doesn't matter much")
    print("  → Advantage is structural, not random luck")
elif np.std(bit_slice_coherences) > 0.02:
    print("⚠️  BIT SLICING VARIES BY SEED")
    print(f"  → Std dev = {np.std(bit_slice_coherences):.4f} (high variance)")
    print("  → Seed optimization could help")
    print(f"  → Use seed {best_bit_slice['seed']} for best results")
else:
    print("→ BIT SLICING HAS MODERATE VARIANCE")
    print(f"  → Std dev = {np.std(bit_slice_coherences):.4f}")
    print("  → Some seed variation exists but advantage holds")

print()

# Check if best separate hash beats worst bit slicing
if best_separate['refined_coherence'] > worst_bit_slice['refined_coherence']:
    print("⚠️  CROSSOVER DETECTED")
    print("  → Best separate hash seed beats worst bit slicing seed")
    print("  → Seed selection matters!")
else:
    print("✓ NO CROSSOVER")
    print("  → Even worst bit slicing seed beats best separate hash seed")
    print("  → Bit slicing dominates regardless of seed choice")

print()
print("=" * 70)
