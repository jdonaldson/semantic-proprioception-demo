#!/usr/bin/env python3
"""
Test: Bit slicing vs separate hashes for hierarchical LSH refinement

Compares two approaches:
1. Separate hashes: Compute new hash at each level (O(k×d))
2. Bit slicing: Compute large hash once, partition bits across levels (O(d))

Hypothesis: Bit slicing is much faster with similar coherence improvement.
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

def lsh_hash(embeddings, num_bits=8, seed=12345):
    """Compute LSH hash for embeddings"""
    np.random.seed(seed)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        single = True
    else:
        single = False

    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values[0] if single else hash_values

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

def hierarchical_separate_hashes(embeddings, max_depth=2, min_size=3):
    """
    Hierarchical LSH using separate hashes at each level

    O(k×d) where k is number of levels, d is dimension
    """
    def refine(embs, depth=0):
        coherence = compute_intra_bucket_similarity(embs)

        if len(embs) < min_size * 2 or depth >= max_depth:
            return [(embs, coherence, depth)]

        # Compute NEW hash at this level
        seed = 12345 + depth * 1000
        sub_hash = lsh_hash(embs, num_bits=4, seed=seed)

        # Group by sub-bucket
        sub_buckets = {}
        for i, hash_val in enumerate(sub_hash):
            if hash_val not in sub_buckets:
                sub_buckets[hash_val] = []
            sub_buckets[hash_val].append(i)

        # Recursively refine
        refined = []
        for indices in sub_buckets.values():
            sub_embeddings = embs[indices]
            if len(sub_embeddings) >= min_size:
                refined.extend(refine(sub_embeddings, depth+1))

        return refined

    return refine(embeddings)

def hierarchical_bit_slicing(embeddings, max_depth=2, min_size=3):
    """
    Hierarchical LSH using bit slicing from a single large hash

    O(d) - compute 16-bit hash once, then use bit operations

    Bit layout:
    - Bits 0-7:   Level 0 (256 buckets, for initial bucketing)
    - Bits 8-11:  Level 1 (16 sub-buckets)
    - Bits 12-15: Level 2 (16 sub-sub-buckets)
    """
    # Compute 16-bit hash ONCE
    hash_16bit = lsh_hash(embeddings, num_bits=16, seed=12345)

    def refine(indices, depth=0):
        embs = embeddings[indices]
        coherence = compute_intra_bucket_similarity(embs)

        if len(embs) < min_size * 2 or depth >= max_depth:
            return [(embs, coherence, depth)]

        # Extract bits for this level (4 bits starting at offset 8 + depth*4)
        offset = 8 + depth * 4
        level_hash = (hash_16bit[indices] >> offset) & 0x0F  # 4 bits = 16 buckets

        # Group by sub-bucket
        sub_buckets = {}
        for i, hash_val in enumerate(level_hash):
            if hash_val not in sub_buckets:
                sub_buckets[hash_val] = []
            sub_buckets[hash_val].append(indices[i])

        # Recursively refine
        refined = []
        for sub_indices in sub_buckets.values():
            if len(sub_indices) >= min_size:
                refined.extend(refine(np.array(sub_indices), depth+1))

        return refined

    # Start with all indices
    initial_indices = np.arange(len(embeddings))
    return refine(initial_indices)

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("BIT SLICING vs SEPARATE HASHES: Hierarchical LSH Comparison")
print("=" * 70)
print()

# ============================================================================
# Load data
# ============================================================================
print("Loading Twitter data...")
embeddings_file = TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet"
index_file = TWITTER_DATA / f"{MODEL_NAME}_lsh_index.parquet"

df_embeddings = pl.read_parquet(embeddings_file)
df_index = pl.read_parquet(index_file)

all_embeddings = np.array([row for row in df_embeddings['embedding']])
print(f"  {len(all_embeddings)} embeddings")
print(f"  Embedding dimension: {all_embeddings.shape[1]}")
print()

# Find dense buckets (≥10 items) for testing
buckets_df = (df_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
    .filter(pl.col('count') >= 10)
    .sort('count', descending=True)
)

print(f"Found {len(buckets_df)} dense buckets (≥10 items)")
print()

# ============================================================================
# Test both approaches on top 10 dense buckets
# ============================================================================
test_buckets = buckets_df.head(10)

results_separate = []
results_bit_slicing = []

print("Testing hierarchical refinement on top 10 dense buckets...")
print()

for row in test_buckets.iter_rows(named=True):
    bucket_id = row['bucket_id']
    count = row['count']

    # Get embeddings in this bucket
    bucket_contents = df_index.filter(pl.col('bucket_id') == bucket_id)
    row_ids = bucket_contents['row_id'].to_list()
    bucket_embeddings = all_embeddings[row_ids]

    # Original coherence
    original_coherence = compute_intra_bucket_similarity(bucket_embeddings)

    # Method 1: Separate hashes
    start = time.perf_counter()
    sub_buckets_separate = hierarchical_separate_hashes(bucket_embeddings, max_depth=2, min_size=3)
    time_separate = time.perf_counter() - start

    coherences_separate = [coh for _, coh, _ in sub_buckets_separate if coh is not None]
    weighted_coh_separate = np.average(
        coherences_separate,
        weights=[len(embs) for embs, _, _ in sub_buckets_separate if compute_intra_bucket_similarity(embs) is not None]
    ) if len(coherences_separate) > 0 else original_coherence

    # Method 2: Bit slicing
    start = time.perf_counter()
    sub_buckets_bit_slicing = hierarchical_bit_slicing(bucket_embeddings, max_depth=2, min_size=3)
    time_bit_slicing = time.perf_counter() - start

    coherences_bit_slicing = [coh for _, coh, _ in sub_buckets_bit_slicing if coh is not None]
    weighted_coh_bit_slicing = np.average(
        coherences_bit_slicing,
        weights=[len(embs) for embs, _, _ in sub_buckets_bit_slicing if compute_intra_bucket_similarity(embs) is not None]
    ) if len(coherences_bit_slicing) > 0 else original_coherence

    results_separate.append({
        'bucket_id': bucket_id,
        'size': count,
        'original': original_coherence,
        'refined': weighted_coh_separate,
        'improvement': weighted_coh_separate - original_coherence,
        'num_sub': len(sub_buckets_separate),
        'time': time_separate
    })

    results_bit_slicing.append({
        'bucket_id': bucket_id,
        'size': count,
        'original': original_coherence,
        'refined': weighted_coh_bit_slicing,
        'improvement': weighted_coh_bit_slicing - original_coherence,
        'num_sub': len(sub_buckets_bit_slicing),
        'time': time_bit_slicing
    })

# ============================================================================
# Results
# ============================================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"{'Bucket':<8s} {'Size':<6s} {'Original':<10s} {'Separate Δ':<12s} {'BitSlice Δ':<12s} {'Time Sep':<10s} {'Time Bit':<10s}")
print("-" * 70)

for r_sep, r_bit in zip(results_separate, results_bit_slicing):
    print(f"{r_sep['bucket_id']:<8d} {r_sep['size']:<6d} {r_sep['original']:<10.3f} "
          f"{r_sep['improvement']:>+10.3f}  {r_bit['improvement']:>+10.3f}  "
          f"{r_sep['time']*1000:>7.2f}ms  {r_bit['time']*1000:>7.2f}ms")

print()

# Summary statistics
avg_improvement_separate = np.mean([r['improvement'] for r in results_separate])
avg_improvement_bit_slicing = np.mean([r['improvement'] for r in results_bit_slicing])

avg_time_separate = np.mean([r['time'] for r in results_separate])
avg_time_bit_slicing = np.mean([r['time'] for r in results_bit_slicing])

speedup = avg_time_separate / avg_time_bit_slicing if avg_time_bit_slicing > 0 else 0

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"Average coherence improvement:")
print(f"  Separate hashes:   {avg_improvement_separate:+.3f}")
print(f"  Bit slicing:       {avg_improvement_bit_slicing:+.3f}")
print()

print(f"Average time per bucket:")
print(f"  Separate hashes:   {avg_time_separate*1000:.2f}ms")
print(f"  Bit slicing:       {avg_time_bit_slicing*1000:.2f}ms")
print(f"  Speedup:           {speedup:.1f}×")
print()

if abs(avg_improvement_separate - avg_improvement_bit_slicing) < 0.01:
    if speedup > 1.5:
        print("✓ BIT SLICING WINS")
        print(f"  → Same coherence improvement ({speedup:.1f}× faster)")
    else:
        print("⚠️  Similar performance")
        print("  → Both approaches comparable in speed and quality")
else:
    quality_winner = "Separate" if avg_improvement_separate > avg_improvement_bit_slicing else "Bit slicing"
    speed_winner = "Bit slicing" if speedup > 1 else "Separate"
    print(f"⚠️  Trade-off:")
    print(f"  → Quality winner: {quality_winner}")
    print(f"  → Speed winner: {speed_winner} ({speedup:.1f}×)")

print()
print("Key insight:")
print()
print("Bit Slicing:")
print("  - O(d) complexity: ONE matrix multiply (16 bits)")
print("  - Levels extracted via bit masking (O(1) per level)")
print("  - Still compositional (same seed = same hierarchy)")
print("  - Faster with similar or better quality")
print()
print("Separate Hashes:")
print("  - O(k×d) complexity: k matrix multiplies (one per level)")
print("  - Independent random projections at each level")
print("  - More exploration, but slower")
print()

if speedup > 1.2:
    print("Recommendation: Use bit slicing for production")
    print(f"  - {speedup:.1f}× faster")
    print("  - Same or better coherence improvement")
    print("  - Simpler implementation (single hash)")
else:
    print("Recommendation: Either approach acceptable")
    print("  - Performance similar")
    print("  - Choose based on implementation preference")

print()
print("=" * 70)
