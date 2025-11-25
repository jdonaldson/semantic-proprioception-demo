#!/usr/bin/env python3
"""
Inspect actual text content in buckets before/after bit-slicing refinement

Shows concrete examples of how coherence improves
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm

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

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("BUCKET COHERENCE INSPECTION: Concrete Examples")
print("=" * 70)
print()

# Load data
embeddings_file = TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet"
index_file = TWITTER_DATA / f"{MODEL_NAME}_lsh_index.parquet"

df_embeddings = pl.read_parquet(embeddings_file)
df_index = pl.read_parquet(index_file)

all_embeddings = np.array([row for row in df_embeddings['embedding']])

# Load original text data
text_file = PROJECT_ROOT / "semantic_proprioception_data" / "twitter_customer_support.csv"
df_text = pl.read_csv(text_file)

print(f"Loaded {len(all_embeddings)} embeddings")
print()

# Find dense buckets
buckets_df = (df_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
    .filter(pl.col('count') >= 10)
    .sort('count', descending=True)
)

# Inspect specific buckets with interesting results
# Bucket 196: bit slicing +0.384 vs separate +0.099 (huge difference)
# Bucket 4: bit slicing +0.542 vs separate +0.186 (3x better)
# Bucket 198: bit slicing +0.275 vs separate -0.048 (separate failed!)

interesting_buckets = [196, 4, 198]

for bucket_id in interesting_buckets:
    print("=" * 70)
    print(f"BUCKET {bucket_id}")
    print("=" * 70)
    print()

    # Get bucket contents
    bucket_contents = df_index.filter(pl.col('bucket_id') == bucket_id)
    row_ids = bucket_contents['row_id'].to_list()
    bucket_embeddings = all_embeddings[row_ids]
    bucket_texts = [df_text[i]['text'] for i in row_ids]

    # Original coherence
    original_coherence = compute_intra_bucket_similarity(bucket_embeddings)

    print(f"Size: {len(bucket_texts)} items")
    print(f"Original coherence: {original_coherence:.3f}")
    print()

    # Show sample texts from original bucket
    print("Sample texts from ORIGINAL bucket:")
    print("-" * 70)
    for i, text in enumerate(bucket_texts[:5], 1):
        print(f"{i}. {text[:80]}...")
    print()

    # Apply bit slicing
    hash_16bit = lsh_hash(bucket_embeddings, num_bits=16, seed=12345)

    # Level 1: bits 8-11 (16 sub-buckets)
    level_1_hash = (hash_16bit >> 8) & 0x0F

    # Group by sub-bucket
    sub_buckets = {}
    for i, hash_val in enumerate(level_1_hash):
        if hash_val not in sub_buckets:
            sub_buckets[hash_val] = []
        sub_buckets[hash_val].append(i)

    # Compute coherence for each sub-bucket
    sub_bucket_stats = []
    for sub_id, indices in sub_buckets.items():
        if len(indices) >= 2:
            sub_embeddings = bucket_embeddings[indices]
            sub_coherence = compute_intra_bucket_similarity(sub_embeddings)
            sub_texts = [bucket_texts[i] for i in indices]
            sub_bucket_stats.append({
                'sub_id': sub_id,
                'size': len(indices),
                'coherence': sub_coherence,
                'texts': sub_texts
            })

    # Sort by size
    sub_bucket_stats.sort(key=lambda x: x['size'], reverse=True)

    # Weighted average coherence after refinement
    if len(sub_bucket_stats) > 0:
        weighted_coherence = np.average(
            [s['coherence'] for s in sub_bucket_stats],
            weights=[s['size'] for s in sub_bucket_stats]
        )
    else:
        weighted_coherence = original_coherence

    print(f"After bit-slicing (Level 1):")
    print(f"  Created {len(sub_bucket_stats)} sub-buckets")
    print(f"  Weighted coherence: {weighted_coherence:.3f}")
    print(f"  Improvement: {weighted_coherence - original_coherence:+.3f}")
    print()

    # Show largest sub-buckets
    print("Largest SUB-BUCKETS:")
    print("-" * 70)
    for i, stats in enumerate(sub_bucket_stats[:3], 1):
        print(f"\nSub-bucket {stats['sub_id']} (size={stats['size']}, coherence={stats['coherence']:.3f}):")
        for j, text in enumerate(stats['texts'][:3], 1):
            print(f"  {j}. {text[:70]}...")

    print()
    print()

print("=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print()
print("Bit slicing improves coherence by creating semantically tighter sub-buckets:")
print()
print("1. Original bucket: Mixed themes (low coherence)")
print("   → Bit slicing separates into distinct sub-topics")
print()
print("2. Each sub-bucket: More homogeneous content (high coherence)")
print("   → Items within sub-buckets are semantically closer")
print()
print("3. Weighted average: Higher overall coherence")
print("   → Most items end up in focused sub-buckets")
print()
