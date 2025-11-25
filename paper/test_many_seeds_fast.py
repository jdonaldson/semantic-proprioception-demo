#!/usr/bin/env python3
"""
Fast test of many random seeds

Goal: Test 50-100 seeds in ~5 minutes to find optimal seeds
Strategy: Test only bit slicing (faster) on all 3 datasets
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

def lsh_bit_slicing(embeddings, num_bits=16, seed=12345):
    """Bit slicing"""
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

def evaluate_seed_fast(embeddings, seed):
    """Fast evaluation - bit slicing only"""
    hash_16bit = lsh_bit_slicing(embeddings, num_bits=16, seed=seed)
    hash_8bit = hash_16bit & 0xFF

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

        level_1_hash = (hash_16bit[indices] >> 8) & 0x0F

        sub_buckets = {}
        for i, sub_hash_val in enumerate(level_1_hash):
            if sub_hash_val not in sub_buckets:
                sub_buckets[sub_hash_val] = []
            sub_buckets[sub_hash_val].append(indices[i])

        for sub_indices in sub_buckets.values():
            if len(sub_indices) >= 2:
                sub_embeddings = embeddings[sub_indices]
                sub_coh = compute_intra_bucket_similarity(sub_embeddings)
                if sub_coh is not None:
                    refined_coherences.append(sub_coh)

    refined_coherence = np.mean(refined_coherences) if len(refined_coherences) > 0 else avg_coherence

    return refined_coherence

# ============================================================================
# Load Datasets
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")

datasets = [
    ("Twitter", PROJECT_ROOT / "semantic_proprioception_data"),
    ("ArXiv", PROJECT_ROOT / "arxiv_demo_data"),
    ("Hacker News", PROJECT_ROOT / "hackernews_demo_data"),
]

print("=" * 70)
print("FAST SEED SEARCH")
print("=" * 70)
print()

# Load all embeddings upfront
all_embeddings = {}
for dataset_name, data_dir in datasets:
    embeddings_file = data_dir / "MiniLM-L6_embeddings.parquet"
    if embeddings_file.exists():
        df = pl.read_parquet(embeddings_file)
        all_embeddings[dataset_name] = np.array([row for row in df['embedding']])
        print(f"Loaded {dataset_name}: {len(all_embeddings[dataset_name])} embeddings")

print()

# ============================================================================
# Generate Test Seeds
# ============================================================================

# Mix of different seed types
test_seeds = []

# Primes (good for avoiding patterns)
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
          73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
          151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
          1009, 2003, 3001, 4001, 5003, 6007, 7001, 8009, 9001,
          10007, 20011, 30011, 40009, 50021, 60013, 70001, 80021, 90001, 99991]

# Powers of 2 (common in computing)
powers_2 = [2**i for i in range(1, 17)]  # 2, 4, 8, ..., 65536

# Fibonacci numbers
fib = [1, 1]
while fib[-1] < 100000:
    fib.append(fib[-1] + fib[-2])

# Mersenne primes
mersenne = [3, 7, 31, 127, 8191, 131071]

# Common seeds
common = [42, 1337, 12345, 54321, 99999, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888]

# Mathematical constants
constants = [
    314,    # π * 100
    271,    # e * 100
    161,    # φ * 100
    31415,  # π * 10000
    27182,  # e * 10000
    16180,  # φ * 10000
]

# Combine and deduplicate
test_seeds = list(set(primes + powers_2 + fib + mersenne + common + constants))
test_seeds.sort()

print(f"Testing {len(test_seeds)} seeds...")
print()

# ============================================================================
# Test All Seeds
# ============================================================================
start_time = time.time()

results = []

for seed_idx, seed in enumerate(test_seeds):
    if (seed_idx + 1) % 20 == 0:
        elapsed = time.time() - start_time
        rate = (seed_idx + 1) / elapsed
        remaining = (len(test_seeds) - seed_idx - 1) / rate
        print(f"Progress: {seed_idx + 1}/{len(test_seeds)} seeds ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    seed_results = {'seed': seed}

    for dataset_name in all_embeddings:
        embeddings = all_embeddings[dataset_name]
        coherence = evaluate_seed_fast(embeddings, seed)
        seed_results[dataset_name] = coherence

    # Compute average rank
    results.append(seed_results)

total_time = time.time() - start_time

print()
print(f"Completed {len(test_seeds)} seeds in {total_time:.1f}s ({total_time/len(test_seeds):.2f}s per seed)")
print()

# ============================================================================
# Find Best Seeds
# ============================================================================
print("=" * 70)
print("TOP 20 SEEDS BY AVERAGE COHERENCE")
print("=" * 70)
print()

# Compute average coherence
for r in results:
    scores = [r[ds] for ds in all_embeddings.keys()]
    r['avg_coherence'] = np.mean(scores)
    r['std_coherence'] = np.std(scores)
    r['cv'] = r['std_coherence'] / r['avg_coherence']

# Sort by average coherence
top_20 = sorted(results, key=lambda x: -x['avg_coherence'])[:20]

print(f"{'Rank':<6s} {'Seed':<10s} {'Avg':<10s} {'Std':<10s} {'CV':<10s} ", end="")
for ds in all_embeddings.keys():
    print(f"{ds:<12s} ", end="")
print()
print("-" * 70)

for rank, r in enumerate(top_20, 1):
    print(f"{rank:<6d} {r['seed']:<10d} {r['avg_coherence']:<10.3f} {r['std_coherence']:<10.4f} {r['cv']:<10.4f} ", end="")
    for ds in all_embeddings.keys():
        print(f"{r[ds]:<12.3f} ", end="")
    print()

print()

# ============================================================================
# Most Stable Seeds
# ============================================================================
print("=" * 70)
print("TOP 20 MOST STABLE SEEDS (lowest CV)")
print("=" * 70)
print()

most_stable = sorted(results, key=lambda x: x['cv'])[:20]

print(f"{'Rank':<6s} {'Seed':<10s} {'CV':<10s} {'Avg':<10s} {'Std':<10s} ", end="")
for ds in all_embeddings.keys():
    print(f"{ds:<12s} ", end="")
print()
print("-" * 70)

for rank, r in enumerate(most_stable, 1):
    print(f"{rank:<6d} {r['seed']:<10d} {r['cv']:<10.4f} {r['avg_coherence']:<10.3f} {r['std_coherence']:<10.4f} ", end="")
    for ds in all_embeddings.keys():
        print(f"{r[ds]:<12.3f} ", end="")
    print()

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

best_overall = top_20[0]
best_stable = most_stable[0]

print(f"Best overall seed:     {best_overall['seed']} (avg coherence = {best_overall['avg_coherence']:.3f})")
print(f"Most stable seed:      {best_stable['seed']} (CV = {best_stable['cv']:.4f}, avg = {best_stable['avg_coherence']:.3f})")
print()

# Check if 10007 is in top 10
seed_10007 = [r for r in results if r['seed'] == 10007][0]
rank_10007 = sorted(results, key=lambda x: -x['avg_coherence']).index(seed_10007) + 1
print(f"Our recommended seed (10007):  rank #{rank_10007} (avg = {seed_10007['avg_coherence']:.3f}, CV = {seed_10007['cv']:.4f})")
print()

print("=" * 70)
