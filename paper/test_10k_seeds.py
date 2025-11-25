#!/usr/bin/env python3
"""
Test 10,000 random seeds to find optimal LSH seeds

Goal: Comprehensive search across seed space
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
print("10,000 SEED SEARCH")
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
# Generate 10,000 Test Seeds
# ============================================================================

test_seeds = set()

# 1. Small numbers (0-1000)
test_seeds.update(range(1, 1001))

# 2. Random samples across full 32-bit range
np.random.seed(42)  # Fixed seed for reproducibility
test_seeds.update(np.random.randint(1, 2**31, size=5000).tolist())

# 3. Powers of 2
test_seeds.update([2**i for i in range(1, 31)])

# 4. Primes up to 10,000
def sieve_of_eratosthenes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

primes = sieve_of_eratosthenes(10000)
test_seeds.update(primes)

# 5. Fibonacci numbers
fib = [1, 1]
while fib[-1] < 2**31:
    fib.append(fib[-1] + fib[-2])
test_seeds.update(fib)

# 6. Common seeds
common = [42, 1337, 12345, 54321, 99999, 11111, 22222, 33333, 44444, 55555,
          66666, 77777, 88888]
test_seeds.update(common)

# Convert to sorted list
test_seeds = sorted(list(test_seeds))[:10000]  # Limit to exactly 10k

print(f"Testing {len(test_seeds)} seeds...")
print(f"Range: {min(test_seeds)} to {max(test_seeds)}")
print()

# ============================================================================
# Test All Seeds
# ============================================================================
start_time = time.time()

results = []

for seed_idx, seed in enumerate(test_seeds):
    if (seed_idx + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        rate = (seed_idx + 1) / elapsed
        remaining = (len(test_seeds) - seed_idx - 1) / rate
        print(f"Progress: {seed_idx + 1:>5d}/{len(test_seeds)} seeds "
              f"({elapsed:>6.1f}s elapsed, ~{remaining:>5.1f}s remaining, "
              f"{rate:>5.1f} seeds/s)")

    seed_results = {'seed': seed}

    for dataset_name in all_embeddings:
        embeddings = all_embeddings[dataset_name]
        coherence = evaluate_seed_fast(embeddings, seed)
        seed_results[dataset_name] = coherence

    results.append(seed_results)

total_time = time.time() - start_time

print()
print(f"Completed {len(test_seeds)} seeds in {total_time:.1f}s ({total_time/len(test_seeds):.3f}s per seed)")
print(f"Rate: {len(test_seeds)/total_time:.1f} seeds/second")
print()

# ============================================================================
# Analyze Results
# ============================================================================

# Compute statistics
for r in results:
    scores = [r[ds] for ds in all_embeddings.keys()]
    r['avg_coherence'] = np.mean(scores)
    r['std_coherence'] = np.std(scores)
    r['cv'] = r['std_coherence'] / r['avg_coherence']

# ============================================================================
# TOP 50 SEEDS
# ============================================================================
print("=" * 70)
print("TOP 50 SEEDS BY AVERAGE COHERENCE")
print("=" * 70)
print()

top_50 = sorted(results, key=lambda x: -x['avg_coherence'])[:50]

print(f"{'Rank':<6s} {'Seed':<10s} {'Avg':<10s} {'Std':<10s} {'CV':<10s} ", end="")
for ds in all_embeddings.keys():
    print(f"{ds:<12s} ", end="")
print()
print("-" * 70)

for rank, r in enumerate(top_50, 1):
    print(f"{rank:<6d} {r['seed']:<10d} {r['avg_coherence']:<10.3f} {r['std_coherence']:<10.4f} {r['cv']:<10.4f} ", end="")
    for ds in all_embeddings.keys():
        print(f"{r[ds]:<12.3f} ", end="")
    print()

print()

# ============================================================================
# Most Stable Seeds
# ============================================================================
print("=" * 70)
print("TOP 50 MOST STABLE SEEDS (lowest CV)")
print("=" * 70)
print()

most_stable = sorted(results, key=lambda x: x['cv'])[:50]

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
# Pattern Analysis
# ============================================================================
print("=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)
print()

# Check if top seeds are primes
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

top_10 = sorted(results, key=lambda x: -x['avg_coherence'])[:10]
prime_count = sum(1 for r in top_10 if is_prime(r['seed']))
print(f"Top 10 seeds that are prime: {prime_count}/10")

top_100 = sorted(results, key=lambda x: -x['avg_coherence'])[:100]
prime_count_100 = sum(1 for r in top_100 if is_prime(r['seed']))
print(f"Top 100 seeds that are prime: {prime_count_100}/100")

# Small number bias?
small_count = sum(1 for r in top_10 if r['seed'] < 100)
print(f"Top 10 seeds < 100: {small_count}/10")

small_count_100 = sum(1 for r in top_100 if r['seed'] < 100)
print(f"Top 100 seeds < 100: {small_count_100}/100")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

best_overall = top_50[0]
best_stable = most_stable[0]

print(f"Best overall seed:     {best_overall['seed']} (avg coherence = {best_overall['avg_coherence']:.3f})")
print(f"  Twitter:             {best_overall['Twitter']:.3f}")
print(f"  ArXiv:               {best_overall['ArXiv']:.3f}")
print(f"  Hacker News:         {best_overall['Hacker News']:.3f}")
print()

print(f"Most stable seed:      {best_stable['seed']} (CV = {best_stable['cv']:.4f}, avg = {best_stable['avg_coherence']:.3f})")
print(f"  Twitter:             {best_stable['Twitter']:.3f}")
print(f"  ArXiv:               {best_stable['ArXiv']:.3f}")
print(f"  Hacker News:         {best_stable['Hacker News']:.3f}")
print()

# Compare to common seeds
common_seeds = [12345, 42, 10007, 99999, 31]
print("Common seed rankings:")
for seed in common_seeds:
    if seed in [r['seed'] for r in results]:
        seed_result = [r for r in results if r['seed'] == seed][0]
        rank = sorted(results, key=lambda x: -x['avg_coherence']).index(seed_result) + 1
        print(f"  Seed {seed:>6d}: rank #{rank:>4d} (avg = {seed_result['avg_coherence']:.3f}, CV = {seed_result['cv']:.4f})")

print()
print("=" * 70)
