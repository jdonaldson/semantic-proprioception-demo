#!/usr/bin/env python3
"""
Test seed generalization across datasets

Question: Do optimal seeds on Twitter also work well on ArXiv and Hacker News?
- If yes: We can recommend universal seeds
- If no: Seed selection is dataset-specific

We'll test:
1. Best/worst seeds from Twitter on other datasets
2. Correlation of seed rankings across datasets
3. Whether bit slicing's stability advantage holds
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

    return refined_coherence

# ============================================================================
# Main Experiment
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")

# Datasets
datasets = [
    ("Twitter", PROJECT_ROOT / "semantic_proprioception_data"),
    ("ArXiv", PROJECT_ROOT / "arxiv_demo_data"),
    ("Hacker News", PROJECT_ROOT / "hackernews_demo_data"),
]

# Test seeds (from Twitter stability analysis)
test_seeds = [
    12345,   # Standard seed (worst for bit slicing on Twitter)
    10007,   # Best for bit slicing on Twitter
    99999,   # Best for separate hashes on Twitter
    42,      # Common seed
    31415,   # Worst for separate hashes on Twitter
    7,       # Lucky number
]

print("=" * 70)
print("SEED GENERALIZATION ACROSS DATASETS")
print("=" * 70)
print()

# Store all results
all_results = {
    'bit_slicing': {},
    'separate': {}
}

for dataset_name, data_dir in datasets:
    print(f"Dataset: {dataset_name}")
    print("-" * 70)

    # Load embeddings
    embeddings_file = data_dir / "MiniLM-L6_embeddings.parquet"

    if not embeddings_file.exists():
        print(f"  ⚠️  Embeddings not found: {embeddings_file}")
        print()
        continue

    df_embeddings = pl.read_parquet(embeddings_file)
    all_embeddings = np.array([row for row in df_embeddings['embedding']])

    print(f"  {len(all_embeddings)} embeddings (dim={all_embeddings.shape[1]})")
    print()

    # Test bit slicing
    bit_slicing_results = []
    for seed in test_seeds:
        coherence = evaluate_seed(all_embeddings, seed, use_bit_slicing=True)
        bit_slicing_results.append((seed, coherence))

    all_results['bit_slicing'][dataset_name] = bit_slicing_results

    # Test separate hashes
    separate_results = []
    for seed in test_seeds:
        coherence = evaluate_seed(all_embeddings, seed, use_bit_slicing=False)
        separate_results.append((seed, coherence))

    all_results['separate'][dataset_name] = separate_results

    # Display results
    print("  Bit Slicing:")
    for seed, coh in sorted(bit_slicing_results, key=lambda x: -x[1]):
        print(f"    Seed {seed:>6d}: {coh:.3f}")

    print()
    print("  Separate Hashes:")
    for seed, coh in sorted(separate_results, key=lambda x: -x[1]):
        print(f"    Seed {seed:>6d}: {coh:.3f}")

    print()

# ============================================================================
# Cross-Dataset Analysis
# ============================================================================
print("=" * 70)
print("CROSS-DATASET ANALYSIS")
print("=" * 70)
print()

# Create rankings for each dataset
def get_rankings(results):
    """Convert coherence scores to rankings (1=best, n=worst)"""
    sorted_results = sorted(results, key=lambda x: -x[1])
    rankings = {}
    for rank, (seed, _) in enumerate(sorted_results, 1):
        rankings[seed] = rank
    return rankings

print("BIT SLICING - Seed Rankings Across Datasets:")
print(f"{'Seed':<10s} ", end="")
for dataset_name in ["Twitter", "ArXiv", "Hacker News"]:
    if dataset_name in all_results['bit_slicing']:
        print(f"{dataset_name:<15s} ", end="")
print()
print("-" * 70)

for seed in test_seeds:
    print(f"{seed:<10d} ", end="")
    for dataset_name in ["Twitter", "ArXiv", "Hacker News"]:
        if dataset_name in all_results['bit_slicing']:
            results = all_results['bit_slicing'][dataset_name]
            rankings = get_rankings(results)
            rank = rankings[seed]
            coherence = dict(results)[seed]
            print(f"#{rank} ({coherence:.3f})     ", end="")
    print()

print()

print("SEPARATE HASHES - Seed Rankings Across Datasets:")
print(f"{'Seed':<10s} ", end="")
for dataset_name in ["Twitter", "ArXiv", "Hacker News"]:
    if dataset_name in all_results['separate']:
        print(f"{dataset_name:<15s} ", end="")
print()
print("-" * 70)

for seed in test_seeds:
    print(f"{seed:<10d} ", end="")
    for dataset_name in ["Twitter", "ArXiv", "Hacker News"]:
        if dataset_name in all_results['separate']:
            results = all_results['separate'][dataset_name]
            rankings = get_rankings(results)
            rank = rankings[seed]
            coherence = dict(results)[seed]
            print(f"#{rank} ({coherence:.3f})     ", end="")
    print()

print()

# ============================================================================
# Find Universal Best Seeds
# ============================================================================
print("=" * 70)
print("UNIVERSAL BEST SEEDS")
print("=" * 70)
print()

# Average rankings across datasets
def compute_avg_rank(seed, results_dict):
    """Compute average rank across all datasets"""
    ranks = []
    for dataset_results in results_dict.values():
        rankings = get_rankings(dataset_results)
        ranks.append(rankings[seed])
    return np.mean(ranks)

print("Bit Slicing - Average Rank (lower is better):")
bit_avg_ranks = [(seed, compute_avg_rank(seed, all_results['bit_slicing']))
                 for seed in test_seeds]
for seed, avg_rank in sorted(bit_avg_ranks, key=lambda x: x[1]):
    print(f"  Seed {seed:>6d}: avg rank = {avg_rank:.2f}")

print()

print("Separate Hashes - Average Rank (lower is better):")
sep_avg_ranks = [(seed, compute_avg_rank(seed, all_results['separate']))
                 for seed in test_seeds]
for seed, avg_rank in sorted(sep_avg_ranks, key=lambda x: x[1]):
    print(f"  Seed {seed:>6d}: avg rank = {avg_rank:.2f}")

print()

# ============================================================================
# Stability Comparison
# ============================================================================
print("=" * 70)
print("STABILITY ACROSS DATASETS")
print("=" * 70)
print()

print("Coefficient of Variation (CV = std/mean) for each seed:")
print(f"{'Seed':<10s} {'Bit Slicing CV':<20s} {'Separate Hash CV':<20s}")
print("-" * 70)

for seed in test_seeds:
    # Get coherence scores across datasets
    bit_scores = [dict(all_results['bit_slicing'][ds])[seed]
                  for ds in all_results['bit_slicing']]
    sep_scores = [dict(all_results['separate'][ds])[seed]
                  for ds in all_results['separate']]

    bit_cv = np.std(bit_scores) / np.mean(bit_scores) if np.mean(bit_scores) > 0 else 0
    sep_cv = np.std(sep_scores) / np.mean(sep_scores) if np.mean(sep_scores) > 0 else 0

    print(f"{seed:<10d} {bit_cv:<20.4f} {sep_cv:<20.4f}")

print()

# ============================================================================
# Verdict
# ============================================================================
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()

best_bit_seed = sorted(bit_avg_ranks, key=lambda x: x[1])[0][0]
best_sep_seed = sorted(sep_avg_ranks, key=lambda x: x[1])[0][0]

print(f"Best universal seed for bit slicing:   {best_bit_seed}")
print(f"Best universal seed for separate hash: {best_sep_seed}")
print()

# Check if Twitter's best seeds generalize
twitter_bit_best = 10007
twitter_sep_best = 99999

twitter_bit_rank = compute_avg_rank(twitter_bit_best, all_results['bit_slicing'])
twitter_sep_rank = compute_avg_rank(twitter_sep_best, all_results['separate'])

print("Twitter's best seeds generalization:")
print(f"  Seed {twitter_bit_best} (Twitter best for bit slicing):   avg rank = {twitter_bit_rank:.2f}")
print(f"  Seed {twitter_sep_best} (Twitter best for separate hash): avg rank = {twitter_sep_rank:.2f}")
print()

if best_bit_seed == twitter_bit_best:
    print("✓ TWITTER'S BEST BIT SLICING SEED GENERALIZES")
else:
    print("✗ TWITTER'S BEST BIT SLICING SEED DOESN'T GENERALIZE")
    print(f"  Universal best is seed {best_bit_seed}")

print()

if best_sep_seed == twitter_sep_best:
    print("✓ TWITTER'S BEST SEPARATE HASH SEED GENERALIZES")
else:
    print("✗ TWITTER'S BEST SEPARATE HASH SEED DOESN'T GENERALIZE")
    print(f"  Universal best is seed {best_sep_seed}")

print()
print("=" * 70)
