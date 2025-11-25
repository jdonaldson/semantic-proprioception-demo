#!/usr/bin/env python3
"""
Test top seeds across different datasets

Question: Do optimal seeds generalize across different text types?

Datasets to test:
- Twitter (short, informal social media)
- ArXiv (long, formal academic papers)
- HackerNews (medium, technical discussions)
- Amazon (consumer product reviews)
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm

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
# Load All Datasets
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
DATA_DIR = PROJECT_ROOT / "semantic_proprioception_data"
DEMO_DIR = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo")

datasets = {
    "Twitter": DEMO_DIR / "semantic_proprioception_data" / "MiniLM-L6_embeddings.parquet",
    "ArXiv": DEMO_DIR / "arxiv_demo_data" / "MiniLM-L6_embeddings.parquet",
    "HackerNews": DEMO_DIR / "hackernews_demo_data" / "MiniLM-L6_embeddings.parquet",
    "Amazon": DEMO_DIR / "amazon_demo_data" / "MiniLM-L6_embeddings.parquet",
}

print("=" * 70)
print("CROSS-DATASET SEED GENERALIZATION TEST")
print("=" * 70)
print()

# Load all dataset embeddings
all_embeddings = {}
for dataset_name, embeddings_file in datasets.items():
    if embeddings_file.exists():
        df = pl.read_parquet(embeddings_file)
        embeddings = np.array([row for row in df['embedding']])
        all_embeddings[dataset_name] = embeddings
        print(f"Loaded {dataset_name}: {len(embeddings)} embeddings, dim={embeddings.shape[1]}")
    else:
        print(f"⚠️  {dataset_name} not found: {embeddings_file}")

print()

# ============================================================================
# Seeds to Test
# ============================================================================

test_seeds = [
    ("Best overall", 31),              # Winner for MiniLM-L6
    ("Top from 10k", 4751),            # 2nd best
    ("Most stable", 1056240716),       # Highest stability across models
    ("Previous best", 10007),          # Our earlier recommendation
    ("Common good", 99999),            # Best common seed
    ("Common bad", 12345),             # Standard (terrible)
    ("Worst common", 42),              # Popular (worst)
]

# ============================================================================
# Test All Seeds Across All Datasets
# ============================================================================

results = []

print("Testing seeds across datasets...")
print()

for dataset_name in all_embeddings:
    embeddings = all_embeddings[dataset_name]

    for seed_label, seed in test_seeds:
        coherence = evaluate_seed_fast(embeddings, seed)
        results.append({
            'dataset': dataset_name,
            'seed_label': seed_label,
            'seed': seed,
            'coherence': coherence
        })

print("Completed!")
print()

# ============================================================================
# Results by Dataset
# ============================================================================

for dataset_name in all_embeddings:
    print("=" * 70)
    print(f"DATASET: {dataset_name} ({len(all_embeddings[dataset_name])} samples)")
    print("=" * 70)
    print()

    dataset_results = [r for r in results if r['dataset'] == dataset_name]
    dataset_results_sorted = sorted(dataset_results, key=lambda x: -x['coherence'])

    print(f"{'Rank':<6s} {'Seed':<30s} {'Coherence':<12s}")
    print("-" * 70)

    for rank, r in enumerate(dataset_results_sorted, 1):
        print(f"{rank:<6d} {r['seed_label']:<30s} {r['coherence']:<12.3f}")

    print()

# ============================================================================
# Cross-Dataset Analysis
# ============================================================================

print("=" * 70)
print("CROSS-DATASET SEED RANKINGS")
print("=" * 70)
print()

print(f"{'Seed':<30s} ", end="")
for dataset_name in all_embeddings:
    print(f"{dataset_name:<15s} ", end="")
print("Avg")
print("-" * 70)

for seed_label, seed in test_seeds:
    print(f"{seed_label:<30s} ", end="")

    seed_results = [r for r in results if r['seed'] == seed]
    scores = []

    for dataset_name in all_embeddings:
        dataset_result = [r for r in seed_results if r['dataset'] == dataset_name][0]
        print(f"{dataset_result['coherence']:<15.3f} ", end="")
        scores.append(dataset_result['coherence'])

    avg = np.mean(scores)
    print(f"{avg:.3f}")

print()

# ============================================================================
# Best Seed by Dataset
# ============================================================================

print("=" * 70)
print("BEST SEED FOR EACH DATASET")
print("=" * 70)
print()

for dataset_name in all_embeddings:
    dataset_results = [r for r in results if r['dataset'] == dataset_name]
    best = max(dataset_results, key=lambda x: x['coherence'])
    print(f"{dataset_name:<15s} Best: {best['seed_label']:<30s} ({best['coherence']:.3f})")

print()

# ============================================================================
# Seed Stability Across Datasets
# ============================================================================

print("=" * 70)
print("SEED STABILITY ACROSS DATASETS")
print("=" * 70)
print()

print(f"{'Seed':<30s} {'Mean':<10s} {'Std':<10s} {'CV':<10s} {'Range':<15s}")
print("-" * 70)

seed_stats = []
for seed_label, seed in test_seeds:
    seed_results = [r for r in results if r['seed'] == seed]
    scores = [r['coherence'] for r in seed_results]

    mean = np.mean(scores)
    std = np.std(scores)
    cv = std / mean if mean > 0 else 0
    range_val = f"[{min(scores):.3f}, {max(scores):.3f}]"

    print(f"{seed_label:<30s} {mean:<10.3f} {std:<10.4f} {cv:<10.4f} {range_val:<15s}")

    seed_stats.append({
        'label': seed_label,
        'seed': seed,
        'mean': mean,
        'std': std,
        'cv': cv
    })

print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 70)
print("VERDICT")
print("=" * 70)
print()

# Best average seed
best_avg = max(seed_stats, key=lambda x: x['mean'])
print(f"Best average performance: {best_avg['label']} ({best_avg['seed']})")
print(f"  Mean: {best_avg['mean']:.3f}")
print(f"  CV:   {best_avg['cv']:.4f}")
print()

# Most stable seed
most_stable = min(seed_stats, key=lambda x: x['cv'])
print(f"Most stable across datasets: {most_stable['label']} ({most_stable['seed']})")
print(f"  Mean: {most_stable['mean']:.3f}")
print(f"  CV:   {most_stable['cv']:.4f}")
print()

# Check if seed 31 generalizes
seed_31_stats = [s for s in seed_stats if s['seed'] == 31][0]
rank = sorted(seed_stats, key=lambda x: -x['mean']).index(seed_31_stats) + 1

print(f"Seed 31 (best for single model):")
print(f"  Cross-dataset rank: #{rank}/{len(seed_stats)}")
print(f"  Mean: {seed_31_stats['mean']:.3f}")
print(f"  CV:   {seed_31_stats['cv']:.4f}")
print()

if rank == 1:
    print("✓ SEED 31 GENERALIZES PERFECTLY ACROSS DATASETS")
    print("  → Universal recommendation confirmed")
elif rank <= 2:
    print("✓ SEED 31 GENERALIZES WELL (top 2)")
    print("  → Good universal choice")
else:
    print("✗ SEED 31 DOESN'T GENERALIZE")
    print(f"  → Better universal seed: {best_avg['label']} ({best_avg['seed']})")

print()

# Check if bad seeds are consistently bad
seed_12345_stats = [s for s in seed_stats if s['seed'] == 12345][0]
seed_42_stats = [s for s in seed_stats if s['seed'] == 42][0]

rank_12345 = sorted(seed_stats, key=lambda x: -x['mean']).index(seed_12345_stats) + 1
rank_42 = sorted(seed_stats, key=lambda x: -x['mean']).index(seed_42_stats) + 1

print("Bad seeds consistency:")
print(f"  Seed 12345: rank #{rank_12345}/{len(seed_stats)} (mean={seed_12345_stats['mean']:.3f})")
print(f"  Seed 42:    rank #{rank_42}/{len(seed_stats)} (mean={seed_42_stats['mean']:.3f})")

if rank_12345 >= len(seed_stats) - 1 and rank_42 >= len(seed_stats) - 1:
    print("  ✓ Bad seeds are consistently bad across datasets")
else:
    print("  ⚠️  Bad seeds vary by dataset")

print()
print("=" * 70)
