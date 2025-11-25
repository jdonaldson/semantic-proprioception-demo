#!/usr/bin/env python3
"""
Comprehensive Model × Dataset × Seed Analysis

Tests all combinations to answer:
1. Which seeds work across models AND datasets?
2. Is there a universal best seed?
3. What are the stability trade-offs?
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
# Load All Model × Dataset Combinations
# ============================================================================
DEMO_DIR = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo")

datasets = {
    "Twitter": DEMO_DIR / "semantic_proprioception_data",
    "ArXiv": DEMO_DIR / "arxiv_demo_data",
    "HackerNews": DEMO_DIR / "hackernews_demo_data",
    "Amazon": DEMO_DIR / "amazon_demo_data",
}

models = ["MiniLM-L3", "MiniLM-L6", "MiniLM-L12", "MPNet-base"]

print("=" * 80)
print("COMPREHENSIVE MODEL × DATASET × SEED ANALYSIS")
print("=" * 80)
print()

# Load all combinations
all_embeddings = {}
for dataset_name, dataset_path in datasets.items():
    for model_name in models:
        embeddings_file = dataset_path / f"{model_name}_embeddings.parquet"
        if embeddings_file.exists():
            df = pl.read_parquet(embeddings_file)
            embeddings = np.array([row for row in df['embedding']])
            key = f"{dataset_name}_{model_name}"
            all_embeddings[key] = embeddings
            print(f"✓ {key:<30s} {len(embeddings):>4d} samples, {embeddings.shape[1]:>3d}D")
        else:
            print(f"✗ {key:<30s} NOT FOUND")

print()
print(f"Total combinations loaded: {len(all_embeddings)}")
print()

# ============================================================================
# Seeds to Test
# ============================================================================

test_seeds = [
    ("Best overall", 31),
    ("Top from 10k", 4751),
    ("Most stable", 1056240716),
    ("Previous best", 10007),
    ("Common good", 99999),
    ("Common bad", 12345),
    ("Worst common", 42),
]

# ============================================================================
# Test All Combinations
# ============================================================================

results = []

print("Testing all combinations...")
start_time = time.time()

total_tests = len(all_embeddings) * len(test_seeds)
completed = 0

for key in sorted(all_embeddings.keys()):
    embeddings = all_embeddings[key]
    dataset_name, model_name = key.rsplit('_', 1)

    for seed_label, seed in test_seeds:
        coherence = evaluate_seed_fast(embeddings, seed)
        results.append({
            'dataset': dataset_name,
            'model': model_name,
            'seed_label': seed_label,
            'seed': seed,
            'coherence': coherence
        })

        completed += 1
        if completed % 10 == 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = (total_tests - completed) / rate
            print(f"  Progress: {completed}/{total_tests} ({100*completed/total_tests:.1f}%) - {remaining:.0f}s remaining")

elapsed = time.time() - start_time
print(f"\nCompleted {total_tests} tests in {elapsed:.1f}s ({total_tests/elapsed:.1f} tests/sec)")
print()

# ============================================================================
# Analysis 1: Best Seed Overall (Across ALL Dimensions)
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: BEST SEED ACROSS ALL MODELS AND DATASETS")
print("=" * 80)
print()

seed_stats = {}
for seed_label, seed in test_seeds:
    seed_results = [r for r in results if r['seed'] == seed]
    scores = [r['coherence'] for r in seed_results]

    seed_stats[seed] = {
        'label': seed_label,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
        'min': min(scores),
        'max': max(scores),
        'range': max(scores) - min(scores)
    }

# Sort by mean coherence
sorted_seeds = sorted(seed_stats.items(), key=lambda x: -x[1]['mean'])

print(f"{'Rank':<6s} {'Seed':<30s} {'Mean':<8s} {'Std':<8s} {'CV':<8s} {'Range':<15s}")
print("-" * 80)

for rank, (seed, stats) in enumerate(sorted_seeds, 1):
    range_str = f"[{stats['min']:.3f}, {stats['max']:.3f}]"
    print(f"{rank:<6d} {stats['label']:<30s} {stats['mean']:<8.3f} {stats['std']:<8.4f} {stats['cv']:<8.4f} {range_str:<15s}")

print()

# ============================================================================
# Analysis 2: Seed Performance by Dataset
# ============================================================================

print("=" * 80)
print("ANALYSIS 2: SEED PERFORMANCE BY DATASET")
print("=" * 80)
print()

dataset_names = sorted(set(r['dataset'] for r in results))

for dataset_name in dataset_names:
    print(f"\n{dataset_name}")
    print("-" * 80)

    dataset_results = [r for r in results if r['dataset'] == dataset_name]

    # Aggregate across models for this dataset
    seed_scores = {}
    for seed_label, seed in test_seeds:
        seed_dataset_results = [r for r in dataset_results if r['seed'] == seed]
        scores = [r['coherence'] for r in seed_dataset_results]
        seed_scores[seed] = {
            'label': seed_label,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
        }

    sorted_dataset_seeds = sorted(seed_scores.items(), key=lambda x: -x[1]['mean'])

    print(f"  {'Rank':<6s} {'Seed':<30s} {'Mean':<8s} {'Std':<8s} {'CV':<8s}")
    for rank, (seed, stats) in enumerate(sorted_dataset_seeds, 1):
        print(f"  {rank:<6d} {stats['label']:<30s} {stats['mean']:<8.3f} {stats['std']:<8.4f} {stats['cv']:<8.4f}")

print()

# ============================================================================
# Analysis 3: Seed Performance by Model
# ============================================================================

print("=" * 80)
print("ANALYSIS 3: SEED PERFORMANCE BY MODEL")
print("=" * 80)
print()

model_names = sorted(set(r['model'] for r in results))

for model_name in model_names:
    print(f"\n{model_name}")
    print("-" * 80)

    model_results = [r for r in results if r['model'] == model_name]

    # Aggregate across datasets for this model
    seed_scores = {}
    for seed_label, seed in test_seeds:
        seed_model_results = [r for r in model_results if r['seed'] == seed]
        scores = [r['coherence'] for r in seed_model_results]
        seed_scores[seed] = {
            'label': seed_label,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
        }

    sorted_model_seeds = sorted(seed_scores.items(), key=lambda x: -x[1]['mean'])

    print(f"  {'Rank':<6s} {'Seed':<30s} {'Mean':<8s} {'Std':<8s} {'CV':<8s}")
    for rank, (seed, stats) in enumerate(sorted_model_seeds, 1):
        print(f"  {rank:<6d} {stats['label']:<30s} {stats['mean']:<8.3f} {stats['std']:<8.4f} {stats['cv']:<8.4f}")

print()

# ============================================================================
# Analysis 4: Full Matrix View
# ============================================================================

print("=" * 80)
print("ANALYSIS 4: FULL MATRIX (DATASET × MODEL × SEED)")
print("=" * 80)
print()

for seed_label, seed in test_seeds:
    print(f"\nSeed: {seed_label} ({seed})")
    print("-" * 80)

    # Create matrix header
    print(f"{'Dataset':<15s} ", end="")
    for model_name in model_names:
        print(f"{model_name:<12s} ", end="")
    print("Mean")

    print("-" * 80)

    # Print matrix rows
    for dataset_name in dataset_names:
        print(f"{dataset_name:<15s} ", end="")

        row_scores = []
        for model_name in model_names:
            key = f"{dataset_name}_{model_name}"
            if key in all_embeddings:
                result = [r for r in results if r['dataset'] == dataset_name and r['model'] == model_name and r['seed'] == seed]
                if result:
                    score = result[0]['coherence']
                    row_scores.append(score)
                    print(f"{score:<12.3f} ", end="")
                else:
                    print(f"{'---':<12s} ", end="")
            else:
                print(f"{'---':<12s} ", end="")

        if row_scores:
            print(f"{np.mean(row_scores):.3f}")
        else:
            print("---")

print()

# ============================================================================
# Analysis 5: Variance Decomposition
# ============================================================================

print("=" * 80)
print("ANALYSIS 5: VARIANCE DECOMPOSITION")
print("=" * 80)
print()

all_scores = [r['coherence'] for r in results]
total_variance = np.var(all_scores)

# Variance by seed
seed_means = {}
for seed_label, seed in test_seeds:
    seed_results = [r for r in results if r['seed'] == seed]
    seed_means[seed] = np.mean([r['coherence'] for r in seed_results])

seed_variance = np.var(list(seed_means.values()))

# Variance by dataset
dataset_means = {}
for dataset_name in dataset_names:
    dataset_results = [r for r in results if r['dataset'] == dataset_name]
    dataset_means[dataset_name] = np.mean([r['coherence'] for r in dataset_results])

dataset_variance = np.var(list(dataset_means.values()))

# Variance by model
model_means = {}
for model_name in model_names:
    model_results = [r for r in results if r['model'] == model_name]
    model_means[model_name] = np.mean([r['coherence'] for r in model_results])

model_variance = np.var(list(model_means.values()))

print(f"Total variance: {total_variance:.6f}")
print()
print(f"Variance explained by:")
print(f"  Seed choice:      {seed_variance:.6f} ({100*seed_variance/total_variance:.1f}%)")
print(f"  Dataset choice:   {dataset_variance:.6f} ({100*dataset_variance/total_variance:.1f}%)")
print(f"  Model choice:     {model_variance:.6f} ({100*model_variance/total_variance:.1f}%)")
print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

best_overall = sorted_seeds[0]
most_stable = min(seed_stats.items(), key=lambda x: x[1]['cv'])

print(f"Best overall performance: {best_overall[1]['label']} (seed {best_overall[0]})")
print(f"  Mean: {best_overall[1]['mean']:.3f}")
print(f"  CV:   {best_overall[1]['cv']:.4f}")
print(f"  Range: [{best_overall[1]['min']:.3f}, {best_overall[1]['max']:.3f}]")
print()

print(f"Most stable (lowest CV): {most_stable[1]['label']} (seed {most_stable[0]})")
print(f"  Mean: {most_stable[1]['mean']:.3f}")
print(f"  CV:   {most_stable[1]['cv']:.4f}")
print(f"  Range: [{most_stable[1]['min']:.3f}, {most_stable[1]['max']:.3f}]")
print()

# Check if they're the same
if best_overall[0] == most_stable[0]:
    print("✓ SAME SEED wins both metrics - clear winner!")
else:
    performance_gap = best_overall[1]['mean'] - most_stable[1]['mean']
    stability_gap = most_stable[1]['cv'] - best_overall[1]['cv']
    print(f"⚠️  TRADE-OFF detected:")
    print(f"  Best performance is {performance_gap:.3f} better ({100*performance_gap/most_stable[1]['mean']:.1f}%)")
    print(f"  Most stable has {stability_gap:.4f} lower CV ({100*stability_gap/best_overall[1]['cv']:.1f}% more stable)")

print()

# Bad seeds
worst_seed = sorted_seeds[-1]
print(f"Worst seed: {worst_seed[1]['label']} (seed {worst_seed[0]})")
print(f"  Mean: {worst_seed[1]['mean']:.3f} ({100*(best_overall[1]['mean']-worst_seed[1]['mean'])/worst_seed[1]['mean']:.1f}% worse than best)")
print()

print("=" * 80)
print("Recommendations:")
print()
print("1. For production (best average):     Use seed", best_overall[0])
print("2. For stability (lowest variance):   Use seed", most_stable[0])
print("3. Avoid:                              Seeds 42 and 12345")
print("=" * 80)
