#!/usr/bin/env python3
"""
Test top seeds against all hashing techniques

Question: Does seed choice matter more than technique choice?

Compare:
1. Bit slicing with best seeds
2. Separate hashes with best seeds
3. Cross-compare to find optimal combination
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

def evaluate_technique_seed(embeddings, seed, use_bit_slicing=True):
    """Evaluate technique+seed combination"""
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
# Load Datasets
# ============================================================================
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")

datasets = [
    ("Twitter", PROJECT_ROOT / "semantic_proprioception_data"),
    ("ArXiv", PROJECT_ROOT / "arxiv_demo_data"),
    ("Hacker News", PROJECT_ROOT / "hackernews_demo_data"),
]

print("=" * 70)
print("TOP SEEDS × ALL TECHNIQUES")
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
# Seeds to Test
# ============================================================================

test_seeds = [
    # Top performers from 10k search
    ("Best overall", 4751),
    ("Most stable", 1056240716),
    ("Top 3", 267562796),
    ("Top 10", 321),

    # Previously identified seeds
    ("Previous best (bit slice)", 10007),
    ("Previous best (separate)", 99999),
    ("Small prime winner", 31),

    # Common seeds (for comparison)
    ("Standard (terrible)", 12345),
    ("Popular (terrible)", 42),
]

# ============================================================================
# Test All Combinations
# ============================================================================

results = []

print("Testing all seed × technique combinations...")
print()

for dataset_name in all_embeddings:
    embeddings = all_embeddings[dataset_name]

    for seed_label, seed in test_seeds:
        # Test bit slicing
        bit_slice_coh = evaluate_technique_seed(embeddings, seed, use_bit_slicing=True)

        # Test separate hashes
        separate_coh = evaluate_technique_seed(embeddings, seed, use_bit_slicing=False)

        results.append({
            'dataset': dataset_name,
            'seed_label': seed_label,
            'seed': seed,
            'bit_slicing': bit_slice_coh,
            'separate': separate_coh,
            'difference': separate_coh - bit_slice_coh
        })

print("Completed!")
print()

# ============================================================================
# Results by Dataset
# ============================================================================

for dataset_name in all_embeddings:
    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)
    print()

    dataset_results = [r for r in results if r['dataset'] == dataset_name]

    print(f"{'Seed':<30s} {'Bit Slice':<12s} {'Separate':<12s} {'Δ':<12s} {'Winner':<15s}")
    print("-" * 70)

    for r in dataset_results:
        winner = "Bit Slicing" if r['bit_slicing'] > r['separate'] else "Separate" if r['separate'] > r['bit_slicing'] else "Tie"
        print(f"{r['seed_label']:<30s} {r['bit_slicing']:<12.3f} {r['separate']:<12.3f} {r['difference']:>+10.3f}  {winner:<15s}")

    print()

# ============================================================================
# Overall Analysis
# ============================================================================
print("=" * 70)
print("OVERALL ANALYSIS")
print("=" * 70)
print()

# Compute averages across datasets
avg_by_seed = {}
for seed_label, seed in test_seeds:
    seed_results = [r for r in results if r['seed'] == seed]

    bit_scores = [r['bit_slicing'] for r in seed_results]
    sep_scores = [r['separate'] for r in seed_results]

    avg_by_seed[seed_label] = {
        'seed': seed,
        'bit_avg': np.mean(bit_scores),
        'sep_avg': np.mean(sep_scores),
        'bit_std': np.std(bit_scores),
        'sep_std': np.std(sep_scores),
        'bit_cv': np.std(bit_scores) / np.mean(bit_scores),
        'sep_cv': np.std(sep_scores) / np.mean(sep_scores),
        'diff': np.mean(sep_scores) - np.mean(bit_scores)
    }

# Sort by best average performance (either technique)
sorted_seeds = sorted(avg_by_seed.items(),
                     key=lambda x: max(x[1]['bit_avg'], x[1]['sep_avg']),
                     reverse=True)

print("Seed Rankings (by best technique):")
print()
print(f"{'Rank':<6s} {'Seed':<30s} {'Best Tech':<15s} {'Score':<10s} {'Alt Score':<10s} {'Stability (CV)':<15s}")
print("-" * 70)

for rank, (label, stats) in enumerate(sorted_seeds, 1):
    if stats['bit_avg'] > stats['sep_avg']:
        best_tech = "Bit Slicing"
        best_score = stats['bit_avg']
        alt_score = stats['sep_avg']
        cv = stats['bit_cv']
    else:
        best_tech = "Separate"
        best_score = stats['sep_avg']
        alt_score = stats['bit_avg']
        cv = stats['sep_cv']

    print(f"{rank:<6d} {label:<30s} {best_tech:<15s} {best_score:<10.3f} {alt_score:<10.3f} {cv:<15.4f}")

print()

# ============================================================================
# Technique Comparison
# ============================================================================
print("=" * 70)
print("TECHNIQUE COMPARISON (ACROSS ALL SEEDS)")
print("=" * 70)
print()

bit_wins = sum(1 for r in results if r['bit_slicing'] > r['separate'])
sep_wins = sum(1 for r in results if r['separate'] > r['bit_slicing'])
ties = sum(1 for r in results if r['bit_slicing'] == r['separate'])

print(f"Total comparisons: {len(results)}")
print(f"Bit slicing wins:  {bit_wins} ({bit_wins/len(results)*100:.1f}%)")
print(f"Separate wins:     {sep_wins} ({sep_wins/len(results)*100:.1f}%)")
print(f"Ties:              {ties}")
print()

# Average advantage
bit_scores = [r['bit_slicing'] for r in results]
sep_scores = [r['separate'] for r in results]

print(f"Average bit slicing score: {np.mean(bit_scores):.3f} ± {np.std(bit_scores):.3f}")
print(f"Average separate score:    {np.mean(sep_scores):.3f} ± {np.std(sep_scores):.3f}")
print(f"Average difference:        {np.mean(sep_scores) - np.mean(bit_scores):+.3f}")
print()

# ============================================================================
# Final Recommendation
# ============================================================================
print("=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)
print()

best_combo = max(results, key=lambda x: max(x['bit_slicing'], x['separate']))
best_score = max(best_combo['bit_slicing'], best_combo['separate'])
best_tech = "Bit Slicing" if best_combo['bit_slicing'] > best_combo['separate'] else "Separate Hashes"

print(f"Best combination found:")
print(f"  Dataset:   {best_combo['dataset']}")
print(f"  Seed:      {best_combo['seed_label']} ({best_combo['seed']})")
print(f"  Technique: {best_tech}")
print(f"  Score:     {best_score:.3f}")
print()

# Overall best seed
best_overall_seed = sorted_seeds[0]
print(f"Best overall seed: {best_overall_seed[0]} ({best_overall_seed[1]['seed']})")
print(f"  Best technique: {'Bit Slicing' if best_overall_seed[1]['bit_avg'] > best_overall_seed[1]['sep_avg'] else 'Separate Hashes'}")
print(f"  Average score:  {max(best_overall_seed[1]['bit_avg'], best_overall_seed[1]['sep_avg']):.3f}")
print()

# Does technique matter?
technique_variance = np.var([r['difference'] for r in results])
seed_variance = np.var([stats['bit_avg'] for stats in avg_by_seed.values()])

print(f"Variance analysis:")
print(f"  Seed choice variance:      {seed_variance:.6f}")
print(f"  Technique choice variance: {technique_variance:.6f}")
print()

if seed_variance > technique_variance * 2:
    print("✓ SEED CHOICE MATTERS MORE THAN TECHNIQUE")
    print("  → Focus on finding good seeds, technique is secondary")
elif technique_variance > seed_variance * 2:
    print("✓ TECHNIQUE MATTERS MORE THAN SEED")
    print("  → Separate hashes vs bit slicing is the key decision")
else:
    print("⚠️  BOTH SEED AND TECHNIQUE MATTER")
    print("  → Must optimize both for best results")

print()
print("=" * 70)
