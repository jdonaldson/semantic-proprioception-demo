#!/usr/bin/env python3
"""
Test Wikipedia-learned hyperplanes vs random hyperplanes

Compares:
1. Random hyperplanes (baseline)
2. Wikipedia PCA hyperplanes (learned, general-purpose)

Datasets: Twitter, ArXiv, HackerNews

Metrics:
- Dense bucket coherence
- Density distribution (number of dense buckets)
- Hierarchical refinement quality (with bit slicing)
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm

def lsh_hash_custom(embeddings, hyperplanes):
    """Compute LSH hash using custom hyperplanes"""
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        single = True
    else:
        single = False

    num_bits = hyperplanes.shape[0]
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values[0] if single else hash_values

def lsh_hash_random(embeddings, num_bits=8, seed=12345):
    """Random hyperplanes (baseline)"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    return lsh_hash_custom(embeddings, hyperplanes)

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

def evaluate_hyperplanes(embeddings, hyperplanes, method_name):
    """Evaluate hyperplane quality on dataset"""
    # Hash all embeddings
    if method_name == "Random":
        hash_values = lsh_hash_random(embeddings, num_bits=8, seed=12345)
    else:
        # Use first 8 components for 8-bit hash (level 0)
        hash_values = lsh_hash_custom(embeddings, hyperplanes[:8])

    # Group into buckets
    buckets = {}
    for i, hash_val in enumerate(hash_values):
        if hash_val not in buckets:
            buckets[hash_val] = []
        buckets[hash_val].append(i)

    # Find dense buckets (≥5 items)
    dense_buckets = {bid: indices for bid, indices in buckets.items() if len(indices) >= 5}

    # Compute coherence for dense buckets
    coherences = []
    for indices in dense_buckets.values():
        bucket_embeddings = embeddings[indices]
        coh = compute_intra_bucket_similarity(bucket_embeddings)
        if coh is not None:
            coherences.append(coh)

    avg_coherence = np.mean(coherences) if len(coherences) > 0 else 0.0

    # Test hierarchical refinement (bit slicing with 16-bit hash)
    if method_name != "Random":
        # Use all 16 components for hierarchical
        hash_16bit = lsh_hash_custom(embeddings, hyperplanes[:16])
    else:
        hash_16bit = lsh_hash_random(embeddings, num_bits=16, seed=12345)

    # Refine dense buckets using bits 8-11
    refined_coherences = []
    for indices in dense_buckets.values():
        if len(indices) < 6:  # Need at least 6 to split
            continue

        # Extract level 1 hash (bits 8-11)
        level_1_hash = (hash_16bit[indices] >> 8) & 0x0F

        # Group into sub-buckets
        sub_buckets = {}
        for i, sub_hash in enumerate(level_1_hash):
            if sub_hash not in sub_buckets:
                sub_buckets[sub_hash] = []
            sub_buckets[sub_hash].append(indices[i])

        # Compute coherence for sub-buckets
        for sub_indices in sub_buckets.values():
            if len(sub_indices) >= 2:
                sub_embeddings = embeddings[sub_indices]
                sub_coh = compute_intra_bucket_similarity(sub_embeddings)
                if sub_coh is not None:
                    refined_coherences.append(sub_coh)

    avg_refined = np.mean(refined_coherences) if len(refined_coherences) > 0 else avg_coherence

    return {
        'method': method_name,
        'num_buckets': len(buckets),
        'num_dense': len(dense_buckets),
        'avg_coherence': avg_coherence,
        'avg_refined': avg_refined,
        'improvement': avg_refined - avg_coherence
    }

# ============================================================================
# Load Wikipedia hyperplanes
# ============================================================================
MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")

hyperplanes_file = Path(__file__).parent / f"{MODEL_NAME}_wikipedia_hyperplanes.npy"

if not hyperplanes_file.exists():
    print(f"ERROR: Wikipedia hyperplanes not found: {hyperplanes_file}")
    print("Run: python3 compute_wikipedia_hyperplanes.py")
    exit(1)

wikipedia_hyperplanes = np.load(hyperplanes_file)

print("=" * 70)
print("WIKIPEDIA HYPERPLANES vs RANDOM: Comparison")
print("=" * 70)
print()
print(f"Loaded Wikipedia hyperplanes: {wikipedia_hyperplanes.shape}")
print()

# ============================================================================
# Test on three datasets
# ============================================================================
datasets = [
    ("Twitter", PROJECT_ROOT / "semantic_proprioception_data"),
    ("ArXiv", PROJECT_ROOT / "arxiv_demo_data"),
    ("Hacker News", PROJECT_ROOT / "hackernews_demo_data"),
]

all_results = []

for dataset_name, data_dir in datasets:
    print(f"Dataset: {dataset_name}")
    print("-" * 70)

    embeddings_file = data_dir / f"MiniLM-L6_embeddings.parquet"

    if not embeddings_file.exists():
        print(f"  Embeddings not found: {embeddings_file}")
        print()
        continue

    df_embeddings = pl.read_parquet(embeddings_file)
    all_embeddings = np.array([row for row in df_embeddings['embedding']])

    print(f"  {len(all_embeddings):,} embeddings")

    # Test random hyperplanes
    results_random = evaluate_hyperplanes(all_embeddings, None, "Random")

    # Test Wikipedia hyperplanes
    results_wikipedia = evaluate_hyperplanes(all_embeddings, wikipedia_hyperplanes, "Wikipedia PCA")

    # Display results
    print()
    print(f"  {'Method':<20s} {'Buckets':<10s} {'Dense':<8s} {'Coherence':<12s} {'Refined':<12s} {'Δ':<10s}")
    print(f"  {'-'*70}")

    for r in [results_random, results_wikipedia]:
        print(f"  {r['method']:<20s} {r['num_buckets']:<10d} {r['num_dense']:<8d} "
              f"{r['avg_coherence']:<12.3f} {r['avg_refined']:<12.3f} {r['improvement']:>+8.3f}")

    # Compare
    coherence_gain = results_wikipedia['avg_coherence'] - results_random['avg_coherence']
    refined_gain = results_wikipedia['avg_refined'] - results_random['avg_refined']

    print()
    print(f"  Wikipedia advantage:")
    print(f"    Initial coherence: {coherence_gain:+.3f}")
    print(f"    Refined coherence: {refined_gain:+.3f}")

    all_results.append({
        'dataset': dataset_name,
        'random': results_random,
        'wikipedia': results_wikipedia,
        'coherence_gain': coherence_gain,
        'refined_gain': refined_gain
    })

    print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"{'Dataset':<15s} {'Random Coh':<12s} {'Wiki Coh':<12s} {'Gain':<10s} {'Winner':<15s}")
print("-" * 70)

for r in all_results:
    rand_coh = r['random']['avg_coherence']
    wiki_coh = r['wikipedia']['avg_coherence']
    gain = r['coherence_gain']

    winner = "Wikipedia" if gain > 0.01 else "Random" if gain < -0.01 else "Tie"

    print(f"{r['dataset']:<15s} {rand_coh:<12.3f} {wiki_coh:<12.3f} {gain:>+8.3f}  {winner:<15s}")

print()

# Overall statistics
avg_gain = np.mean([r['coherence_gain'] for r in all_results])
avg_refined_gain = np.mean([r['refined_gain'] for r in all_results])

wiki_wins = sum(1 for r in all_results if r['coherence_gain'] > 0.01)

print(f"Average coherence gain (Wikipedia vs Random): {avg_gain:+.3f}")
print(f"Average refined gain: {avg_refined_gain:+.3f}")
print(f"Wikipedia wins: {wiki_wins}/{len(all_results)} datasets")
print()

if avg_gain > 0.02:
    print("✓ WIKIPEDIA HYPERPLANES WIN")
    print(f"  → {avg_gain:+.3f} average coherence improvement")
    print("  → Learned structure from general corpus helps")
    print()
    print("Recommendation: Publish Wikipedia-learned hyperplanes with model")
    print("  - Still compositional (same for all users of this model)")
    print("  - Better than random (learned from diverse text)")
    print("  - Fast to use (pre-computed, just download)")
elif avg_gain < -0.02:
    print("✓ RANDOM HYPERPLANES WIN")
    print(f"  → Wikipedia hyperplanes {avg_gain:+.3f} worse")
    print("  → Random projections better for diversity")
    print()
    print("Recommendation: Stick with random hyperplanes")
else:
    print("⚠️  SIMILAR PERFORMANCE")
    print(f"  → Only {avg_gain:+.3f} difference")
    print("  → Random hyperplanes simpler, no training needed")
    print()
    print("Recommendation: Stick with random (simpler)")

print()
print("=" * 70)
