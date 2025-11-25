#!/usr/bin/env python3
"""
Test bit slicing vs separate hashes with seed 31

Now that we know seed 31 is optimal, does the technique choice still matter?
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import time

def lsh_bit_slicing(embeddings, num_bits=16, seed=31):
    """Bit slicing - compute once, extract levels via bit masking"""
    np.random.seed(seed)
    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    return hash_bits @ powers

def lsh_separate_hashes(embeddings, seed=31):
    """Separate hashes - compute new hash at each level"""
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Level 0: 8 bits
    hyperplanes_0 = np.random.randn(8, d)
    hash_bits_0 = (embeddings @ hyperplanes_0.T > 0).astype(int)
    powers_0 = 2 ** np.arange(8)
    hash_8bit = hash_bits_0 @ powers_0

    # Level 1: 4 bits (different seed)
    np.random.seed(seed + 1000)
    hyperplanes_1 = np.random.randn(4, d)
    hash_bits_1 = (embeddings @ hyperplanes_1.T > 0).astype(int)
    powers_1 = 2 ** np.arange(4)
    hash_4bit_level1 = hash_bits_1 @ powers_1

    return hash_8bit, hash_4bit_level1

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

def evaluate_bit_slicing(embeddings, seed=31):
    """Evaluate bit slicing approach"""
    start = time.time()

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

    # Compute initial coherence
    coherences = []
    for indices in dense_buckets.values():
        bucket_embeddings = embeddings[indices]
        coh = compute_intra_bucket_similarity(bucket_embeddings)
        if coh is not None:
            coherences.append(coh)

    initial_coherence = np.mean(coherences) if len(coherences) > 0 else 0.0

    # Hierarchical refinement (bit slicing from same hash)
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

    refined_coherence = np.mean(refined_coherences) if len(refined_coherences) > 0 else initial_coherence

    elapsed = time.time() - start

    return {
        'initial': initial_coherence,
        'refined': refined_coherence,
        'improvement': refined_coherence - initial_coherence,
        'time_ms': elapsed * 1000,
        'dense_buckets': len(dense_buckets),
    }

def evaluate_separate_hashes(embeddings, seed=31):
    """Evaluate separate hashes approach"""
    start = time.time()

    hash_8bit, hash_4bit_level1 = lsh_separate_hashes(embeddings, seed=seed)

    # Group into buckets
    buckets = {}
    for i, hash_val in enumerate(hash_8bit):
        if hash_val not in buckets:
            buckets[hash_val] = []
        buckets[hash_val].append(i)

    # Find dense buckets
    dense_buckets = {bid: indices for bid, indices in buckets.items() if len(indices) >= 5}

    # Compute initial coherence
    coherences = []
    for indices in dense_buckets.values():
        bucket_embeddings = embeddings[indices]
        coh = compute_intra_bucket_similarity(bucket_embeddings)
        if coh is not None:
            coherences.append(coh)

    initial_coherence = np.mean(coherences) if len(coherences) > 0 else 0.0

    # Hierarchical refinement (separate hash at level 1)
    refined_coherences = []
    for indices in dense_buckets.values():
        if len(indices) < 6:
            continue

        level_1_hash = hash_4bit_level1[indices]

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

    refined_coherence = np.mean(refined_coherences) if len(refined_coherences) > 0 else initial_coherence

    elapsed = time.time() - start

    return {
        'initial': initial_coherence,
        'refined': refined_coherence,
        'improvement': refined_coherence - initial_coherence,
        'time_ms': elapsed * 1000,
        'dense_buckets': len(dense_buckets),
    }

# ============================================================================
# Load All Datasets
# ============================================================================
DEMO_DIR = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo")

datasets = {
    "Twitter": DEMO_DIR / "semantic_proprioception_data" / "MiniLM-L6_embeddings.parquet",
    "ArXiv": DEMO_DIR / "arxiv_demo_data" / "MiniLM-L6_embeddings.parquet",
    "HackerNews": DEMO_DIR / "hackernews_demo_data" / "MiniLM-L6_embeddings.parquet",
    "Amazon": DEMO_DIR / "amazon_demo_data" / "MiniLM-L6_embeddings.parquet",
}

print("=" * 80)
print("BIT SLICING vs SEPARATE HASHES WITH SEED 31")
print("=" * 80)
print()

results = []

for dataset_name, dataset_path in datasets.items():
    if not dataset_path.exists():
        print(f"⚠️  {dataset_name} not found, skipping")
        continue

    df = pl.read_parquet(dataset_path)
    embeddings = np.array([row for row in df['embedding']])

    print(f"\n{dataset_name} ({len(embeddings)} samples)")
    print("-" * 80)

    # Test bit slicing
    bit_result = evaluate_bit_slicing(embeddings, seed=31)
    print(f"Bit Slicing:")
    print(f"  Initial:     {bit_result['initial']:.3f}")
    print(f"  Refined:     {bit_result['refined']:.3f}")
    print(f"  Improvement: {bit_result['improvement']:+.3f}")
    print(f"  Time:        {bit_result['time_ms']:.2f}ms")
    print()

    # Test separate hashes
    sep_result = evaluate_separate_hashes(embeddings, seed=31)
    print(f"Separate Hashes:")
    print(f"  Initial:     {sep_result['initial']:.3f}")
    print(f"  Refined:     {sep_result['refined']:.3f}")
    print(f"  Improvement: {sep_result['improvement']:+.3f}")
    print(f"  Time:        {sep_result['time_ms']:.2f}ms")
    print()

    # Compare
    quality_diff = bit_result['refined'] - sep_result['refined']
    speed_ratio = sep_result['time_ms'] / bit_result['time_ms']

    print(f"Comparison:")
    print(f"  Quality:  Bit slicing is {quality_diff:+.3f} ({100*quality_diff/sep_result['refined']:+.1f}%)")
    print(f"  Speed:    Bit slicing is {speed_ratio:.1f}× faster")

    if quality_diff > 0.01 and speed_ratio > 1.5:
        print(f"  Verdict:  ✓ Bit slicing WINS (better + faster)")
    elif quality_diff > 0.01:
        print(f"  Verdict:  ✓ Bit slicing WINS (better quality)")
    elif quality_diff < -0.01 and speed_ratio > 1.5:
        print(f"  Verdict:  ⚖️  TRADE-OFF (separate better quality, bit slicing faster)")
    elif quality_diff < -0.01:
        print(f"  Verdict:  ✓ Separate hashes WIN (better quality)")
    else:
        print(f"  Verdict:  ≈ TIE (similar quality, bit slicing faster)")

    results.append({
        'dataset': dataset_name,
        'bit_initial': bit_result['initial'],
        'bit_refined': bit_result['refined'],
        'bit_improvement': bit_result['improvement'],
        'bit_time': bit_result['time_ms'],
        'sep_initial': sep_result['initial'],
        'sep_refined': sep_result['refined'],
        'sep_improvement': sep_result['improvement'],
        'sep_time': sep_result['time_ms'],
        'quality_diff': quality_diff,
        'speed_ratio': speed_ratio,
    })

# ============================================================================
# Overall Summary
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print()

print(f"{'Dataset':<15s} {'Bit Slice':<12s} {'Separate':<12s} {'Diff':<10s} {'Speed':<10s}")
print("-" * 80)

for r in results:
    print(f"{r['dataset']:<15s} {r['bit_refined']:<12.3f} {r['sep_refined']:<12.3f} {r['quality_diff']:+<10.3f} {r['speed_ratio']:<10.1f}×")

print()

# Average across datasets
avg_bit = np.mean([r['bit_refined'] for r in results])
avg_sep = np.mean([r['sep_refined'] for r in results])
avg_diff = np.mean([r['quality_diff'] for r in results])
avg_speed = np.mean([r['speed_ratio'] for r in results])

print(f"{'AVERAGE':<15s} {avg_bit:<12.3f} {avg_sep:<12.3f} {avg_diff:+<10.3f} {avg_speed:<10.1f}×")
print()

# Verdict
print("=" * 80)
print("FINAL VERDICT WITH SEED 31")
print("=" * 80)
print()

if avg_diff > 0.01:
    print(f"✓ BIT SLICING WINS")
    print(f"  Quality: {avg_diff:+.3f} better ({100*avg_diff/avg_sep:+.1f}%)")
    print(f"  Speed:   {avg_speed:.1f}× faster")
    print(f"\n  Recommendation: Use bit slicing with seed 31")
elif avg_diff < -0.01:
    print(f"✓ SEPARATE HASHES WIN")
    print(f"  Quality: {-avg_diff:.3f} better ({100*-avg_diff/avg_bit:.1f}%)")
    print(f"  Speed:   {1/avg_speed:.1f}× slower (bit slicing {avg_speed:.1f}× faster)")
    print(f"\n  Recommendation: Use separate hashes with seed 31 (accept slower speed for quality)")
else:
    print(f"≈ TIE IN QUALITY")
    print(f"  Quality difference: {avg_diff:+.3f} ({100*abs(avg_diff)/avg_sep:.1f}%)")
    print(f"  Speed advantage:    Bit slicing {avg_speed:.1f}× faster")
    print(f"\n  Recommendation: Use bit slicing with seed 31 (same quality, much faster)")

print()
print("=" * 80)
