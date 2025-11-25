#!/usr/bin/env python3
"""
Test: Orthogonal basis vs random seeds for hierarchical LSH refinement

Compares two approaches to hierarchical LSH:
1. Random seeds: Different random hyperplanes at each level (current)
2. Orthogonal basis: QR-decomposed basis, partition columns across levels

Hypothesis: Orthogonal basis provides better coherence improvement because
it guarantees information gain (no redundant projections).
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm

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

def lsh_hash_random_seed(embeddings, num_bits=8, seed=12345):
    """Traditional LSH: Random hyperplanes with given seed"""
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

def lsh_hash_orthogonal(embeddings, num_bits=8, offset=0, seed=12345):
    """Orthogonal LSH: Use columns offset:offset+num_bits from orthogonal basis"""
    np.random.seed(seed)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        single = True
    else:
        single = False

    d = embeddings.shape[1]

    # Generate orthogonal basis via QR decomposition
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)

    # Use columns offset:offset+num_bits
    hyperplanes = Q[:, offset:offset+num_bits].T
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values[0] if single else hash_values

def hierarchical_refinement_random(embeddings, max_depth=2, min_size=3):
    """
    Hierarchical LSH using different random seeds at each level

    Returns: List of (sub_bucket_embeddings, coherence, depth) tuples
    """
    def refine(embs, depth=0):
        coherence = compute_intra_bucket_similarity(embs)

        if len(embs) < min_size * 2 or depth >= max_depth:
            return [(embs, coherence, depth)]

        # Use different seed at each depth
        seed = 12345 + depth * 1000
        sub_hash = lsh_hash_random_seed(embs, num_bits=4, seed=seed)

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

def hierarchical_refinement_orthogonal(embeddings, max_depth=2, min_size=3):
    """
    Hierarchical LSH using orthogonal basis (partition columns)

    Level 0: columns 0-7 (8 bits for initial bucketing - done elsewhere)
    Level 1: columns 8-11 (4 bits for first refinement)
    Level 2: columns 12-15 (4 bits for second refinement)

    Returns: List of (sub_bucket_embeddings, coherence, depth) tuples
    """
    def refine(embs, depth=0):
        coherence = compute_intra_bucket_similarity(embs)

        if len(embs) < min_size * 2 or depth >= max_depth:
            return [(embs, coherence, depth)]

        # Use orthogonal columns: offset = 8 + depth * 4
        offset = 8 + depth * 4
        sub_hash = lsh_hash_orthogonal(embs, num_bits=4, offset=offset, seed=12345)

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

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("ORTHOGONAL BASIS vs RANDOM SEEDS: Hierarchical LSH Comparison")
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

results_random = []
results_orthogonal = []

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

    # Method 1: Random seeds
    sub_buckets_random = hierarchical_refinement_random(bucket_embeddings, max_depth=2, min_size=3)
    coherences_random = [coh for _, coh, _ in sub_buckets_random if coh is not None]
    weighted_coh_random = np.average(
        coherences_random,
        weights=[len(embs) for embs, _, _ in sub_buckets_random if compute_intra_bucket_similarity(embs) is not None]
    ) if len(coherences_random) > 0 else original_coherence

    # Method 2: Orthogonal basis
    sub_buckets_orthogonal = hierarchical_refinement_orthogonal(bucket_embeddings, max_depth=2, min_size=3)
    coherences_orthogonal = [coh for _, coh, _ in sub_buckets_orthogonal if coh is not None]
    weighted_coh_orthogonal = np.average(
        coherences_orthogonal,
        weights=[len(embs) for embs, _, _ in sub_buckets_orthogonal if compute_intra_bucket_similarity(embs) is not None]
    ) if len(coherences_orthogonal) > 0 else original_coherence

    results_random.append({
        'bucket_id': bucket_id,
        'size': count,
        'original': original_coherence,
        'refined': weighted_coh_random,
        'improvement': weighted_coh_random - original_coherence,
        'num_sub': len(sub_buckets_random)
    })

    results_orthogonal.append({
        'bucket_id': bucket_id,
        'size': count,
        'original': original_coherence,
        'refined': weighted_coh_orthogonal,
        'improvement': weighted_coh_orthogonal - original_coherence,
        'num_sub': len(sub_buckets_orthogonal)
    })

# ============================================================================
# Results
# ============================================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"{'Bucket':<8s} {'Size':<6s} {'Original':<10s} {'Random Δ':<12s} {'Orthogonal Δ':<15s} {'Winner':<10s}")
print("-" * 70)

for r_rand, r_orth in zip(results_random, results_orthogonal):
    winner = 'Orthogonal' if r_orth['improvement'] > r_rand['improvement'] else 'Random'
    if abs(r_orth['improvement'] - r_rand['improvement']) < 0.01:
        winner = 'Tie'

    print(f"{r_rand['bucket_id']:<8d} {r_rand['size']:<6d} {r_rand['original']:<10.3f} "
          f"{r_rand['improvement']:>+10.3f}  {r_orth['improvement']:>+12.3f}  {winner:<10s}")

print()

# Summary statistics
avg_improvement_random = np.mean([r['improvement'] for r in results_random])
avg_improvement_orthogonal = np.mean([r['improvement'] for r in results_orthogonal])

orthogonal_wins = sum(1 for r_rand, r_orth in zip(results_random, results_orthogonal)
                      if r_orth['improvement'] > r_rand['improvement'])

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"Average coherence improvement:")
print(f"  Random seeds:      {avg_improvement_random:+.3f} ({avg_improvement_random/np.mean([r['original'] for r in results_random])*100:+.1f}%)")
print(f"  Orthogonal basis:  {avg_improvement_orthogonal:+.3f} ({avg_improvement_orthogonal/np.mean([r['original'] for r in results_orthogonal])*100:+.1f}%)")
print()

advantage = (avg_improvement_orthogonal - avg_improvement_random) / abs(avg_improvement_random) * 100 if avg_improvement_random != 0 else 0
print(f"Orthogonal advantage: {advantage:+.1f}%")
print(f"Orthogonal wins:      {orthogonal_wins}/{len(results_random)} buckets")
print()

if avg_improvement_orthogonal > avg_improvement_random * 1.1:
    print("✓ ORTHOGONAL BASIS WINS")
    print("  → Guaranteed orthogonality provides better information gain")
elif avg_improvement_orthogonal > avg_improvement_random:
    print("✓ Orthogonal basis marginally better")
    print(f"  → {advantage:.1f}% improvement over random seeds")
elif avg_improvement_random > avg_improvement_orthogonal * 1.1:
    print("✓ RANDOM SEEDS WIN")
    print("  → Independent random projections work better")
else:
    print("⚠️  Similar performance")
    print("  → Both approaches comparable")

print()
print("Interpretation:")
print()
print("Random Seeds:")
print("  - Independent random hyperplanes at each level")
print("  - May have redundant/correlated projections")
print("  - Simple to implement")
print()
print("Orthogonal Basis:")
print("  - Guaranteed orthogonal hyperplanes across all levels")
print("  - No redundant information")
print("  - Still compositional (same seed = same basis)")
print("  - Better theoretical grounding (related to JL lemma)")
print()

if avg_improvement_orthogonal > avg_improvement_random:
    print("Recommendation: Use orthogonal basis for hierarchical LSH")
    print("  - Partition QR-decomposed random matrix columns")
    print("  - Level 0: columns 0-7, Level 1: columns 8-11, etc.")
else:
    print("Recommendation: Random seeds sufficient")
    print("  - Simpler implementation")
    print("  - Performance comparable")

print()
print("=" * 70)
