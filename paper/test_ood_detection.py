#!/usr/bin/env python3
"""
Test out-of-distribution detection using LSH bucket density

Hypothesis: Out-of-distribution text (random, nonsense, or unrelated domains)
will hash to sparse/empty buckets compared to in-distribution text.
"""
import polars as pl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def lsh_hash(embeddings, num_bits=8, seed=12345):
    """
    Compute LSH hash for embeddings using random hyperplanes

    Args:
        embeddings: np.array of shape (n, d) or (d,) for single embedding
        num_bits: Number of hash bits
        seed: Random seed

    Returns:
        np.array of hash values (integers) or single integer
    """
    np.random.seed(seed)

    # Handle single embedding vs batch
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        single = True
    else:
        single = False

    d = embeddings.shape[1]

    # Generate random hyperplanes
    hyperplanes = np.random.randn(num_bits, d)

    # Compute hash: sign of dot product
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)

    # Convert binary to integer
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values[0] if single else hash_values

# Data paths
ARXIV_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/arxiv_demo_data")
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("OUT-OF-DISTRIBUTION DETECTION TEST")
print("=" * 70)
print()
print("Testing whether OOD text lands in sparse/empty buckets")
print()

# ============================================================================
# Load ArXiv data (in-distribution)
# ============================================================================
print("Loading ArXiv data (in-distribution)...")
embeddings_file = ARXIV_DATA / f"{MODEL_NAME}_embeddings.parquet"
index_file = ARXIV_DATA / f"{MODEL_NAME}_lsh_index.parquet"

df_embeddings = pl.read_parquet(embeddings_file)
df_index = pl.read_parquet(index_file)

print(f"  {len(df_embeddings):,} ArXiv abstracts")
print()

# Compute bucket density distribution
bucket_counts = (df_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
)

dense_threshold = 5
dense_buckets = set(bucket_counts.filter(pl.col('count') >= dense_threshold)['bucket_id'].to_list())
sparse_buckets = set(bucket_counts.filter((pl.col('count') > 0) & (pl.col('count') < dense_threshold))['bucket_id'].to_list())
occupied_buckets = set(bucket_counts['bucket_id'].to_list())

print(f"Bucket distribution:")
print(f"  Dense buckets (≥{dense_threshold}):     {len(dense_buckets)}")
print(f"  Sparse buckets (1-{dense_threshold-1}):        {len(sparse_buckets)}")
print(f"  Empty buckets:            {256 - len(occupied_buckets)} (out of 256 total)")
print()

# ============================================================================
# Load embedding model for OOD text
# ============================================================================
print("Loading embedding model...")
model = SentenceTransformer(f"sentence-transformers/all-{MODEL_NAME}-v2")
print(f"  ✓ Loaded {MODEL_NAME}")
print()

# ============================================================================
# Test 1: In-distribution (ArXiv abstracts)
# ============================================================================
print("TEST 1: In-Distribution (sample ArXiv abstracts)")
print("-" * 70)

# Sample 50 random ArXiv abstracts
sample_size = 50
arxiv_sample = df_embeddings.sample(n=sample_size, seed=42)

in_dist_buckets = []
for row in arxiv_sample.iter_rows(named=True):
    embedding = np.array(row['embedding'])
    bucket_id = lsh_hash(embedding, num_bits=8, seed=12345)

    matches = bucket_counts.filter(pl.col('bucket_id') == bucket_id)
    bucket_size = matches['count'][0] if len(matches) > 0 else 0
    in_dist_buckets.append((bucket_id, bucket_size))

in_dist_dense = sum(1 for _, size in in_dist_buckets if size >= dense_threshold)
in_dist_sparse = sum(1 for _, size in in_dist_buckets if 0 < size < dense_threshold)
in_dist_empty = sum(1 for _, size in in_dist_buckets if size == 0)
in_dist_avg_density = np.mean([size for _, size in in_dist_buckets])

print(f"Sample size: {sample_size}")
print(f"  Dense buckets (≥{dense_threshold}):     {in_dist_dense:3d} ({in_dist_dense/sample_size*100:.1f}%)")
print(f"  Sparse buckets (1-{dense_threshold-1}):        {in_dist_sparse:3d} ({in_dist_sparse/sample_size*100:.1f}%)")
print(f"  Empty buckets (0):        {in_dist_empty:3d} ({in_dist_empty/sample_size*100:.1f}%)")
print(f"  Avg bucket density:       {in_dist_avg_density:.2f}")
print()

# ============================================================================
# Test 2: Out-of-distribution (random nonsense)
# ============================================================================
print("TEST 2: Out-of-Distribution (random word salad)")
print("-" * 70)

# Generate random nonsense text
np.random.seed(42)
word_bank = ["quantum", "blockchain", "synergy", "paradigm", "leverage",
             "disrupt", "innovate", "optimize", "scalable", "robust",
             "algorithm", "framework", "methodology", "architecture", "infrastructure",
             "banana", "purple", "fluffy", "dancing", "sparkle", "unicorn",
             "potato", "rainbow", "giggle", "wobble", "splendid"]

ood_texts = []
for _ in range(sample_size):
    # Random 10-20 word sequences
    length = np.random.randint(10, 21)
    words = np.random.choice(word_bank, size=length, replace=True)
    ood_texts.append(" ".join(words))

# Embed OOD texts
print(f"Generating {sample_size} random word salad texts...")
ood_embeddings = model.encode(ood_texts, show_progress_bar=False)

ood_buckets = []
for embedding in ood_embeddings:
    bucket_id = lsh_hash(embedding, num_bits=8, seed=12345)

    matches = bucket_counts.filter(pl.col('bucket_id') == bucket_id)
    bucket_size = matches['count'][0] if len(matches) > 0 else 0
    ood_buckets.append((bucket_id, bucket_size))

ood_dense = sum(1 for _, size in ood_buckets if size >= dense_threshold)
ood_sparse = sum(1 for _, size in ood_buckets if 0 < size < dense_threshold)
ood_empty = sum(1 for _, size in ood_buckets if size == 0)
ood_avg_density = np.mean([size for _, size in ood_buckets])

print(f"Sample size: {sample_size}")
print(f"  Dense buckets (≥{dense_threshold}):     {ood_dense:3d} ({ood_dense/sample_size*100:.1f}%)")
print(f"  Sparse buckets (1-{dense_threshold-1}):        {ood_sparse:3d} ({ood_sparse/sample_size*100:.1f}%)")
print(f"  Empty buckets (0):        {ood_empty:3d} ({ood_empty/sample_size*100:.1f}%)")
print(f"  Avg bucket density:       {ood_avg_density:.2f}")
print()

# ============================================================================
# Test 3: Cross-domain (Twitter customer support in ArXiv index)
# ============================================================================
print("TEST 3: Cross-Domain (Twitter customer support vs ArXiv index)")
print("-" * 70)

# Load Twitter data
TWITTER_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/semantic_proprioception_data")
twitter_embeddings_file = TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet"

if twitter_embeddings_file.exists():
    df_twitter = pl.read_parquet(twitter_embeddings_file).sample(n=sample_size, seed=42)

    print(f"Testing {sample_size} Twitter customer support tweets against ArXiv index...")

    cross_domain_buckets = []
    for row in df_twitter.iter_rows(named=True):
        embedding = np.array(row['embedding'])
        bucket_id = lsh_hash(embedding, num_bits=8, seed=12345)

        matches = bucket_counts.filter(pl.col('bucket_id') == bucket_id)
        bucket_size = matches['count'][0] if len(matches) > 0 else 0
        cross_domain_buckets.append((bucket_id, bucket_size))

    cross_dense = sum(1 for _, size in cross_domain_buckets if size >= dense_threshold)
    cross_sparse = sum(1 for _, size in cross_domain_buckets if 0 < size < dense_threshold)
    cross_empty = sum(1 for _, size in cross_domain_buckets if size == 0)
    cross_avg_density = np.mean([size for _, size in cross_domain_buckets])

    print(f"Sample size: {sample_size}")
    print(f"  Dense buckets (≥{dense_threshold}):     {cross_dense:3d} ({cross_dense/sample_size*100:.1f}%)")
    print(f"  Sparse buckets (1-{dense_threshold-1}):        {cross_sparse:3d} ({cross_sparse/sample_size*100:.1f}%)")
    print(f"  Empty buckets (0):        {cross_empty:3d} ({cross_empty/sample_size*100:.1f}%)")
    print(f"  Avg bucket density:       {cross_avg_density:.2f}")
else:
    print("  Twitter data not found, skipping cross-domain test")

print()

# ============================================================================
# RESULTS
# ============================================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print(f"{'Test':<30s} {'Dense %':<10s} {'Sparse %':<10s} {'Empty %':<10s} {'Avg Density':<12s}")
print("-" * 70)
print(f"{'In-Distribution (ArXiv)':<30s} {in_dist_dense/sample_size*100:>7.1f}%  {in_dist_sparse/sample_size*100:>7.1f}%  {in_dist_empty/sample_size*100:>7.1f}%  {in_dist_avg_density:>10.2f}")
print(f"{'Out-of-Distribution (Random)':<30s} {ood_dense/sample_size*100:>7.1f}%  {ood_sparse/sample_size*100:>7.1f}%  {ood_empty/sample_size*100:>7.1f}%  {ood_avg_density:>10.2f}")
if twitter_embeddings_file.exists():
    print(f"{'Cross-Domain (Twitter→ArXiv)':<30s} {cross_dense/sample_size*100:>7.1f}%  {cross_sparse/sample_size*100:>7.1f}%  {cross_empty/sample_size*100:>7.1f}%  {cross_avg_density:>10.2f}")
print()

# Statistical summary
print("Key findings:")
print()
print(f"1. In-distribution (ArXiv) → {in_dist_dense/sample_size*100:.1f}% land in dense buckets")
print(f"   Average density: {in_dist_avg_density:.2f} items/bucket")
print()
print(f"2. Out-of-distribution (random) → {ood_dense/sample_size*100:.1f}% land in dense buckets")
print(f"   Average density: {ood_avg_density:.2f} items/bucket")
print(f"   Density reduction: {(1 - ood_avg_density/in_dist_avg_density)*100:.1f}%")
print()

if twitter_embeddings_file.exists():
    print(f"3. Cross-domain (Twitter→ArXiv) → {cross_dense/sample_size*100:.1f}% land in dense buckets")
    print(f"   Average density: {cross_avg_density:.2f} items/bucket")
    print(f"   Density reduction: {(1 - cross_avg_density/in_dist_avg_density)*100:.1f}%")
    print()

# Interpretation
print("-" * 70)
if ood_avg_density < in_dist_avg_density * 0.5:
    print("✓ OOD text shows SIGNIFICANTLY lower bucket density")
    print("  → Bucket density is a viable OOD detection signal")
elif ood_avg_density < in_dist_avg_density * 0.8:
    print("✓ OOD text shows MODERATELY lower bucket density")
    print("  → Bucket density provides some OOD signal, but not strong")
else:
    print("⚠️  OOD text shows SIMILAR bucket density to in-distribution")
    print("  → Bucket density may not be a strong OOD detector")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
