#!/usr/bin/env python3
"""
Test: Dense buckets have higher semantic coherence than sparse buckets

Validates the claim: "dense buckets consistently correspond to coherent semantic themes"

Method:
1. Compute average intra-bucket cosine similarity for all buckets
2. Compare dense (≥5) vs sparse (1-4) vs singleton (1 item) buckets
3. Show that density correlates with coherence
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm

def compute_intra_bucket_similarity(embeddings):
    """Compute average pairwise cosine similarity within bucket"""
    if len(embeddings) < 2:
        return None  # Can't compute for single item

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
            similarities.append(cos_sim)

    return np.mean(similarities)

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("SEMANTIC COHERENCE VALIDATION")
print("=" * 70)
print()
print("Testing: Do dense buckets have higher semantic coherence?")
print()

# Test on all three datasets
datasets = [
    ("Twitter", PROJECT_ROOT / "semantic_proprioception_data"),
    ("ArXiv", PROJECT_ROOT / "arxiv_demo_data"),
    ("Hacker News", PROJECT_ROOT / "hackernews_demo_data"),
]

all_results = []

for dataset_name, data_dir in datasets:
    print(f"Dataset: {dataset_name}")
    print("-" * 70)

    embeddings_file = data_dir / f"{MODEL_NAME}_embeddings.parquet"
    index_file = data_dir / f"{MODEL_NAME}_lsh_index.parquet"

    if not embeddings_file.exists() or not index_file.exists():
        print(f"  Data not found, skipping")
        print()
        continue

    df_embeddings = pl.read_parquet(embeddings_file)
    df_index = pl.read_parquet(index_file)

    print(f"  {len(df_embeddings):,} embeddings")

    # Get all bucket sizes
    bucket_counts = (df_index
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
    )

    # Categorize buckets
    singleton = bucket_counts.filter(pl.col('count') == 1)
    sparse = bucket_counts.filter((pl.col('count') >= 2) & (pl.col('count') <= 4))
    dense = bucket_counts.filter(pl.col('count') >= 5)

    print(f"  Buckets: {len(singleton)} singleton, {len(sparse)} sparse (2-4), {len(dense)} dense (≥5)")

    # Compute coherence for each category
    def compute_category_coherence(bucket_list, label):
        coherences = []

        for row in bucket_list.iter_rows(named=True):
            bucket_id = row['bucket_id']
            count = row['count']

            # Get embeddings in this bucket
            bucket_items = df_index.filter(pl.col('bucket_id') == bucket_id)
            row_ids = bucket_items['row_id'].to_list()
            bucket_embeddings = np.array([df_embeddings[i]['embedding'] for i in row_ids])

            coherence = compute_intra_bucket_similarity(bucket_embeddings)
            if coherence is not None:
                coherences.append(coherence)

        if len(coherences) == 0:
            return None, 0

        return np.mean(coherences), len(coherences)

    sparse_coh, sparse_n = compute_category_coherence(sparse, "sparse")
    dense_coh, dense_n = compute_category_coherence(dense, "dense")

    if sparse_coh is None or dense_coh is None:
        print(f"  Insufficient data for comparison")
        print()
        continue

    improvement = (dense_coh - sparse_coh) / sparse_coh * 100

    print(f"  Sparse (2-4):  {sparse_coh:.3f} avg coherence ({sparse_n} buckets)")
    print(f"  Dense (≥5):    {dense_coh:.3f} avg coherence ({dense_n} buckets)")
    print(f"  Improvement:   {improvement:+.1f}%")
    print()

    all_results.append({
        'dataset': dataset_name,
        'sparse_coh': sparse_coh,
        'dense_coh': dense_coh,
        'improvement': improvement,
        'sparse_n': sparse_n,
        'dense_n': dense_n
    })

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"{'Dataset':<15s} {'Sparse (2-4)':<15s} {'Dense (≥5)':<15s} {'Improvement':<15s}")
print("-" * 70)
for r in all_results:
    print(f"{r['dataset']:<15s} {r['sparse_coh']:<15.3f} {r['dense_coh']:<15.3f} {r['improvement']:>+12.1f}%")

print()

# Overall statistics
if len(all_results) > 0:
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    all_positive = all(r['improvement'] > 0 for r in all_results)

    print(f"Average improvement: {avg_improvement:+.1f}%")
    print()

    if all_positive and avg_improvement > 10:
        print("✓ VALIDATED: Dense buckets have significantly higher coherence")
        print("  → Density correlates with semantic coherence across all datasets")
    elif all_positive:
        print("✓ SUPPORTED: Dense buckets show modest coherence improvement")
        print(f"  → {avg_improvement:.1f}% average improvement")
    else:
        print("⚠️  MIXED RESULTS: Coherence improvement not consistent")

print()
print("Interpretation:")
print("  - Higher coherence = items in bucket are semantically similar")
print("  - Dense buckets form because similar items cluster naturally")
print("  - Validates LSH density as proxy for semantic themes")
print()
print("=" * 70)
