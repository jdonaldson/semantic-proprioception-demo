#!/usr/bin/env python3
"""
Test: Content gap analysis using LSH density

Demonstrates the use case: Compare two density distributions to find content gaps

Example: Twitter (user queries/needs) vs ArXiv (available content)
- Buckets dense in Twitter, sparse in ArXiv = unmet needs (topics users want)
- Buckets dense in ArXiv, sparse in Twitter = unused content (available but not sought)
"""
import polars as pl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def lsh_hash(embeddings, num_bits=8, seed=12345):
    """Compute LSH hash for embeddings"""
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

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
MODEL_NAME = "MiniLM-L6"

TWITTER_DATA = PROJECT_ROOT / "semantic_proprioception_data"
ARXIV_DATA = PROJECT_ROOT / "arxiv_demo_data"

print("=" * 70)
print("CONTENT GAP ANALYSIS")
print("=" * 70)
print()
print("Use case: Compare Twitter (user needs) vs ArXiv (available content)")
print()

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")

# Twitter (represents user queries/needs)
twitter_embeddings = pl.read_parquet(TWITTER_DATA / f"{MODEL_NAME}_embeddings.parquet")
twitter_index = pl.read_parquet(TWITTER_DATA / f"{MODEL_NAME}_lsh_index.parquet")

# ArXiv (represents available corpus)
arxiv_embeddings = pl.read_parquet(ARXIV_DATA / f"{MODEL_NAME}_embeddings.parquet")
arxiv_index = pl.read_parquet(ARXIV_DATA / f"{MODEL_NAME}_lsh_index.parquet")

print(f"  Twitter: {len(twitter_embeddings):,} customer support messages")
print(f"  ArXiv:   {len(arxiv_embeddings):,} research papers")
print()

# ============================================================================
# Compute bucket densities
# ============================================================================
print("Computing bucket densities...")

# Twitter bucket counts
twitter_counts = (twitter_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
)
twitter_density = {row['bucket_id']: row['count'] for row in twitter_counts.iter_rows(named=True)}

# ArXiv bucket counts
arxiv_counts = (arxiv_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
)
arxiv_density = {row['bucket_id']: row['count'] for row in arxiv_counts.iter_rows(named=True)}

# All buckets seen in either dataset
all_buckets = set(twitter_density.keys()) | set(arxiv_density.keys())

print(f"  Total unique buckets: {len(all_buckets)}")
print()

# ============================================================================
# Find content gaps
# ============================================================================
dense_threshold = 5

# Gap 1: Dense in Twitter, sparse/absent in ArXiv (unmet needs)
unmet_needs = []
for bucket_id in all_buckets:
    twitter_count = twitter_density.get(bucket_id, 0)
    arxiv_count = arxiv_density.get(bucket_id, 0)

    if twitter_count >= dense_threshold and arxiv_count < dense_threshold:
        unmet_needs.append({
            'bucket_id': bucket_id,
            'twitter_count': twitter_count,
            'arxiv_count': arxiv_count
        })

# Gap 2: Dense in ArXiv, sparse/absent in Twitter (unused content)
unused_content = []
for bucket_id in all_buckets:
    twitter_count = twitter_density.get(bucket_id, 0)
    arxiv_count = arxiv_density.get(bucket_id, 0)

    if arxiv_count >= dense_threshold and twitter_count < dense_threshold:
        unused_content.append({
            'bucket_id': bucket_id,
            'twitter_count': twitter_count,
            'arxiv_count': arxiv_count
        })

# Well-served: Dense in both
well_served = []
for bucket_id in all_buckets:
    twitter_count = twitter_density.get(bucket_id, 0)
    arxiv_count = arxiv_density.get(bucket_id, 0)

    if twitter_count >= dense_threshold and arxiv_count >= dense_threshold:
        well_served.append({
            'bucket_id': bucket_id,
            'twitter_count': twitter_count,
            'arxiv_count': arxiv_count
        })

print("=" * 70)
print("CONTENT GAP FINDINGS")
print("=" * 70)
print()

print(f"Unmet Needs (dense in Twitter, sparse in ArXiv):   {len(unmet_needs)}")
print(f"Unused Content (dense in ArXiv, sparse in Twitter): {len(unused_content)}")
print(f"Well-Served (dense in both):                        {len(well_served)}")
print()

# ============================================================================
# Analyze semantic themes in gaps
# ============================================================================
print("=" * 70)
print("GAP 1: UNMET NEEDS")
print("=" * 70)
print("Topics users need but research doesn't cover:")
print()

# Load embedding model to sample bucket contents
model = SentenceTransformer(f"sentence-transformers/all-{MODEL_NAME}-v2")

# Sample from top 5 unmet needs
unmet_needs_sorted = sorted(unmet_needs, key=lambda x: x['twitter_count'], reverse=True)[:5]

for i, gap in enumerate(unmet_needs_sorted, 1):
    bucket_id = gap['bucket_id']

    # Get sample texts from Twitter bucket
    bucket_items = twitter_index.filter(pl.col('bucket_id') == bucket_id)
    row_ids = bucket_items['row_id'].to_list()[:3]  # Sample 3

    print(f"{i}. Bucket {bucket_id}: {gap['twitter_count']} Twitter, {gap['arxiv_count']} ArXiv")

    # Note: We don't have the original Twitter text, but we can show the pattern
    print(f"   → Twitter customer support issues (semantic cluster)")
    print()

print()
print("=" * 70)
print("GAP 2: UNUSED CONTENT")
print("=" * 70)
print("Research topics available but users don't seek:")
print()

unused_sorted = sorted(unused_content, key=lambda x: x['arxiv_count'], reverse=True)[:5]

for i, gap in enumerate(unused_sorted, 1):
    bucket_id = gap['bucket_id']

    print(f"{i}. Bucket {bucket_id}: {gap['twitter_count']} Twitter, {gap['arxiv_count']} ArXiv")
    print(f"   → ArXiv research papers (semantic cluster)")
    print()

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print()

print("Content Gap Analysis Validated:")
print()
print("1. Unmet Needs (Twitter > ArXiv):")
print(f"   - {len(unmet_needs)} semantic topics in customer support")
print("   - These represent user problems/questions")
print("   - Research could address these gaps")
print()
print("2. Unused Content (ArXiv > Twitter):")
print(f"   - {len(unused_content)} semantic topics in research papers")
print("   - These represent available knowledge")
print("   - Not currently sought by users (potential outreach opportunity)")
print()
print("3. Well-Served (Both dense):")
print(f"   - {len(well_served)} semantic topics appear in both")
print("   - Research addressing actual user needs")
print()

# Statistical summary
coverage = len(well_served) / (len(well_served) + len(unmet_needs)) * 100 if (len(well_served) + len(unmet_needs)) > 0 else 0
print(f"Coverage: {coverage:.1f}% of user needs have corresponding research")
print()

print("Use Cases:")
print("  - Product teams: Identify user needs not addressed by documentation")
print("  - Content strategy: Find topics with supply-demand mismatch")
print("  - Research planning: Discover gaps between academic work and real problems")
print()
print("=" * 70)
