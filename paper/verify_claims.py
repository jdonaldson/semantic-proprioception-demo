#!/usr/bin/env python3
"""
Verify all empirical claims in the paper against actual data
"""
import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict

# Data paths
TWITTER_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/semantic_proprioception_data")
ARXIV_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/arxiv_demo_data")
HN_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/hackernews_demo_data")

MODEL = "MiniLM-L6"

print("=" * 70)
print("VERIFYING EMPIRICAL CLAIMS IN SEMANTIC PROPRIOCEPTION PAPER")
print("=" * 70)
print()

# ============================================================================
# CLAIM 1: Dataset sizes
# ============================================================================
print("CLAIM 1: Dataset Sizes")
print("-" * 70)

for name, data_dir in [("Twitter", TWITTER_DATA), ("ArXiv", ARXIV_DATA), ("Hacker News", HN_DATA)]:
    emb_file = data_dir / f"{MODEL}_embeddings.parquet"
    if emb_file.exists():
        df = pl.read_parquet(emb_file)
        print(f"{name:15s}: {len(df):4d} items")
    else:
        print(f"{name:15s}: FILE NOT FOUND")

print()

# ============================================================================
# CLAIM 2: Dense bucket counts (≥5 items)
# ============================================================================
print("CLAIM 2: Dense Bucket Counts (threshold ≥5)")
print("-" * 70)

dense_bucket_data = {}

for name, data_dir in [("Twitter", TWITTER_DATA), ("ArXiv", ARXIV_DATA), ("Hacker News", HN_DATA)]:
    index_file = data_dir / f"{MODEL}_lsh_index.parquet"

    if not index_file.exists():
        print(f"{name:15s}: INDEX NOT FOUND")
        continue

    # Count buckets by density
    buckets = (pl.scan_parquet(index_file)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .collect()
    )

    total_buckets = len(buckets)
    dense_5 = len(buckets.filter(pl.col('count') >= 5))
    dense_10 = len(buckets.filter(pl.col('count') >= 10))

    max_size = buckets['count'].max()
    avg_size = buckets['count'].mean()

    dense_bucket_data[name] = buckets['bucket_id'].to_list()

    print(f"{name:15s}: {total_buckets:3d} total buckets, {dense_5:3d} dense (≥5), {dense_10:3d} very dense (≥10)")
    print(f"                 Max bucket size: {max_size}, Avg: {avg_size:.1f}")

print()

# ============================================================================
# CLAIM 3: Composability - bucket overlap across datasets
# ============================================================================
print("CLAIM 3: Composability (Bucket Overlap)")
print("-" * 70)

if len(dense_bucket_data) == 3:
    twitter_buckets = set(dense_bucket_data.get("Twitter", []))
    arxiv_buckets = set(dense_bucket_data.get("ArXiv", []))
    hn_buckets = set(dense_bucket_data.get("Hacker News", []))

    # Get dense buckets (≥5) specifically
    twitter_dense = set()
    arxiv_dense = set()
    hn_dense = set()

    for name, data_dir, bucket_set in [
        ("Twitter", TWITTER_DATA, twitter_dense),
        ("ArXiv", ARXIV_DATA, arxiv_dense),
        ("Hacker News", HN_DATA, hn_dense)
    ]:
        index_file = data_dir / f"{MODEL}_lsh_index.parquet"
        dense = (pl.scan_parquet(index_file)
            .group_by('bucket_id')
            .agg(pl.count('row_id').alias('count'))
            .filter(pl.col('count') >= 5)
            .collect()
        )
        bucket_set.update(dense['bucket_id'].to_list())

    twitter_only = twitter_dense - arxiv_dense - hn_dense
    arxiv_only = arxiv_dense - twitter_dense - hn_dense
    hn_only = hn_dense - twitter_dense - arxiv_dense

    twitter_arxiv = twitter_dense & arxiv_dense - hn_dense
    twitter_hn = twitter_dense & hn_dense - arxiv_dense
    arxiv_hn = arxiv_dense & hn_dense - twitter_dense
    all_three = twitter_dense & arxiv_dense & hn_dense

    print(f"Twitter dense buckets:        {len(twitter_dense)}")
    print(f"ArXiv dense buckets:          {len(arxiv_dense)}")
    print(f"Hacker News dense buckets:    {len(hn_dense)}")
    print()
    print(f"Twitter only:                 {len(twitter_only)}")
    print(f"ArXiv only:                   {len(arxiv_only)}")
    print(f"Hacker News only:             {len(hn_only)}")
    print()
    print(f"Twitter ∩ ArXiv:              {len(twitter_arxiv)}")
    print(f"Twitter ∩ Hacker News:        {len(twitter_hn)}")
    print(f"ArXiv ∩ Hacker News:          {len(arxiv_hn)}")
    print(f"All three datasets:           {len(all_three)}")

    if all_three:
        print(f"\nShared bucket IDs: {sorted(all_three)}")

print()

# ============================================================================
# CLAIM 4: Discovered themes (examples from paper)
# ============================================================================
print("CLAIM 4: Example Themes from Paper")
print("-" * 70)
print("Paper claims examples like:")
print("  - Twitter bucket 181: login/password issues")
print("  - ArXiv bucket 142: neural attention/transformers")
print()
print("Checking if these buckets actually exist and are dense...")
print()

for name, data_dir, bucket_id in [
    ("Twitter", TWITTER_DATA, 181),
    ("ArXiv", ARXIV_DATA, 142)
]:
    index_file = data_dir / f"{MODEL}_lsh_index.parquet"

    if not index_file.exists():
        print(f"{name} bucket {bucket_id}: INDEX NOT FOUND")
        continue

    bucket_contents = (pl.scan_parquet(index_file)
        .filter(pl.col('bucket_id') == bucket_id)
        .collect()
    )

    if len(bucket_contents) == 0:
        print(f"{name} bucket {bucket_id}: DOES NOT EXIST")
    else:
        print(f"{name} bucket {bucket_id}: EXISTS with {len(bucket_contents)} items")

print()

# ============================================================================
# CLAIM 5: LSH configuration
# ============================================================================
print("CLAIM 5: LSH Configuration")
print("-" * 70)
print("Paper claims: 8-bit LSH signatures (256 possible buckets), seed=12345")
print()

# Check total unique buckets across all datasets
all_buckets = set()
for name, data_dir in [("Twitter", TWITTER_DATA), ("ArXiv", ARXIV_DATA), ("Hacker News", HN_DATA)]:
    index_file = data_dir / f"{MODEL}_lsh_index.parquet"
    if index_file.exists():
        buckets = (pl.scan_parquet(index_file)
            .select('bucket_id')
            .unique()
            .collect()
        )
        all_buckets.update(buckets['bucket_id'].to_list())

print(f"Total unique buckets observed: {len(all_buckets)}")
print(f"Max bucket ID: {max(all_buckets) if all_buckets else 'N/A'}")
print(f"Min bucket ID: {min(all_buckets) if all_buckets else 'N/A'}")
print()
print(f"Expected with 8 bits: 256 possible buckets (0-255)")

if max(all_buckets) >= 256:
    print("⚠️  WARNING: Bucket IDs exceed 255 - may not be 8-bit LSH!")

print()

# ============================================================================
# CLAIM 6: Model comparison
# ============================================================================
print("CLAIM 6: Model Comparison (different embedding models)")
print("-" * 70)

models = ["MiniLM-L3", "MiniLM-L6", "MiniLM-L12", "MPNet-base"]
model_results = defaultdict(dict)

for model in models:
    for name, data_dir in [("Twitter", TWITTER_DATA), ("ArXiv", ARXIV_DATA), ("HN", HN_DATA)]:
        index_file = data_dir / f"{model}_lsh_index.parquet"

        if not index_file.exists():
            continue

        dense = (pl.scan_parquet(index_file)
            .group_by('bucket_id')
            .agg(pl.count('row_id').alias('count'))
            .filter(pl.col('count') >= 5)
            .collect()
        )

        model_results[model][name] = len(dense)

print(f"{'Model':<15s} {'Twitter':<10s} {'ArXiv':<10s} {'HN':<10s}")
print("-" * 50)
for model in models:
    tw = model_results[model].get('Twitter', '-')
    ar = model_results[model].get('ArXiv', '-')
    hn = model_results[model].get('HN', '-')
    print(f"{model:<15s} {str(tw):<10s} {str(ar):<10s} {str(hn):<10s}")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
