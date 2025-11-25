#!/usr/bin/env python3
"""
Benchmark: Krapivin O(1) density queries vs. Polars group_by

Compares performance of finding dense buckets using:
1. Polars group_by + filter (what we currently use)
2. Krapivin hash table with O(1) density queries (what we claim)
"""
import polars as pl
import numpy as np
import time
from pathlib import Path
import sys

# Add krapivin-python to path
sys.path.append("/Users/jdonaldson/Projects/semantic-proprioception-demo/krapivin-python")

try:
    from krapivin_hash_rs import PyLSHIndex
    HAS_KRAPIVIN = True
except ImportError:
    print("⚠️  krapivin_hash_rs not available. Run: maturin develop")
    HAS_KRAPIVIN = False
    sys.exit(1)

# Data paths
TWITTER_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/semantic_proprioception_data")
MODEL = "MiniLM-L6"

print("=" * 70)
print("KRAPIVIN vs POLARS BENCHMARK: O(1) Density Queries")
print("=" * 70)
print()

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
embeddings_file = TWITTER_DATA / f"{MODEL}_embeddings.parquet"
index_file = TWITTER_DATA / f"{MODEL}_lsh_index.parquet"

df_embeddings = pl.read_parquet(embeddings_file)
df_index = pl.read_parquet(index_file)

n_embeddings = len(df_embeddings)
n_entries = len(df_index)

print(f"  {n_embeddings:,} embeddings")
print(f"  {n_entries:,} index entries")
print()

# ============================================================================
# METHOD 1: Polars group_by (current approach)
# ============================================================================
print("METHOD 1: Polars group_by + filter")
print("-" * 70)

# Warm up
_ = (pl.scan_parquet(index_file)
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
    .filter(pl.col('count') >= 5)
    .collect()
)

# Benchmark
n_trials = 100
times_polars = []

for _ in range(n_trials):
    start = time.perf_counter()

    dense = (pl.scan_parquet(index_file)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= 5)
        .collect()
    )

    elapsed = time.perf_counter() - start
    times_polars.append(elapsed)

avg_polars = np.mean(times_polars) * 1000  # ms
std_polars = np.std(times_polars) * 1000

print(f"  Query time: {avg_polars:.2f} ± {std_polars:.2f} ms ({n_trials} trials)")
print(f"  Found: {len(dense)} dense buckets")
print()

# ============================================================================
# METHOD 2: Krapivin hash table
# ============================================================================
print("METHOD 2: Krapivin hash table (PyLSHIndex)")
print("-" * 70)

# Build index
print("  Building Krapivin index...")
build_start = time.perf_counter()

krapivin_index = PyLSHIndex(
    seed=12345,
    num_bits=8,
    embedding_dim=384,
    capacity=n_embeddings,
    delta=0.3
)

# Add all embeddings
for row in df_embeddings.iter_rows(named=True):
    embedding = row['embedding']
    row_id = row['row_id'] if 'row_id' in row else row.get('id', 0)
    krapivin_index.add_embedding(embedding, "twitter", row_id)

build_time = time.perf_counter() - build_start
print(f"  Build time: {build_time * 1000:.2f} ms")

# Benchmark density queries
times_krapivin = []

for _ in range(n_trials):
    start = time.perf_counter()

    dense_krapivin = krapivin_index.get_dense_buckets(min_count=5)

    elapsed = time.perf_counter() - start
    times_krapivin.append(elapsed)

avg_krapivin = np.mean(times_krapivin) * 1000  # ms
std_krapivin = np.std(times_krapivin) * 1000

print(f"  Query time: {avg_krapivin:.2f} ± {std_krapivin:.2f} ms ({n_trials} trials)")
print(f"  Found: {len(dense_krapivin)} dense buckets")
print()

# ============================================================================
# RESULTS
# ============================================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

speedup = avg_polars / avg_krapivin

print(f"Polars group_by:    {avg_polars:8.2f} ± {std_polars:6.2f} ms")
print(f"Krapivin O(1):      {avg_krapivin:8.2f} ± {std_krapivin:6.2f} ms")
print()
print(f"Speedup:            {speedup:.2f}x")
print()

if speedup > 1.5:
    print(f"✓ Krapivin is {speedup:.1f}x FASTER than Polars")
    print("  → O(1) density queries validated!")
elif speedup > 1.0:
    print(f"✓ Krapivin is marginally faster ({speedup:.2f}x)")
elif speedup > 0.5:
    print(f"⚠️  Krapivin is similar speed ({speedup:.2f}x)")
    print("  → Overhead from Rust/Python boundary?")
else:
    print(f"❌ Krapivin is SLOWER ({speedup:.2f}x)")
    print("  → Polars columnar optimization dominates")

print()

# ============================================================================
# SCALABILITY TEST: How does it scale with dataset size?
# ============================================================================
print("=" * 70)
print("SCALABILITY TEST")
print("=" * 70)
print()

print("Simulating larger datasets by replicating data...")
print()

scale_factors = [1, 2, 5, 10]
polars_times = []
krapivin_times = []

for scale in scale_factors:
    # Replicate Parquet data
    if scale > 1:
        df_scaled = pl.concat([df_index] * scale)
        temp_file = f"/tmp/scaled_index_{scale}x.parquet"
        df_scaled.write_parquet(temp_file)
        index_path = temp_file
    else:
        index_path = str(index_file)

    # Polars
    start = time.perf_counter()
    dense = (pl.scan_parquet(index_path)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= 5)
        .collect()
    )
    polars_time = (time.perf_counter() - start) * 1000
    polars_times.append(polars_time)

    # Krapivin (re-index with scaled data)
    krapivin_scaled = PyLSHIndex(
        seed=12345,
        num_bits=8,
        embedding_dim=384,
        capacity=n_embeddings * scale,
        delta=0.3
    )

    for _ in range(scale):
        for row in df_embeddings.iter_rows(named=True):
            embedding = row['embedding']
            row_id = row.get('row_id', row.get('id', 0))
            krapivin_scaled.add_embedding(embedding, "twitter", row_id)

    start = time.perf_counter()
    dense_k = krapivin_scaled.get_dense_buckets(min_count=5)
    krapivin_time = (time.perf_counter() - start) * 1000
    krapivin_times.append(krapivin_time)

    print(f"  {scale:2d}x ({n_embeddings * scale:5d} items): "
          f"Polars {polars_time:6.2f} ms, "
          f"Krapivin {krapivin_time:6.2f} ms "
          f"(speedup: {polars_time / krapivin_time:.2f}x)")

print()

# Check if Krapivin scales better (should be flatter curve)
polars_growth = polars_times[-1] / polars_times[0]
krapivin_growth = krapivin_times[-1] / krapivin_times[0]

print(f"Growth (1x → {scale_factors[-1]}x):")
print(f"  Polars:   {polars_growth:.2f}x slower")
print(f"  Krapivin: {krapivin_growth:.2f}x slower")
print()

if krapivin_growth < polars_growth:
    print(f"✓ Krapivin scales BETTER ({krapivin_growth:.2f}x vs {polars_growth:.2f}x)")
    print("  → O(1) behavior confirmed at scale")
else:
    print(f"⚠️  Similar scaling ({krapivin_growth:.2f}x vs {polars_growth:.2f}x)")

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
