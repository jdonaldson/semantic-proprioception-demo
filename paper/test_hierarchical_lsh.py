#!/usr/bin/env python3
"""
Test hierarchical LSH subdivision for dense bucket refinement
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import sys

# Paths to demo data
TWITTER_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/semantic_proprioception_data")
MODEL = "MiniLM-L6"

def lsh_hash(embeddings, num_bits=8, seed=12345):
    """
    Compute LSH hash for embeddings using random hyperplanes

    Args:
        embeddings: np.array of shape (n, d)
        num_bits: Number of hash bits
        seed: Random seed

    Returns:
        np.array of hash values (integers)
    """
    np.random.seed(seed)
    d = embeddings.shape[1]

    # Generate random hyperplanes
    hyperplanes = np.random.randn(num_bits, d)

    # Compute hash: sign of dot product
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)

    # Convert binary to integer
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values

def compute_intra_bucket_similarity(embeddings):
    """Compute average pairwise cosine similarity within bucket"""
    if len(embeddings) < 2:
        return 1.0

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
            similarities.append(cos_sim)

    return np.mean(similarities)

def subdivide_bucket_hierarchical(embeddings, depth=0, max_depth=2, min_size=3):
    """
    Recursively subdivide bucket using additional LSH projections

    Args:
        embeddings: Items in current bucket
        depth: Current recursion depth
        max_depth: Maximum subdivisions
        min_size: Minimum bucket size to consider subdividing

    Returns:
        List of (sub_bucket_embeddings, coherence_score) tuples
    """
    coherence = compute_intra_bucket_similarity(embeddings)

    # Base cases
    if len(embeddings) < min_size * 2 or depth >= max_depth:
        return [(embeddings, coherence)]

    # Apply LSH with different seed at each level
    seed = 12345 + depth * 1000
    num_bits = 4  # Use 4 bits for subdivision (16 sub-buckets)

    sub_hash = lsh_hash(embeddings, num_bits=num_bits, seed=seed)

    # Group by sub-bucket
    sub_buckets = {}
    for i, hash_val in enumerate(sub_hash):
        if hash_val not in sub_buckets:
            sub_buckets[hash_val] = []
        sub_buckets[hash_val].append(i)

    # Recursively subdivide sub-buckets
    refined = []
    for indices in sub_buckets.values():
        sub_embeddings = embeddings[indices]
        if len(sub_embeddings) >= min_size:
            refined.extend(subdivide_bucket_hierarchical(sub_embeddings, depth+1, max_depth, min_size))

    return refined

def main():
    print("Testing Hierarchical LSH Subdivision\n")
    print("=" * 60)

    # Load embeddings
    embeddings_file = TWITTER_DATA / f"{MODEL}_embeddings.parquet"
    df = pl.read_parquet(embeddings_file)
    all_embeddings = np.array([row for row in df['embedding']])

    print(f"Loaded {len(all_embeddings)} embeddings from {embeddings_file.name}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")

    # Load LSH index to find dense buckets
    index_file = TWITTER_DATA / f"{MODEL}_lsh_index.parquet"
    buckets_df = (pl.scan_parquet(index_file)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= 10)  # Dense buckets
        .sort('count', descending=True)
        .collect()
    )

    print(f"\nFound {len(buckets_df)} dense buckets (≥10 items)\n")

    # Test hierarchical subdivision on top 5 dense buckets
    results = []

    for row in buckets_df.head(5).iter_rows(named=True):
        bucket_id = row['bucket_id']
        count = row['count']

        # Get embeddings in this bucket
        bucket_contents = (pl.scan_parquet(index_file)
            .filter(pl.col('bucket_id') == bucket_id)
            .collect()
        )

        row_ids = bucket_contents['row_id'].to_list()
        bucket_embeddings = all_embeddings[row_ids]

        # Compute original coherence
        original_coherence = compute_intra_bucket_similarity(bucket_embeddings)

        # Apply hierarchical subdivision
        sub_buckets = subdivide_bucket_hierarchical(bucket_embeddings, max_depth=2, min_size=3)

        # Compute average coherence of sub-buckets
        sub_coherences = [coh for _, coh in sub_buckets]
        avg_sub_coherence = np.mean(sub_coherences)

        # Weighted average (by sub-bucket size)
        weighted_coherence = np.average(
            [coh for _, coh in sub_buckets],
            weights=[len(embs) for embs, _ in sub_buckets]
        )

        improvement = weighted_coherence - original_coherence

        results.append({
            'bucket_id': bucket_id,
            'original_size': count,
            'num_sub_buckets': len(sub_buckets),
            'original_coherence': original_coherence,
            'avg_sub_coherence': avg_sub_coherence,
            'weighted_coherence': weighted_coherence,
            'improvement': improvement
        })

        print(f"Bucket {bucket_id} (size={count}):")
        print(f"  Original coherence:       {original_coherence:.3f}")
        print(f"  Sub-buckets created:      {len(sub_buckets)}")
        print(f"  Sub-bucket sizes:         {[len(e) for e, _ in sub_buckets]}")
        print(f"  Avg sub-bucket coherence: {avg_sub_coherence:.3f}")
        print(f"  Weighted coherence:       {weighted_coherence:.3f}")
        print(f"  Improvement:              {improvement:+.3f}")
        print()

    # Summary statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    improvements = [r['improvement'] for r in results]
    avg_improvement = np.mean(improvements)
    positive_improvements = sum(1 for i in improvements if i > 0)

    print(f"Average coherence improvement: {avg_improvement:+.3f}")
    print(f"Buckets improved: {positive_improvements}/{len(results)}")
    print(f"Average sub-buckets per bucket: {np.mean([r['num_sub_buckets'] for r in results]):.1f}")

    # Save detailed results
    results_df = pl.DataFrame(results)
    output_file = Path(__file__).parent / "hierarchical_lsh_results.csv"
    results_df.write_csv(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
