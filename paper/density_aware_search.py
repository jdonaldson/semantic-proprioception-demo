#!/usr/bin/env python3
"""
Density-Aware Semantic Search

Uses LSH bucket density to automatically refine searches:
- Sparse buckets (< 10 items): Return all candidates
- Dense buckets (≥ 10 items): Refine using hierarchical bit slicing
"""
import polars as pl
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

SEED = 31  # Our optimal seed

class DensityAwareSearch:
    def __init__(self, embeddings_df, model, num_bits=16, density_threshold=10):
        """
        Initialize density-aware search index

        Args:
            embeddings_df: Polars DataFrame with 'id' and 'embedding' columns
            model: SentenceTransformer model for query encoding
            num_bits: LSH hash bits (16 = hierarchical levels available)
            density_threshold: Max bucket size before refinement
        """
        self.embeddings_df = embeddings_df
        self.model = model
        self.num_bits = num_bits
        self.density_threshold = density_threshold

        # Extract embeddings
        self.embeddings = np.array([row for row in embeddings_df['embedding']])
        self.d = self.embeddings.shape[1]

        # Generate LSH hyperplanes (seed 31)
        np.random.seed(SEED)
        self.hyperplanes = np.random.randn(num_bits, self.d)

        # Compute LSH hashes for all documents
        hash_bits = (self.embeddings @ self.hyperplanes.T > 0).astype(int)
        powers = 2 ** np.arange(num_bits)
        self.hashes = hash_bits @ powers

        # Build index: bucket_id -> [doc_indices]
        self.index = {}
        for i, hash_val in enumerate(self.hashes & 0xFF):  # Level 0: 8 bits
            if hash_val not in self.index:
                self.index[hash_val] = []
            self.index[hash_val].append(i)

        # Compute density histogram
        bucket_sizes = [len(indices) for indices in self.index.values()]
        self.density_stats = {
            'num_buckets': len(self.index),
            'avg_size': np.mean(bucket_sizes),
            'max_size': max(bucket_sizes),
            'dense_buckets': sum(1 for s in bucket_sizes if s >= density_threshold),
            'sparse_buckets': sum(1 for s in bucket_sizes if s < density_threshold),
        }

    def search(self, query_text, k=10, verbose=False):
        """
        Search for top-k most similar documents

        Args:
            query_text: Text query
            k: Number of results to return
            verbose: Print search strategy

        Returns:
            List of (doc_id, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]

        # Compute query LSH hash
        query_hash_bits = (query_embedding @ self.hyperplanes.T > 0).astype(int)
        powers = 2 ** np.arange(self.num_bits)
        query_hash = query_hash_bits @ powers

        # Level 0: 8-bit bucket
        bucket_id = query_hash & 0xFF

        if bucket_id not in self.index:
            if verbose:
                print(f"Empty bucket {bucket_id}")
            return []

        candidates = self.index[bucket_id]
        bucket_size = len(candidates)

        # Adaptive refinement based on density
        if bucket_size >= self.density_threshold:
            # Dense bucket - refine using bit slicing
            level_1_hash = (query_hash >> 8) & 0x0F

            # Filter candidates by level 1 hash
            refined_candidates = []
            for idx in candidates:
                doc_level_1 = (self.hashes[idx] >> 8) & 0x0F
                if doc_level_1 == level_1_hash:
                    refined_candidates.append(idx)

            if verbose:
                print(f"Dense bucket {bucket_id} ({bucket_size} docs) → refined to {len(refined_candidates)} docs")

            candidates = refined_candidates if len(refined_candidates) > 0 else candidates
        else:
            if verbose:
                print(f"Sparse bucket {bucket_id} ({bucket_size} docs) - no refinement")

        # Rank candidates by cosine similarity
        candidate_embeddings = self.embeddings[candidates]
        similarities = candidate_embeddings @ query_embedding / (
            norm(candidate_embeddings, axis=1) * norm(query_embedding)
        )

        # Get top-k
        top_k_indices = np.argsort(-similarities)[:k]

        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            doc_idx = candidates[idx]
            doc_id = self.embeddings_df[doc_idx]['id']
            similarity = similarities[idx]
            results.append((doc_id, similarity))

        return results

    def print_stats(self):
        """Print index statistics"""
        print("Index Statistics:")
        print(f"  Total documents:  {len(self.embeddings):,}")
        print(f"  Total buckets:    {self.density_stats['num_buckets']}")
        print(f"  Avg bucket size:  {self.density_stats['avg_size']:.1f}")
        print(f"  Max bucket size:  {self.density_stats['max_size']}")
        print(f"  Dense buckets:    {self.density_stats['dense_buckets']} (≥{self.density_threshold})")
        print(f"  Sparse buckets:   {self.density_stats['sparse_buckets']} (<{self.density_threshold})")


# ============================================================================
# Demo with ArXiv Papers
# ============================================================================

if __name__ == "__main__":
    DEMO_DIR = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo")

    print("=" * 80)
    print("DENSITY-AWARE SEMANTIC SEARCH DEMO")
    print("=" * 80)
    print()

    # Load ArXiv papers with embeddings
    print("Loading ArXiv papers...")
    df = pl.read_parquet(DEMO_DIR / "arxiv_demo_data" / "MiniLM-L6_embeddings.parquet")
    df = df.with_columns(pl.col('arxiv_id').alias('id'))

    print(f"Loaded {len(df):,} papers")
    print()

    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print()

    # Build search index
    print("Building density-aware search index...")
    search = DensityAwareSearch(df, model, density_threshold=10)
    search.print_stats()
    print()

    # Example queries
    queries = [
        "neural networks for image classification",
        "graph algorithms and optimization",
        "machine learning interpretability",
    ]

    for query in queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        results = search.search(query, k=5, verbose=True)

        print(f"\nTop {len(results)} results:")
        for rank, (paper_id, similarity) in enumerate(results, 1):
            paper_row = df.filter(pl.col('id') == paper_id).to_dicts()[0]
            title = paper_row['title']
            category = paper_row['category_name']
            print(f"{rank}. [{category}] {title[:70]}...")
            print(f"   Similarity: {similarity:.3f}")
        print()
