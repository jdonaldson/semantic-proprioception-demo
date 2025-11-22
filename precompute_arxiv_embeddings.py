#!/usr/bin/env python3
"""
Semantic Proprioception - Pre-compute ArXiv paper embeddings

Generates embeddings for ArXiv abstracts using 4 different models.
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add krapivin-python to path
sys.path.insert(0, str(Path(__file__).parent / "krapivin-python"))
from lsh_parquet import save_index_to_parquet
from krapivin_hash_rs import PyLSHIndex


# Model configurations (same as Twitter demo for consistency)
MODELS = [
    {
        "name": "MiniLM-L3",
        "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "dim": 384,
        "description": "Fastest, smallest (61MB) - good for quick similarity",
        "lsh_bits": 8,
    },
    {
        "name": "MiniLM-L6",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Balanced speed/quality (90MB) - general purpose",
        "lsh_bits": 8,
    },
    {
        "name": "MiniLM-L12",
        "model_id": "sentence-transformers/all-MiniLM-L12-v2",
        "dim": 384,
        "description": "Better quality (120MB) - more accurate",
        "lsh_bits": 8,
    },
    {
        "name": "MPNet-base",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "description": "Highest quality (420MB) - best semantic understanding",
        "lsh_bits": 10,
    },
]


def load_arxiv_data(csv_path):
    """Load ArXiv papers"""
    print(f"Loading data from {csv_path}...")

    df = pl.read_csv(csv_path)
    print(f"  Total papers: {len(df):,}")

    # Show category distribution
    print("\n  Papers by category:")
    category_counts = df.group_by("category_name").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in category_counts.head(5).to_dicts():
        print(f"    {row['category_name']}: {row['count']}")

    return df


def embed_abstracts(abstracts, model_config):
    """Embed abstracts using specified model"""
    print(f"\n{'='*70}")
    print(f"Model: {model_config['name']}")
    print(f"  {model_config['description']}")
    print(f"{'='*70}\n")

    # Load model
    print(f"Loading {model_config['model_id']}...")
    model = SentenceTransformer(model_config['model_id'])

    # Embed
    print(f"Embedding {len(abstracts):,} abstracts...")
    print(f"  (This may take a few minutes...)")

    embeddings = model.encode(
        abstracts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"  ✓ Shape: {embeddings.shape}")
    print(f"  ✓ Dimension: {model_config['dim']}")

    return embeddings


def build_lsh_index(embeddings, model_config):
    """Build LSH index for embeddings"""
    print(f"\nBuilding LSH index...")

    index = PyLSHIndex(
        seed=12345,  # Same seed as Twitter for consistency
        num_bits=model_config['lsh_bits'],
        embedding_dim=model_config['dim'],
        capacity=10000,
        delta=0.3,
    )

    # Add all embeddings
    for i, embedding in enumerate(embeddings):
        index.add_embedding(
            embedding.tolist(),
            f"{model_config['name']}.parquet",
            i
        )

    stats = index.stats()
    print(f"  ✓ Indexed {len(embeddings):,} embeddings")
    print(f"  ✓ Buckets used: {stats.num_buckets}")
    print(f"  ✓ Load factor: {stats.load_factor:.2%}")

    # Get bucket statistics
    histogram = index.bucket_size_histogram()
    dense_buckets = index.get_dense_buckets(min_count=5)

    print(f"  ✓ Dense buckets (≥5 papers): {len(dense_buckets)}")

    if histogram:
        max_size = max(histogram.keys())
        print(f"  ✓ Largest bucket: {max_size} papers")

    return index, stats


def save_model_data(model_config, df, embeddings, index):
    """Save embeddings and index to Parquet"""
    print(f"\nSaving to Parquet...")

    output_dir = Path("arxiv_demo_data")
    output_dir.mkdir(exist_ok=True)

    model_name = model_config['name']

    # Save embeddings with metadata
    embeddings_df = df.select([
        pl.col("arxiv_id"),
        pl.col("title"),
        pl.col("abstract"),
        pl.col("category_name"),
        pl.col("published"),
    ]).with_columns([
        pl.Series("embedding", embeddings.tolist()),
    ])

    embeddings_path = output_dir / f"{model_name}_embeddings.parquet"
    embeddings_df.write_parquet(embeddings_path, compression="zstd")

    # Save LSH index
    index_path = output_dir / f"{model_name}_lsh_index.parquet"
    save_index_to_parquet(index, str(index_path))

    # Check file sizes
    emb_size = embeddings_path.stat().st_size / 1024 / 1024
    idx_size = index_path.stat().st_size / 1024 / 1024

    print(f"  ✓ Embeddings: {embeddings_path.name} ({emb_size:.1f} MB)")
    print(f"  ✓ LSH Index: {index_path.name} ({idx_size:.1f} MB)")
    print(f"  ✓ Total: {emb_size + idx_size:.1f} MB")

    return embeddings_path, index_path


def save_metadata(models_info):
    """Save model metadata for demo app"""
    print(f"\n{'='*70}")
    print("Saving model metadata...")
    print(f"{'='*70}\n")

    output_dir = Path("arxiv_demo_data")

    metadata = pl.DataFrame({
        "model_name": [m["name"] for m in models_info],
        "model_id": [m["model_id"] for m in models_info],
        "dimension": [m["dim"] for m in models_info],
        "lsh_bits": [m["lsh_bits"] for m in models_info],
        "description": [m["description"] for m in models_info],
        "num_buckets": [m["stats"]["num_buckets"] for m in models_info],
        "load_factor": [m["stats"]["load_factor"] for m in models_info],
        "dense_buckets": [m["stats"]["dense_buckets"] for m in models_info],
    })

    metadata_path = output_dir / "models_metadata.parquet"
    metadata.write_parquet(metadata_path)

    print(f"  ✓ Saved: {metadata_path.name}")
    print(f"\nModel comparison:")
    print(metadata)


def main():
    print("="*70)
    print("Semantic Proprioception - Pre-computing ArXiv Embeddings")
    print("="*70)
    print()

    # Configuration
    CSV_PATH = "arxiv_data/arxiv_papers.csv"

    if not Path(CSV_PATH).exists():
        print(f"❌ Dataset not found at {CSV_PATH}")
        print("   Run: python fetch_arxiv_data.py")
        return

    # Load data
    df = load_arxiv_data(CSV_PATH)
    abstracts = df.get_column("abstract").to_list()

    # Show samples
    print("\nSample papers:")
    for i, row in enumerate(df.head(3).to_dicts(), 1):
        print(f"\n  {i}. [{row['category_name']}] {row['title'][:60]}...")
        print(f"     Abstract: {row['abstract'][:100]}...")
    print()

    # Process each model
    models_info = []

    for model_config in MODELS:
        # Embed
        embeddings = embed_abstracts(abstracts, model_config)

        # Build LSH index
        index, stats = build_lsh_index(embeddings, model_config)

        # Get dense buckets count
        dense_buckets = index.get_dense_buckets(min_count=5)

        # Save
        save_model_data(model_config, df, embeddings, index)

        # Track info
        models_info.append({
            **model_config,
            "stats": {
                "num_buckets": stats.num_buckets,
                "load_factor": stats.load_factor,
                "dense_buckets": len(dense_buckets),
            }
        })

    # Save metadata
    save_metadata(models_info)

    # Final summary
    print(f"\n{'='*70}")
    print("✓ Pre-computation Complete!")
    print(f"{'='*70}\n")

    output_dir = Path("arxiv_demo_data")
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    total_size_mb = total_size / 1024 / 1024

    print(f"Output directory: {output_dir}/")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Models: {len(MODELS)}")
    print(f"Papers: {len(df):,}")
    print()
    print("Next step: Update semantic_proprioception_demo.py")
    print("  Add dataset selector for Twitter vs ArXiv")
    print()


if __name__ == "__main__":
    main()
