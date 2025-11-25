#!/usr/bin/env python3
"""
Generate figures for semantic proprioception paper
"""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "semantic-proprioception-demo"))

# Paths to demo data (absolute paths to avoid confusion)
TWITTER_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/semantic_proprioception_data")
ARXIV_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/arxiv_demo_data")
HN_DATA = Path("/Users/jdonaldson/Projects/semantic-proprioception-demo/hackernews_demo_data")

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def model_to_filename(model: str) -> str:
    """Convert model name to filename format"""
    # Demo data uses shortened names
    if "MiniLM-L3" in model:
        return "MiniLM-L3"
    elif "MiniLM-L6" in model:
        return "MiniLM-L6"
    elif "MiniLM-L12" in model:
        return "MiniLM-L12"
    elif "MPNet" in model or "mpnet" in model:
        return "MPNet-base"
    return model.replace('/', '_')

def load_dense_buckets(data_dir: Path, model: str = "all-MiniLM-L6-v2", min_count: int = 5):
    """Load dense buckets from LSH index"""
    model_name = model_to_filename(model)
    index_file = data_dir / f"{model_name}_lsh_index.parquet"

    if not index_file.exists():
        print(f"Warning: {index_file} not found")
        return pl.DataFrame()

    # Load and compute bucket densities
    dense = (pl.scan_parquet(index_file)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .filter(pl.col('count') >= min_count)
        .sort('count', descending=True)
        .collect()
    )

    return dense

def load_all_buckets(data_dir: Path, model: str = "all-MiniLM-L6-v2"):
    """Load all bucket counts (including sparse)"""
    model_name = model_to_filename(model)
    index_file = data_dir / f"{model_name}_lsh_index.parquet"

    if not index_file.exists():
        return pl.DataFrame()

    buckets = (pl.scan_parquet(index_file)
        .group_by('bucket_id')
        .agg(pl.count('row_id').alias('count'))
        .sort('count', descending=True)
        .collect()
    )

    return buckets

def compute_bucket_coherence(data_dir: Path, embeddings_file: str, bucket_id: int, model: str = "all-MiniLM-L6-v2"):
    """Compute average cosine similarity within a bucket"""
    from numpy.linalg import norm

    # Load index to get row IDs in this bucket
    model_name = model_to_filename(model)
    index_file = data_dir / f"{model_name}_lsh_index.parquet"
    bucket_contents = (pl.scan_parquet(index_file)
        .filter(pl.col('bucket_id') == bucket_id)
        .collect()
    )

    if len(bucket_contents) < 2:
        return 0.0

    # Load embeddings
    embeddings_path = data_dir / embeddings_file
    df = pl.read_parquet(embeddings_path)

    # Get embeddings for items in bucket
    row_ids = bucket_contents['row_id'].to_list()
    embeddings = np.array([df[row_id, 'embedding'] for row_id in row_ids])

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]))
            similarities.append(cos_sim)

    return np.mean(similarities) if similarities else 0.0

def figure1_bucket_density_distribution():
    """Figure 1: Histogram of bucket densities across datasets"""
    print("Generating Figure 1: Bucket Density Distribution...")

    model = "all-MiniLM-L6-v2"

    # Load bucket counts for all three datasets
    twitter_buckets = load_all_buckets(TWITTER_DATA, model)
    arxiv_buckets = load_all_buckets(ARXIV_DATA, model)
    hn_buckets = load_all_buckets(HN_DATA, model)

    # Create histogram bins
    bins = [1, 2, 5, 10, 20, 50]

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = [
        ("Twitter", twitter_buckets, '#1f77b4'),
        ("ArXiv", arxiv_buckets, '#ff7f0e'),
        ("Hacker News", hn_buckets, '#2ca02c')
    ]

    x = np.arange(len(bins) - 1)
    width = 0.25

    for i, (name, buckets, color) in enumerate(datasets):
        if len(buckets) == 0:
            continue

        counts = buckets['count'].to_list()
        hist, _ = np.histogram(counts, bins=bins)

        ax.bar(x + i*width, hist, width, label=name, color=color, alpha=0.8)

    ax.set_xlabel('Bucket Size (items)', fontsize=12)
    ax.set_ylabel('Number of Buckets', fontsize=12)
    ax.set_title('LSH Bucket Density Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure1_density_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure1_density_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to {OUTPUT_DIR / 'figure1_density_distribution.pdf'}")
    plt.close()

def figure2_coherence_vs_density():
    """Figure 2: Bucket density vs semantic coherence"""
    print("Generating Figure 2: Coherence vs Density...")

    model = "all-MiniLM-L6-v2"

    # Sample buckets and compute coherence
    datasets = [
        ("Twitter", TWITTER_DATA, "MiniLM-L6_embeddings.parquet", '#1f77b4'),
        ("ArXiv", ARXIV_DATA, "MiniLM-L6_embeddings.parquet", '#ff7f0e'),
        ("Hacker News", HN_DATA, "MiniLM-L6_embeddings.parquet", '#2ca02c')
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data_dir, embeddings_file, color in datasets:
        buckets = load_all_buckets(data_dir, model)

        if len(buckets) == 0:
            continue

        # Sample buckets with various densities
        sampled = buckets.filter(pl.col('count') >= 2).sample(min(30, len(buckets)))

        densities = []
        coherences = []

        for row in sampled.iter_rows(named=True):
            bucket_id = row['bucket_id']
            count = row['count']

            coherence = compute_bucket_coherence(data_dir, embeddings_file, bucket_id, model)

            densities.append(count)
            coherences.append(coherence)

        ax.scatter(densities, coherences, label=name, color=color, alpha=0.6, s=50)

    ax.set_xlabel('Bucket Density (number of items)', fontsize=12)
    ax.set_ylabel('Intra-bucket Cosine Similarity', fontsize=12)
    ax.set_title('Dense Buckets Show Higher Semantic Coherence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Add trend line for all points combined
    all_densities = []
    all_coherences = []
    for name, data_dir, embeddings_file, color in datasets:
        buckets = load_all_buckets(data_dir, model)
        if len(buckets) == 0:
            continue
        sampled = buckets.filter(pl.col('count') >= 2).sample(min(30, len(buckets)))
        for row in sampled.iter_rows(named=True):
            coherence = compute_bucket_coherence(data_dir, embeddings_file, row['bucket_id'], model)
            all_densities.append(row['count'])
            all_coherences.append(coherence)

    if all_densities:
        z = np.polyfit(all_densities, all_coherences, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(all_densities), max(all_densities), 100)
        ax.plot(x_trend, p(x_trend), "k--", alpha=0.3, linewidth=2, label='Trend')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure2_coherence_vs_density.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure2_coherence_vs_density.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to {OUTPUT_DIR / 'figure2_coherence_vs_density.pdf'}")
    plt.close()

def figure3_composability():
    """Figure 3: Composability - bucket overlap across datasets"""
    print("Generating Figure 3: Composability...")

    model = "all-MiniLM-L6-v2"

    # Load dense buckets for each dataset
    twitter_buckets = set(load_dense_buckets(TWITTER_DATA, model)['bucket_id'].to_list())
    arxiv_buckets = set(load_dense_buckets(ARXIV_DATA, model)['bucket_id'].to_list())
    hn_buckets = set(load_dense_buckets(HN_DATA, model)['bucket_id'].to_list())

    # Compute overlaps
    twitter_only = len(twitter_buckets - arxiv_buckets - hn_buckets)
    arxiv_only = len(arxiv_buckets - twitter_buckets - hn_buckets)
    hn_only = len(hn_buckets - twitter_buckets - arxiv_buckets)

    twitter_arxiv = len(twitter_buckets & arxiv_buckets - hn_buckets)
    twitter_hn = len(twitter_buckets & hn_buckets - arxiv_buckets)
    arxiv_hn = len(arxiv_buckets & hn_buckets - twitter_buckets)

    all_three = len(twitter_buckets & arxiv_buckets & hn_buckets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar chart
    datasets = ['Twitter', 'ArXiv', 'Hacker News']
    exclusive = [twitter_only, arxiv_only, hn_only]
    shared = [
        len(twitter_buckets & (arxiv_buckets | hn_buckets)),
        len(arxiv_buckets & (twitter_buckets | hn_buckets)),
        len(hn_buckets & (twitter_buckets | arxiv_buckets))
    ]

    x = np.arange(len(datasets))
    width = 0.6

    ax1.bar(x, exclusive, width, label='Exclusive buckets', color='#1f77b4', alpha=0.8)
    ax1.bar(x, shared, width, bottom=exclusive, label='Shared buckets', color='#ff7f0e', alpha=0.8)

    ax1.set_ylabel('Number of Dense Buckets', fontsize=12)
    ax1.set_title('Compositional Indexing: Bucket Overlap', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right: Overlap matrix heatmap
    overlap_matrix = np.array([
        [len(twitter_buckets), twitter_arxiv + all_three, twitter_hn + all_three],
        [twitter_arxiv + all_three, len(arxiv_buckets), arxiv_hn + all_three],
        [twitter_hn + all_three, arxiv_hn + all_three, len(hn_buckets)]
    ])

    im = ax2.imshow(overlap_matrix, cmap='Blues', aspect='auto')

    ax2.set_xticks(np.arange(len(datasets)))
    ax2.set_yticks(np.arange(len(datasets)))
    ax2.set_xticklabels(datasets)
    ax2.set_yticklabels(datasets)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            text = ax2.text(j, i, overlap_matrix[i, j],
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    ax2.set_title('Dense Bucket Overlap Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Number of buckets')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure3_composability.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure3_composability.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to {OUTPUT_DIR / 'figure3_composability.pdf'}")
    plt.close()

def main():
    print("Generating figures for semantic proprioception paper...\n")

    figure1_bucket_density_distribution()
    figure2_coherence_vs_density()
    figure3_composability()

    print("\nAll figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
