#!/usr/bin/env python3
"""
Fetch ArXiv papers for semantic proprioception demo

Downloads recent papers from multiple categories to show diverse research themes.
"""

import arxiv
import polars as pl
from pathlib import Path
from datetime import datetime

# Categories to sample (diverse research areas)
CATEGORIES = {
    'cs.AI': 'Artificial Intelligence',
    'cs.LG': 'Machine Learning',
    'cs.CL': 'Computation and Language (NLP)',
    'cs.CV': 'Computer Vision',
    'physics.gen-ph': 'General Physics',
    'math.CO': 'Combinatorics',
    'q-bio.GN': 'Genomics',
    'stat.ML': 'Machine Learning (Stats)',
    'econ.EM': 'Econometrics',
    'astro-ph.GA': 'Astrophysics of Galaxies',
}

PAPERS_PER_CATEGORY = 100
OUTPUT_DIR = Path("arxiv_data")


def fetch_papers_by_category(category, max_results=100):
    """Fetch recent papers from a specific ArXiv category"""
    print(f"Fetching {max_results} papers from {category}...")

    client = arxiv.Client()
    search = arxiv.Search(
        query=f'cat:{category}',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        papers.append({
            'arxiv_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'abstract': result.summary.replace('\n', ' ').strip(),
            'category': category,
            'category_name': CATEGORIES[category],
            'published': result.published.isoformat(),
            'authors': ', '.join([a.name for a in result.authors[:3]]),  # First 3 authors
        })

    print(f"  Retrieved {len(papers)} papers")
    return papers


def main():
    print("="*70)
    print("Fetching ArXiv Papers for Semantic Proprioception Demo")
    print("="*70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Fetch papers from all categories
    all_papers = []

    for category in CATEGORIES.keys():
        papers = fetch_papers_by_category(category, PAPERS_PER_CATEGORY)
        all_papers.extend(papers)

    # Convert to DataFrame
    df = pl.DataFrame(all_papers)

    print(f"\n{'='*70}")
    print(f"Total papers fetched: {len(df):,}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"{'='*70}\n")

    # Show statistics
    print("Papers per category:")
    category_counts = df.group_by("category_name").agg(pl.len().alias("count")).sort("count", descending=True)
    print(category_counts)

    print("\nAbstract length statistics:")
    df = df.with_columns(
        pl.col("abstract").str.len_chars().alias("abstract_length")
    )

    print(f"  Min: {df['abstract_length'].min()} chars")
    print(f"  Max: {df['abstract_length'].max()} chars")
    print(f"  Mean: {df['abstract_length'].mean():.0f} chars")
    print(f"  Median: {df['abstract_length'].median():.0f} chars")

    # Save to CSV
    output_file = OUTPUT_DIR / "arxiv_papers.csv"
    df.write_csv(output_file)

    print(f"\n✓ Saved to {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Show sample
    print("\nSample papers:")
    for i, row in enumerate(df.head(3).to_dicts(), 1):
        print(f"\n{i}. [{row['category_name']}] {row['title']}")
        print(f"   Abstract: {row['abstract'][:150]}...")

    print(f"\n{'='*70}")
    print("Next step: Run precompute_embeddings.py with ArXiv dataset")
    print("  Edit script to use arxiv_data/arxiv_papers.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
