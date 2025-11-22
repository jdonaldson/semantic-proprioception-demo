#!/usr/bin/env python3
"""
Fetch Hacker News posts for semantic proprioception demo

Downloads top stories and posts from different categories to show diverse topics.
"""

import requests
import polars as pl
from pathlib import Path
from datetime import datetime
import time

# HN API endpoints
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_ITEM_URL = f"{HN_API_BASE}/item"

# Categories to sample (different story types)
CATEGORIES = {
    'topstories': 'Top Stories',
    'beststories': 'Best Stories',
    'newstories': 'New Stories',
    'askstories': 'Ask HN',
    'showstories': 'Show HN',
}

POSTS_PER_CATEGORY = 200
OUTPUT_DIR = Path("hackernews_data")


def fetch_item(item_id):
    """Fetch a single HN item by ID"""
    try:
        response = requests.get(f"{HN_ITEM_URL}/{item_id}.json", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching item {item_id}: {e}")
        return None


def fetch_story_ids(category):
    """Fetch story IDs for a category"""
    print(f"Fetching {category} IDs...")
    try:
        response = requests.get(f"{HN_API_BASE}/{category}.json", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching {category}: {e}")
        return []


def fetch_posts_by_category(category, max_results=200):
    """Fetch posts from a specific HN category"""
    print(f"Fetching {max_results} posts from {category}...")

    story_ids = fetch_story_ids(category)
    if not story_ids:
        return []

    # Take first max_results IDs
    story_ids = story_ids[:max_results * 2]  # Fetch extra to account for filtering

    posts = []
    for i, item_id in enumerate(story_ids):
        if len(posts) >= max_results:
            break

        item = fetch_item(item_id)
        if not item:
            continue

        # Skip deleted/dead posts
        if item.get('deleted') or item.get('dead'):
            continue

        # Must have a title
        if not item.get('title'):
            continue

        # Get text content (for Ask/Show HN) or URL
        text = item.get('text', '')
        url = item.get('url', '')

        # Skip posts with no text content
        if not text and category in ['askstories', 'showstories']:
            continue

        # Clean HTML from text
        if text:
            import html
            text = html.unescape(text)
            # Remove basic HTML tags (simple approach)
            text = text.replace('<p>', '\n')
            text = text.replace('</p>', '')
            text = text.replace('<i>', '')
            text = text.replace('</i>', '')
            text = text.replace('<a>', '')
            text = text.replace('</a>', '')

        posts.append({
            'hn_id': item.get('id'),
            'title': item.get('title'),
            'text': text if text else f"Link: {url}",
            'url': url,
            'category': category,
            'category_name': CATEGORIES[category],
            'score': item.get('score', 0),
            'by': item.get('by', 'unknown'),
            'time': datetime.fromtimestamp(item.get('time', 0)).isoformat(),
        })

        if (i + 1) % 50 == 0:
            print(f"  Retrieved {len(posts)} posts so far...")
            time.sleep(0.5)  # Rate limit

    print(f"  Retrieved {len(posts)} posts")
    return posts


def main():
    print("="*70)
    print("Fetching Hacker News Posts for Semantic Proprioception Demo")
    print("="*70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Fetch posts from all categories
    all_posts = []

    for category in CATEGORIES.keys():
        posts = fetch_posts_by_category(category, POSTS_PER_CATEGORY)
        all_posts.extend(posts)
        time.sleep(1)  # Rate limit between categories

    # Convert to DataFrame
    df = pl.DataFrame(all_posts)

    print(f"\n{'='*70}")
    print(f"Total posts fetched: {len(df):,}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"{'='*70}\n")

    # Show statistics
    print("Posts per category:")
    category_counts = df.group_by("category_name").agg(pl.len().alias("count")).sort("count", descending=True)
    print(category_counts)

    print("\nText length statistics:")
    df = df.with_columns(
        pl.col("text").str.len_chars().alias("text_length")
    )

    print(f"  Min: {df['text_length'].min()} chars")
    print(f"  Max: {df['text_length'].max()} chars")
    print(f"  Mean: {df['text_length'].mean():.0f} chars")
    print(f"  Median: {df['text_length'].median():.0f} chars")

    print("\nScore statistics:")
    print(f"  Min: {df['score'].min()}")
    print(f"  Max: {df['score'].max()}")
    print(f"  Mean: {df['score'].mean():.0f}")
    print(f"  Median: {df['score'].median():.0f}")

    # Save to CSV
    output_file = OUTPUT_DIR / "hackernews_posts.csv"
    df.write_csv(output_file)

    print(f"\n✓ Saved to {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Show sample
    print("\nSample posts:")
    for i, row in enumerate(df.head(3).to_dicts(), 1):
        print(f"\n{i}. [{row['category_name']}] {row['title']}")
        print(f"   Score: {row['score']} | By: {row['by']}")
        print(f"   Text: {row['text'][:150]}...")

    print(f"\n{'='*70}")
    print("Next step: Run precompute_hackernews_embeddings.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
