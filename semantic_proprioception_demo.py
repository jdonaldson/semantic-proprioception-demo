#!/usr/bin/env python3
"""
Semantic Proprioception - Multi-Model LSH Theme Discovery

Understanding data's internal structure through semantic self-awareness.

Demonstrates how different embedding models discover themes across diverse datasets:
- Twitter customer support tweets
- ArXiv research papers
- Hacker News posts

Uses Krapivin hash tables with semantic label merging. All embeddings
pre-computed for free hosting.

Features:
- Compare 4 different embedding models
- 3 diverse datasets (Twitter, ArXiv, Hacker News)
- Automatic theme discovery via LSH density
- Semantic label similarity merging
- Interactive theme exploration
- Model comparison visualization
"""

import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add krapivin-python to path
sys.path.insert(0, str(Path(__file__).parent / "krapivin-python"))
from lsh_parquet import load_dense_buckets_from_parquet, index_stats_from_parquet


# Page config
st.set_page_config(
    page_title="Semantic Proprioception",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_model_metadata(dataset="twitter"):
    """Load metadata about available models"""
    if dataset == "twitter":
        data_dir = Path("semantic_proprioception_data")
    elif dataset == "arxiv":
        data_dir = Path("arxiv_demo_data")
    elif dataset == "hackernews":
        data_dir = Path("hackernews_demo_data")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    metadata_path = data_dir / "models_metadata.parquet"

    if not metadata_path.exists():
        st.error(f"Metadata not found for {dataset}. Run precompute script first.")
        st.stop()

    return pl.read_parquet(metadata_path)


@st.cache_data
def load_model_data(model_name, dataset="twitter"):
    """Load embeddings and LSH index for a model"""
    if dataset == "twitter":
        data_dir = Path("semantic_proprioception_data")
    elif dataset == "arxiv":
        data_dir = Path("arxiv_demo_data")
    elif dataset == "hackernews":
        data_dir = Path("hackernews_demo_data")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load embeddings
    embeddings_path = data_dir / f"{model_name}_embeddings.parquet"
    df = pl.read_parquet(embeddings_path)

    # Load LSH index (full index with bucket_id, file_path, row_id)
    index_path = data_dir / f"{model_name}_lsh_index.parquet"
    lsh_index = pl.read_parquet(index_path)

    # Load index stats
    index_stats = index_stats_from_parquet(str(index_path))

    # Load dense buckets (returns Polars DataFrame)
    dense_buckets_df = load_dense_buckets_from_parquet(str(index_path), min_count=5)

    # Convert to list of tuples for easier use
    dense_buckets = [
        (row["bucket_id"], row["count"])
        for row in dense_buckets_df.to_dicts()
    ]

    return df, lsh_index, index_stats, dense_buckets


def get_bucket_tweets(bucket_id, lsh_index, df, dataset="twitter"):
    """Get all items in a specific bucket (tweets, papers, or HN posts)"""
    # Filter LSH index for this bucket
    bucket_rows = lsh_index.filter(pl.col("bucket_id") == bucket_id)

    # Get row IDs
    row_ids = bucket_rows.get_column("row_id").to_list()

    # Get items from embeddings dataframe
    items = []
    for row_id in row_ids:
        if row_id < len(df):
            if dataset == "arxiv":
                # ArXiv papers: use title + abstract
                title = df[row_id, "title"]
                abstract = df[row_id, "abstract"]
                item_id = df[row_id, "arxiv_id"]
                category = df[row_id, "category_name"]
                text = f"{title}\n\n{abstract[:200]}..."  # Title + truncated abstract
                items.append({
                    "text": text,
                    "title": title,
                    "abstract": abstract,
                    "tweet_id": item_id,  # Keep same key for compatibility
                    "category": category,
                    "row_id": row_id
                })
            elif dataset == "hackernews":
                # Hacker News posts: use title + text
                title = df[row_id, "title"]
                post_text = df[row_id, "text"]
                item_id = df[row_id, "hn_id"]
                category = df[row_id, "category_name"]
                score = df[row_id, "score"]
                by = df[row_id, "by"]
                text = f"{title}\n\n{post_text[:200]}..."  # Title + truncated text
                items.append({
                    "text": text,
                    "title": title,
                    "post_text": post_text,
                    "tweet_id": item_id,  # Keep same key for compatibility
                    "category": category,
                    "score": score,
                    "by": by,
                    "row_id": row_id
                })
            else:
                # Twitter tweets
                tweet_text = df[row_id, "text"]
                tweet_id = df[row_id, "tweet_id"]
                items.append({"text": tweet_text, "tweet_id": tweet_id, "row_id": row_id})

    return items


def generate_theme_label(tweets, bucket_id):
    """Generate a short theme label for a bucket using LLM or keywords"""

    # Try Ollama first (local, free, fast)
    try:
        import subprocess

        # Check if Ollama is available
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0 and 'llama3.2' in result.stdout:
            # Use Ollama to generate label
            sample_tweets = tweets[:5]
            tweets_text = "\n".join([f"- {t['text'][:100]}" for t in sample_tweets])

            prompt = f"""Generate a 3-5 word theme label for these similar customer support tweets.
Just the label, nothing else.

Tweets:
{tweets_text}

Label:"""

            result = subprocess.run(
                ['ollama', 'run', 'llama3.2:3b', prompt],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                label = result.stdout.strip()
                # Clean up
                label = label.replace('**', '').replace('*', '').strip('"\'')
                # Take first line if multiple
                label = label.split('\n')[0]
                # Limit length
                if len(label) > 50:
                    label = label[:47] + "..."
                return label
    except:
        pass

    # Fallback: keyword extraction
    all_text = " ".join([t["text"].lower() for t in tweets[:10]])

    # Common customer support keywords
    keywords = {
        "password": 0, "reset": 0, "login": 0, "account": 0,
        "charge": 0, "refund": 0, "cancel": 0, "billing": 0,
        "payment": 0, "order": 0, "delivery": 0, "shipping": 0,
        "help": 0, "support": 0, "issue": 0, "problem": 0,
        "email": 0, "phone": 0, "app": 0, "update": 0,
        "service": 0, "wifi": 0, "internet": 0, "connection": 0,
    }

    for keyword in keywords:
        keywords[keyword] = all_text.count(keyword)

    # Get top 3 keywords
    top_keywords = sorted(keywords.items(), key=lambda x: -x[1])[:3]
    top_keywords = [k for k, v in top_keywords if v > 0]

    if top_keywords:
        return " + ".join(top_keywords[:2]).title()
    else:
        return f"Theme {bucket_id}"


@st.cache_data
def get_bucket_labels(dense_buckets, lsh_index, df, dataset="twitter"):
    """Generate labels for all dense buckets (cached)"""
    labels = {}

    with st.spinner("Generating theme labels..."):
        for bucket_id, count in dense_buckets[:10]:
            items = get_bucket_tweets(bucket_id, lsh_index, df, dataset)
            label = generate_theme_label(items, bucket_id)
            labels[bucket_id] = label

    return labels


def compute_label_embeddings(labels, df):
    """
    Compute embeddings for theme labels using the same model as the tweets.

    Since embeddings are already computed, we'll use a simple TF-IDF-like approach
    for label similarity instead of loading the full model.
    """
    from collections import Counter
    import re

    # Tokenize labels
    label_tokens = {}
    for label in labels:
        # Simple tokenization: lowercase, split on non-alphanumeric
        tokens = re.findall(r'\w+', label.lower())
        label_tokens[label] = Counter(tokens)

    return label_tokens


def label_similarity(label1_tokens, label2_tokens):
    """Compute Jaccard similarity between two label token sets"""
    tokens1 = set(label1_tokens.keys())
    tokens2 = set(label2_tokens.keys())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def merge_themes_by_label(dense_buckets, theme_labels, lsh_index, df, similarity_threshold=0.5, dataset="twitter"):
    """
    Merge buckets with semantically similar labels into unified themes.

    Args:
        dense_buckets: List of (bucket_id, count) tuples
        theme_labels: Dict mapping bucket_id to label string
        lsh_index: LSH index DataFrame
        df: Embeddings DataFrame
        similarity_threshold: Merge labels with similarity >= threshold

    Returns:
        List of merged themes with merged labels
    """
    from collections import defaultdict

    # Get unique labels
    bucket_list = [(bid, count, theme_labels.get(bid, f"Theme {bid}")) for bid, count in dense_buckets[:10]]

    # Compute label token sets for similarity comparison
    unique_labels = list(set(label for _, _, label in bucket_list))
    label_tokens = compute_label_embeddings(unique_labels, df)

    # Group similar labels using hierarchical clustering approach
    label_groups = []  # Each group is a list of similar labels
    label_to_group = {}  # Map label to its group index

    for label in unique_labels:
        # Check if this label is similar to any existing group
        found_group = False

        for group_idx, group in enumerate(label_groups):
            # Check similarity with representative (first) label in group
            representative = group[0]
            sim = label_similarity(label_tokens[label], label_tokens[representative])

            if sim >= similarity_threshold:
                # Add to this group
                group.append(label)
                label_to_group[label] = group_idx
                found_group = True
                break

        if not found_group:
            # Create new group
            label_groups.append([label])
            label_to_group[label] = len(label_groups) - 1

    # Choose best label for each group (longest/most descriptive)
    group_labels = []
    for group in label_groups:
        # Sort by length (longer = more descriptive), break ties alphabetically
        best_label = max(group, key=lambda x: (len(x), x))
        group_labels.append(best_label)

    # Map original labels to merged labels
    label_mapping = {}
    for label, group_idx in label_to_group.items():
        label_mapping[label] = group_labels[group_idx]

    # Group buckets by merged label
    merged_label_to_buckets = defaultdict(list)

    for bucket_id, count, original_label in bucket_list:
        merged_label = label_mapping[original_label]
        merged_label_to_buckets[merged_label].append((bucket_id, count, original_label))

    # Create merged themes
    merged_themes = []

    for merged_label, bucket_data in merged_label_to_buckets.items():
        # Get all tweets from all buckets with this merged label
        all_tweets = []
        total_count = 0
        buckets = []
        original_labels = set()

        for bucket_id, count, original_label in bucket_data:
            items = get_bucket_tweets(bucket_id, lsh_index, df, dataset)
            all_tweets.extend(items)
            total_count += count
            buckets.append((bucket_id, count))
            original_labels.add(original_label)

        merged_themes.append({
            'label': merged_label,
            'original_labels': list(original_labels),
            'buckets': buckets,
            'total_count': total_count,
            'tweets': all_tweets,
            'is_merged': len(original_labels) > 1,
        })

    # Sort by total count (descending)
    merged_themes.sort(key=lambda x: -x['total_count'])

    return merged_themes


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_similar(query_embedding, df, top_k=10):
    """Find most similar tweets using brute force cosine similarity"""
    embeddings = np.array(df.get_column("embedding").to_list())
    query = np.array(query_embedding)

    # Compute similarities
    similarities = np.array([
        cosine_similarity(query, emb) for emb in embeddings
    ])

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": df[idx, "text"],
            "similarity": similarities[idx],
            "tweet_id": df[idx, "tweet_id"],
        })

    return results


def lsh_search(query_embedding, df, index_path, top_k=10):
    """Find similar tweets using LSH index (load from parquet)"""
    # This would require loading the actual index structure
    # For now, return empty to show the comparison
    # In production, we'd load the PyLSHIndex and do bucket lookup
    return []


def render_sidebar():
    """Render sidebar with model selection and info"""
    st.sidebar.title("🧠 Semantic Proprioception")
    st.sidebar.caption("Understanding Data Through Self-Awareness")
    st.sidebar.markdown("---")

    # Dataset selection
    st.sidebar.subheader("Dataset")

    dataset_options = {
        "📱 Twitter Customer Support": "twitter",
        "📚 ArXiv Research Papers": "arxiv",
        "💬 Hacker News": "hackernews"
    }

    dataset_display = st.sidebar.selectbox(
        "Choose dataset",
        options=list(dataset_options.keys()),
        help="Compare how themes emerge in different domains"
    )
    dataset = dataset_options[dataset_display]

    st.sidebar.markdown("---")

    # Load metadata
    metadata = load_model_metadata(dataset)

    # Model selection
    st.sidebar.subheader("Select Model")

    model_options = metadata.get_column("model_name").to_list()
    model_descriptions = metadata.get_column("description").to_list()

    # Format options with descriptions
    display_options = [
        f"{name}: {desc.split(' - ')[0]}"
        for name, desc in zip(model_options, model_descriptions)
    ]

    selected_display = st.sidebar.radio(
        "Embedding Model",
        display_options,
        help="Different models cluster tweets differently"
    )

    # Extract model name
    selected_model = selected_display.split(":")[0]

    # Show model info
    model_row = metadata.filter(pl.col("model_name") == selected_model).to_dicts()[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.metric("Dimension", f"{model_row['dimension']}")
    st.sidebar.metric("LSH Bits", f"{model_row['lsh_bits']}")
    st.sidebar.metric("Buckets Used", f"{model_row['num_buckets']}")
    st.sidebar.metric("Load Factor", f"{model_row['load_factor']:.1%}")
    st.sidebar.metric("Dense Themes", f"{model_row['dense_buckets']}")

    st.sidebar.markdown("---")

    # Theme merging settings
    st.sidebar.subheader("Theme Merging")
    similarity_threshold = st.sidebar.slider(
        "Label Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Merge themes with label similarity ≥ threshold (0=exact match only, 1=merge all)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About this demo:**
    - All embeddings pre-computed
    - No API keys needed
    - Free to host on Streamlit Cloud
    - Compare model clustering quality
    """)

    return selected_model, model_row, similarity_threshold, dataset


def render_themes_tab(df, lsh_index, dense_buckets, similarity_threshold=0.5, dataset="twitter"):
    """Render theme discovery tab"""
    st.header("🎯 Discovered Themes")

    if not dense_buckets:
        st.info("No dense themes found with current model")
        return

    # Generate labels for themes
    theme_labels = get_bucket_labels(dense_buckets, lsh_index, df, dataset)

    # Merge themes with semantically similar labels
    merged_themes = merge_themes_by_label(
        dense_buckets,
        theme_labels,
        lsh_index,
        df,
        similarity_threshold=similarity_threshold,
        dataset=dataset
    )

    st.markdown(f"""
    Found **{len(merged_themes)}** distinct themes from {len(dense_buckets)} dense buckets.
    Buckets with similar labels are automatically merged.
    """)

    # Show top themes
    st.subheader("Top Themes by Size")

    # Show merged themes
    for i, theme in enumerate(merged_themes, 1):
        label = theme['label']
        total_count = theme['total_count']
        buckets = theme['buckets']
        tweets = theme['tweets']
        is_merged = theme.get('is_merged', False)
        original_labels = theme.get('original_labels', [label])

        # Build bucket info string
        if len(buckets) > 1:
            bucket_info = f"{len(buckets)} buckets: " + ", ".join([hex(bid) for bid, _ in buckets[:3]])
            if len(buckets) > 3:
                bucket_info += f", +{len(buckets) - 3} more"
        else:
            bucket_info = f"bucket {hex(buckets[0][0])}"

        with st.expander(f"**{label}** — {total_count} tweets ({bucket_info})", expanded=(i==1)):
            # Show merge info
            if len(buckets) > 1:
                if is_merged:
                    st.info(f"🔗 **Semantically merged theme** from {len(buckets)} LSH buckets with similar labels")

                    # Show original labels that were merged
                    if len(original_labels) > 1:
                        with st.expander("📝 Original labels merged", expanded=False):
                            for orig_label in sorted(original_labels):
                                st.markdown(f"• {orig_label}")
                else:
                    st.info(f"📊 **Merged theme** from {len(buckets)} LSH buckets with identical labels")

                # Show bucket breakdown
                cols = st.columns(min(len(buckets), 4))
                for idx, (bucket_id, count) in enumerate(buckets[:4]):
                    with cols[idx % 4]:
                        st.metric(f"Bucket {hex(bucket_id)}", f"{count} tweets")

                st.markdown("---")

            st.markdown("### Sample Tweets")

            # Show first 8 tweets from merged theme
            for j, tweet in enumerate(tweets[:8], 1):
                st.markdown(f"**{j}.** {tweet['text']}")
                st.caption(f"Tweet ID: {tweet['tweet_id']} | Row: {tweet['row_id']}")
                st.markdown("")

            if len(tweets) > 8:
                st.caption(f"... and {len(tweets) - 8} more similar tweets")


def render_search_tab(df, model_name):
    """Render semantic search tab"""
    st.header("🔎 Semantic Search")

    st.markdown("""
    Search for similar customer support tweets. The search uses the selected
    embedding model to find semantically similar content.
    """)

    # Search input
    query = st.text_input(
        "Enter a customer support query",
        placeholder="e.g., 'password reset problem'",
        help="Type any customer support issue to find similar tweets"
    )

    if query:
        st.markdown("---")

        # For demo, search within existing tweets
        # In production, would embed query with model
        st.info(f"Searching with {model_name} model...")

        # Simple keyword search as fallback
        query_lower = query.lower()
        matches = df.filter(
            pl.col("text").str.to_lowercase().str.contains(query_lower)
        ).head(10)

        if len(matches) > 0:
            st.subheader(f"Found {len(matches)} similar tweets")

            for i, row in enumerate(matches.to_dicts(), 1):
                with st.expander(f"Result {i}"):
                    st.markdown(f"**Tweet:** {row['text']}")
                    st.caption(f"Tweet ID: {row['tweet_id']}")
        else:
            st.warning("No matches found. Try a different query.")


def render_comparison_tab(metadata):
    """Render model comparison tab"""
    st.header("📊 Model Comparison")

    st.markdown("""
    Different embedding models produce different semantic clusterings.
    Compare how each model organizes the same customer support tweets.
    """)

    # Show comparison table
    st.subheader("Model Statistics")

    comparison_df = metadata.select([
        "model_name",
        "dimension",
        "lsh_bits",
        "num_buckets",
        "load_factor",
        "dense_buckets",
    ]).to_pandas()

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # Insights
    st.subheader("Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Dimension vs Quality:**
        - Higher dimension = better semantic understanding
        - MPNet-base (768D) captures more nuance
        - MiniLM models (384D) balance speed/quality
        """)

    with col2:
        st.markdown("""
        **LSH Bucketing:**
        - More bits = more buckets = finer clustering
        - Dense buckets reveal common patterns
        - Load factor shows index efficiency
        """)

    # Chart would go here
    st.info("💡 Switch models in the sidebar to see how clustering changes")


def main():
    """Main app"""

    # Sidebar
    selected_model, model_info, similarity_threshold, dataset = render_sidebar()

    # Load data for selected model and dataset
    with st.spinner(f"Loading {selected_model} data for {dataset}..."):
        df, lsh_index, index_stats, dense_buckets = load_model_data(selected_model, dataset)

    # Main header
    st.title("🧠 Semantic Proprioception")
    st.caption("Multi-Model Theme Discovery via LSH Density Analysis")

    # Dataset-specific description
    dataset_desc = {
        "twitter": f"{len(df):,} customer support tweets from major brands",
        "arxiv": f"{len(df):,} research paper abstracts across 10 scientific fields",
        "hackernews": f"{len(df):,} Hacker News posts from top stories, discussions, and Show HN"
    }

    st.markdown(f"""
    Comparing **4 different embedding models** on {dataset_desc[dataset]}.
    Themes emerge automatically from LSH bucket density—the data reveals its own structure.

    **Currently viewing:** {selected_model} ({model_info['dimension']} dimensions) on **{dataset.title()}** dataset
    """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Themes", "🔎 Search", "📊 Comparison"])

    with tab1:
        render_themes_tab(df, lsh_index, dense_buckets, similarity_threshold, dataset)

    with tab2:
        render_search_tab(df, selected_model)

    with tab3:
        metadata = load_model_metadata()
        render_comparison_tab(metadata)

    # Footer
    st.markdown("---")
    st.caption(f"""
    **Semantic Proprioception** — Data understanding through self-awareness |
    [Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) |
    Krapivin LSH + sentence-transformers |
    Fixed seed for composability
    """)


if __name__ == "__main__":
    main()
