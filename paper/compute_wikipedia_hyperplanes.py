#!/usr/bin/env python3
"""
Compute universal LSH hyperplanes from Wikipedia sample

Strategy: Train PCA on diverse Wikipedia articles, use as hyperplanes
Goal: Better than random, still compositional (same for all corpora)
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset
import json
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
N_SAMPLES = 10000  # 10K articles
N_COMPONENTS = 16  # 16-bit hash for hierarchical LSH

print("=" * 70)
print("COMPUTING UNIVERSAL LSH HYPERPLANES FROM WIKIPEDIA")
print("=" * 70)
print()
print(f"Model: {MODEL_NAME}")
print(f"Corpus: Wikipedia (en, 20220301)")
print(f"Sample size: {N_SAMPLES} articles")
print(f"Components: {N_COMPONENTS} (for 16-bit LSH)")
print()

# ============================================================================
# Load Wikipedia
# ============================================================================
print("Loading Wikipedia dataset...")
print("  (This may take a few minutes on first run)")

try:
    # Try newer Parquet-based Wikipedia dataset
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{N_SAMPLES}]"
    )
    print(f"  ✓ Loaded {len(wiki)} articles")
except Exception as e:
    print(f"  ✗ Error loading Wikipedia dataset: {e}")
    print()
    print("Using fallback: C4 (Colossal Clean Crawled Corpus)")
    print("  (General web text, similar diversity to Wikipedia)")
    # Fallback to C4 - general web text
    wiki = load_dataset(
        "allenai/c4",
        "en",
        split=f"train[:{N_SAMPLES}]",
        streaming=False
    )
    N_SAMPLES = len(wiki)
    print(f"  ✓ Loaded {N_SAMPLES} C4 samples")

print()

# Sample texts (take first paragraph of each article to keep embeddings manageable)
texts = []
for article in wiki:
    text = article['text']
    # Take first 500 chars (roughly first paragraph)
    paragraph = text[:500] if len(text) > 500 else text
    texts.append(paragraph)

print(f"Sample texts prepared: {len(texts)} paragraphs")
print()

# ============================================================================
# Embed with sentence-transformers
# ============================================================================
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(f"sentence-transformers/{MODEL_NAME}")
print(f"  ✓ Loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
print()

print(f"Embedding {len(texts)} Wikipedia paragraphs...")
print("  (This will take ~10-15 minutes on CPU)")
print()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print()
print(f"  ✓ Embeddings computed: shape {embeddings.shape}")
print()

# ============================================================================
# Compute PCA hyperplanes
# ============================================================================
print("Computing PCA for hyperplanes...")

pca = PCA(n_components=N_COMPONENTS)
pca.fit(embeddings)

print(f"  ✓ PCA fitted")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_[:8]}")
print(f"  Cumulative variance: {np.sum(pca.explained_variance_ratio_):.3f}")
print()

# Hyperplanes are the principal components
hyperplanes = pca.components_

# ============================================================================
# Analyze hyperplanes
# ============================================================================
print("Analyzing learned hyperplanes...")

# Project embeddings onto components to find extremes
for i in range(min(3, N_COMPONENTS)):
    component = pca.components_[i]
    projections = embeddings @ component

    # Find extremes
    top_idx = np.argmax(projections)
    bot_idx = np.argmin(projections)

    print(f"\nComponent {i} (variance: {pca.explained_variance_ratio_[i]:.3f}):")
    print(f"  Positive extreme: {texts[top_idx][:80]}...")
    print(f"  Negative extreme: {texts[bot_idx][:80]}...")

print()

# ============================================================================
# Save hyperplanes
# ============================================================================
output_dir = Path(__file__).parent
hyperplanes_file = output_dir / f"{MODEL_NAME}_wikipedia_hyperplanes.npy"
metadata_file = output_dir / f"{MODEL_NAME}_wikipedia_hyperplanes.json"

print("Saving hyperplanes...")
np.save(hyperplanes_file, hyperplanes)
print(f"  ✓ Saved: {hyperplanes_file}")

# Save metadata
metadata = {
    'model': MODEL_NAME,
    'corpus': 'wikipedia_20220301_en',
    'n_samples': N_SAMPLES,
    'n_components': N_COMPONENTS,
    'embedding_dim': embeddings.shape[1],
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'cumulative_variance': float(np.sum(pca.explained_variance_ratio_)),
    'mean': pca.mean_.tolist(),
}

with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved metadata: {metadata_file}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"Learned hyperplanes from {N_SAMPLES} Wikipedia articles")
print(f"Model: {MODEL_NAME}")
print(f"Components: {N_COMPONENTS} (for 16-bit hierarchical LSH)")
print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.1%}")
print()
print("Files created:")
print(f"  - {hyperplanes_file.name}")
print(f"  - {metadata_file.name}")
print()
print("Next step: Test on Twitter/ArXiv/HackerNews datasets")
print("  Run: python3 test_wikipedia_hyperplanes.py")
print()
print("=" * 70)
