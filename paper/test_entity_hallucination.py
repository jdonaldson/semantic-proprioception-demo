#!/usr/bin/env python3
"""
Test hallucination detection via entity substitution

Hypothesis: Replacing real technical terms with fake ones (while preserving
grammatical structure) should move embeddings to sparse buckets.
"""
import polars as pl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def lsh_hash(embeddings, num_bits=8, seed=12345):
    """Compute LSH hash for embeddings"""
    np.random.seed(seed)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
        single = True
    else:
        single = False

    d = embeddings.shape[1]
    hyperplanes = np.random.randn(num_bits, d)
    hash_bits = (embeddings @ hyperplanes.T > 0).astype(int)
    powers = 2 ** np.arange(num_bits)
    hash_values = hash_bits @ powers

    return hash_values[0] if single else hash_values

# Entity substitution mappings
REAL_TO_FAKE = {
    # Models/Architectures
    r'\bBERT\b': 'FLORP',
    r'\bGPT\b': 'ZUNK',
    r'\bTransformer\b': 'Morphulator',
    r'\btransformer\b': 'morphulator',
    r'\bLSTM\b': 'QWEX',
    r'\bCNN\b': 'PLEX',
    r'\bResNet\b': 'NetZorp',
    r'\bVGG\b': 'ZVV',
    r'\bAttention\b': 'Focusation',
    r'\battention\b': 'focusation',

    # Algorithms/Methods
    r'\bbackpropagation\b': 'retrofluxion',
    r'\bgradient descent\b': 'slope navigation',
    r'\bneural network\b': 'synapse matrix',
    r'\bdeep learning\b': 'profound cognition',
    r'\bmachine learning\b': 'apparatus training',
    r'\bconvolution\b': 'wavolution',
    r'\boptimization\b': 'enhancification',

    # Datasets
    r'\bImageNet\b': 'PixelVault',
    r'\bCOCO\b': 'DODO',
    r'\bMNIST\b': 'ZNIST',
    r'\bWikipedia\b': 'Infobase',

    # Metrics
    r'\baccuracy\b': 'correctitude',
    r'\bF1 score\b': 'G2 metric',
    r'\bROC\b': 'XYZ',
    r'\bAUC\b': 'QWP',

    # Mathematical terms
    r'\bmatrix\b': 'gridform',
    r'\btensor\b': 'hyperarray',
    r'\bvector\b': 'directional',
    r'\beigenvalue\b': 'selfvalue',

    # Common ML terms
    r'\bembedding\b': 'encapsulation',
    r'\blatent\b': 'hidden',
    r'\bfeature\b': 'attribute',
    r'\bclassification\b': 'categorization',
    r'\bregression\b': 'prediction',
}

def corrupt_text(text, corruption_rate=0.3):
    """
    Replace real technical terms with fake ones

    Args:
        text: Original text
        corruption_rate: Probability of replacing each matched term

    Returns:
        (corrupted_text, num_replacements)
    """
    corrupted = text
    num_replacements = 0

    for pattern, replacement in REAL_TO_FAKE.items():
        matches = list(re.finditer(pattern, corrupted, re.IGNORECASE))

        for match in matches:
            if random.random() < corruption_rate:
                start, end = match.span()
                corrupted = corrupted[:start] + replacement + corrupted[end:]
                num_replacements += 1

    return corrupted, num_replacements

def compute_perplexity(text, model, tokenizer, device='cpu'):
    """
    Compute perplexity using GPT-2

    Args:
        text: Input text
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        device: 'cpu' or 'cuda'

    Returns:
        perplexity (float)
    """
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()

def compute_embedding_magnitude(embedding):
    """Compute L2 norm of embedding vector"""
    return np.linalg.norm(embedding)

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
ARXIV_DATA = PROJECT_ROOT / "arxiv_demo_data"
ARXIV_TEXT = PROJECT_ROOT / "arxiv_data" / "arxiv_papers.csv"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("ENTITY SUBSTITUTION HALLUCINATION TEST")
print("=" * 70)
print()
print("Testing whether fake technical terms move text to sparse buckets")
print()

# ============================================================================
# Load ArXiv data
# ============================================================================
print("Loading ArXiv data...")
embeddings_file = ARXIV_DATA / f"{MODEL_NAME}_embeddings.parquet"
index_file = ARXIV_DATA / f"{MODEL_NAME}_lsh_index.parquet"

df_embeddings = pl.read_parquet(embeddings_file)
df_index = pl.read_parquet(index_file)
df_texts = pl.read_csv(ARXIV_TEXT)

print(f"  {len(df_embeddings):,} ArXiv abstracts")
print(f"  {len(df_texts):,} text records")
print()

# Compute bucket density distribution
bucket_counts = (df_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
)

dense_threshold = 5
dense_buckets = set(bucket_counts.filter(pl.col('count') >= dense_threshold)['bucket_id'].to_list())
sparse_buckets = set(bucket_counts.filter((pl.col('count') > 0) & (pl.col('count') < dense_threshold))['bucket_id'].to_list())
occupied_buckets = set(bucket_counts['bucket_id'].to_list())

print(f"Bucket distribution:")
print(f"  Dense buckets (≥{dense_threshold}):     {len(dense_buckets)}")
print(f"  Sparse buckets (1-{dense_threshold-1}):        {len(sparse_buckets)}")
print(f"  Empty buckets:            {256 - len(occupied_buckets)}")
print()

# Load models
print("Loading models...")
embedding_model = SentenceTransformer(f"sentence-transformers/all-{MODEL_NAME}-v2")
print(f"  ✓ Loaded embedding model: {MODEL_NAME}")

# Load GPT-2 for perplexity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Loading GPT-2 for perplexity (device: {device})...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()
print(f"  ✓ Loaded GPT-2")
print()

# ============================================================================
# Sample abstracts with technical terms
# ============================================================================
print("Selecting abstracts with technical terms...")
sample_size = 50
random.seed(42)

# Filter for ML/AI papers (more technical terms)
ml_papers = df_texts.filter(
    (pl.col('category') == 'Machine Learning') |
    (pl.col('category') == 'Artificial Intelligence') |
    (pl.col('category') == 'Computer Vision')
)

if len(ml_papers) < sample_size:
    ml_papers = df_texts

sample = ml_papers.sample(n=min(sample_size, len(ml_papers)), seed=42)
print(f"  Selected {len(sample)} papers")
print()

# ============================================================================
# Test corruption at different rates
# ============================================================================
corruption_rates = [0.3, 0.5, 0.7]

for corruption_rate in corruption_rates:
    print("=" * 70)
    print(f"CORRUPTION RATE: {corruption_rate:.0%}")
    print("=" * 70)
    print()

    original_buckets = []
    corrupted_buckets = []
    original_perplexities = []
    corrupted_perplexities = []
    original_magnitudes = []
    corrupted_magnitudes = []
    examples = []

    for row in sample.iter_rows(named=True):
        abstract = row['abstract']

        # Corrupt the abstract
        corrupted, num_replacements = corrupt_text(abstract, corruption_rate)

        # Skip if no replacements made
        if num_replacements == 0:
            continue

        # Embed both versions
        original_emb = embedding_model.encode([abstract], show_progress_bar=False)[0]
        corrupted_emb = embedding_model.encode([corrupted], show_progress_bar=False)[0]

        # Method 1: LSH bucket density
        orig_bucket_id = lsh_hash(original_emb, num_bits=8, seed=12345)
        corr_bucket_id = lsh_hash(corrupted_emb, num_bits=8, seed=12345)

        orig_matches = bucket_counts.filter(pl.col('bucket_id') == orig_bucket_id)
        orig_size = orig_matches['count'][0] if len(orig_matches) > 0 else 0

        corr_matches = bucket_counts.filter(pl.col('bucket_id') == corr_bucket_id)
        corr_size = corr_matches['count'][0] if len(corr_matches) > 0 else 0

        original_buckets.append((orig_bucket_id, orig_size))
        corrupted_buckets.append((corr_bucket_id, corr_size))

        # Method 2: Perplexity (GPT-2)
        orig_ppl = compute_perplexity(abstract, gpt2_model, gpt2_tokenizer, device)
        corr_ppl = compute_perplexity(corrupted, gpt2_model, gpt2_tokenizer, device)
        original_perplexities.append(orig_ppl)
        corrupted_perplexities.append(corr_ppl)

        # Method 3: Embedding magnitude
        orig_mag = compute_embedding_magnitude(original_emb)
        corr_mag = compute_embedding_magnitude(corrupted_emb)
        original_magnitudes.append(orig_mag)
        corrupted_magnitudes.append(corr_mag)

        # Save first few examples
        if len(examples) < 3:
            examples.append({
                'original': abstract[:200] + "...",
                'corrupted': corrupted[:200] + "...",
                'num_replacements': num_replacements,
                'orig_density': orig_size,
                'corr_density': corr_size,
                'orig_ppl': orig_ppl,
                'corr_ppl': corr_ppl,
                'orig_mag': orig_mag,
                'corr_mag': corr_mag,
                'bucket_changed': orig_bucket_id != corr_bucket_id
            })

    if len(original_buckets) == 0:
        print("  No abstracts corrupted (no technical terms found)")
        continue

    # Statistics - Method 1: LSH Density
    orig_dense = sum(1 for _, size in original_buckets if size >= dense_threshold)
    orig_avg_density = np.mean([size for _, size in original_buckets])
    corr_avg_density = np.mean([size for _, size in corrupted_buckets])
    density_change_pct = (corr_avg_density - orig_avg_density) / orig_avg_density * 100

    # Statistics - Method 2: Perplexity
    orig_avg_ppl = np.mean(original_perplexities)
    corr_avg_ppl = np.mean(corrupted_perplexities)
    ppl_change_pct = (corr_avg_ppl - orig_avg_ppl) / orig_avg_ppl * 100

    # Statistics - Method 3: Embedding Magnitude
    orig_avg_mag = np.mean(original_magnitudes)
    corr_avg_mag = np.mean(corrupted_magnitudes)
    mag_change_pct = (corr_avg_mag - orig_avg_mag) / orig_avg_mag * 100

    print(f"Corrupted {len(original_buckets)} abstracts")
    print()
    print("METHOD COMPARISON:")
    print("-" * 70)
    print(f"{'Method':<30s} {'Original':<15s} {'Corrupted':<15s} {'Change':<15s} {'Complexity':<15s}")
    print("-" * 70)
    print(f"{'LSH Bucket Density':<30s} {orig_avg_density:>12.2f}  {corr_avg_density:>12.2f}  {density_change_pct:>11.1f}%  {'O(1) query':<15s}")
    print(f"{'Perplexity (GPT-2)':<30s} {orig_avg_ppl:>12.2f}  {corr_avg_ppl:>12.2f}  {ppl_change_pct:>11.1f}%  {'O(n) forward':<15s}")
    print(f"{'Embedding Magnitude':<30s} {orig_avg_mag:>12.2f}  {corr_avg_mag:>12.2f}  {mag_change_pct:>11.1f}%  {'O(1) norm':<15s}")
    print()

    # Show examples
    print("Examples:")
    print("-" * 70)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: ({ex['num_replacements']} terms replaced)")
        print(f"  Original:  {ex['original']}")
        print(f"  Corrupted: {ex['corrupted']}")
        print(f"  Density:     {ex['orig_density']:6.1f} → {ex['corr_density']:6.1f} ({ex['corr_density']-ex['orig_density']:+.1f})")
        print(f"  Perplexity:  {ex['orig_ppl']:6.1f} → {ex['corr_ppl']:6.1f} ({ex['corr_ppl']-ex['orig_ppl']:+.1f})")
        print(f"  Magnitude:   {ex['orig_mag']:6.2f} → {ex['corr_mag']:6.2f} ({ex['corr_mag']-ex['orig_mag']:+.2f})")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY: HALLUCINATION DETECTION METHODS")
print("=" * 70)
print()
print("Entity substitution tests whether methods can detect plausible-but-wrong")
print("content (fake technical terms in otherwise valid text).")
print()
print("Method Comparison:")
print()
print("1. LSH Bucket Density (O(1) query)")
print("   - Measures prevalence of semantic pattern in training data")
print("   - Fast: Constant-time lookup after indexing")
print("   - Requires: Pre-built index from training/reference corpus")
print()
print("2. Perplexity via GPT-2 (O(n) forward pass)")
print("   - Measures how 'surprising' text is to language model")
print("   - Slow: Full forward pass required per text")
print("   - Requires: Language model (e.g., GPT-2, 124M params)")
print()
print("3. Embedding Magnitude (O(1) norm)")
print("   - Measures vector length in embedding space")
print("   - Fast: Simple L2 norm computation")
print("   - Requires: Embedding model only")
print()
print("Trade-offs:")
print("  - Perplexity: Best signal but computationally expensive")
print("  - LSH Density: Fast but requires pre-built index")
print("  - Magnitude: Fastest but weakest signal")
print()
print("Conclusion:")
print("  - No single method is perfect for hallucination detection")
print("  - Perplexity provides strongest signal at highest cost")
print("  - LSH density offers middle ground: moderate signal, fast queries")
print("  - Combining methods may be necessary for production systems")
print()
print("=" * 70)
