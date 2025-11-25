#!/usr/bin/env python3
"""
Test which types of hallucinations LSH density can detect

Compares 4 hallucination types:
1. Topic drift - mid-paragraph topic shifts
2. Domain mismatch - wrong vocabulary for domain
3. Genre shift - formal → casual language
4. Semantic category violations - type errors (emotions to algorithms, etc.)
"""
import polars as pl
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

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

def compute_perplexity(text, model, tokenizer, device='cpu'):
    """Compute perplexity using GPT-2"""
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()

# Corruption strategies
TOPIC_DRIFT_INSERTIONS = [
    "Meanwhile, the best way to make chocolate chip cookies is to cream butter and sugar together.",
    "Interestingly, the capital of France is Paris, which was founded in the 3rd century BC.",
    "On a related note, photosynthesis occurs in the chloroplasts of plant cells.",
    "It's worth mentioning that the average lifespan of a house cat is 15 years.",
    "Notably, the Pacific Ocean is the largest ocean on Earth, covering 165 million square kilometers.",
    "As an aside, Shakespeare wrote Hamlet around 1600.",
    "In cooking news, the optimal temperature for baking bread is 450°F.",
    "Speaking of which, Mount Everest is 29,032 feet tall.",
    "Incidentally, the chemical symbol for gold is Au, from the Latin aurum.",
    "By the way, the human body contains approximately 37 trillion cells."
]

DOMAIN_MISMATCH_INSERTIONS = {
    'legal': [
        "pursuant to Section 12(b) of the Securities Exchange Act",
        "in accordance with the doctrine of res judicata",
        "subject to the statute of limitations as defined in Title 18",
        "notwithstanding the precedent established in Brown v. Board",
        "as per the terms of the licensing agreement and indemnification clause"
    ],
    'medical': [
        "showing symptoms of acute myocardial infarction",
        "requiring immediate administration of epinephrine",
        "diagnosed with stage III metastatic carcinoma",
        "presenting with bilateral pulmonary edema",
        "indicating contraindications for beta-blocker therapy"
    ],
    'culinary': [
        "seasoned with kosher salt and freshly ground black pepper",
        "sautéed over medium-high heat until caramelized",
        "garnished with chopped parsley and lemon zest",
        "marinated overnight in olive oil and herbs",
        "baked at 375°F for 25-30 minutes until golden brown"
    ]
}

GENRE_SHIFT_INSERTIONS = [
    "Yo this is straight fire 🔥 no cap fr fr.",
    "Like, it's literally so cringe rn ngl.",
    "Bruh this slaps harder than my grandma's chancla lmao.",
    "Lowkey this hits different when you really think about it.",
    "Deadass the vibes are immaculate, not gonna lie.",
    "Fr tho this be bussin on god.",
    "Nah fam this ain't it chief.",
    "Ong this is mid at best, touch grass.",
    "Sheesh that's cap, ratio + L + cope.",
    "Real talk tho this finna change the game periodt."
]

CATEGORY_VIOLATION_INSERTIONS = [
    # Emotions to algorithms
    "The neural network felt anxious about overfitting.",
    "The optimizer was disappointed with the learning rate.",
    "The loss function experienced existential dread.",
    "The gradient descent algorithm fell in love with the local minimum.",

    # Physical properties to abstract concepts
    "The probability distribution weighed approximately 3 kilograms.",
    "The Gaussian process was 7 feet tall and bright blue.",
    "The training set tasted like strawberries and smelled of pine.",
    "The hyperparameter space was smooth to the touch.",

    # Biological properties to code
    "The Python function reproduced asexually every 3 hours.",
    "The database grew teeth and began hunting for prey.",
    "The API evolved photosynthesis capabilities over time.",
    "The compiler developed a respiratory system for better oxygen intake.",

    # Temporal impossibilities for objects
    "The vector embedding celebrated its 500th birthday yesterday.",
    "The convolutional layer remembered the Renaissance.",
    "The attention mechanism was born before the Big Bang.",
    "The dataset witnessed the extinction of dinosaurs."
]

def corrupt_topic_drift(text):
    """Insert random off-topic sentence in middle"""
    sentences = text.split('. ')
    if len(sentences) < 3:
        return text, 0

    insert_pos = len(sentences) // 2
    insertion = random.choice(TOPIC_DRIFT_INSERTIONS)
    corrupted = '. '.join(sentences[:insert_pos]) + '. ' + insertion + ' ' + '. '.join(sentences[insert_pos:])
    return corrupted, 1

def corrupt_domain_mismatch(text):
    """Insert wrong-domain vocabulary"""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return text, 0

    # Pick random domain (not ML/CS)
    domain = random.choice(['legal', 'medical', 'culinary'])
    insertion = random.choice(DOMAIN_MISMATCH_INSERTIONS[domain])

    insert_pos = random.randint(1, len(sentences) - 1)
    sentences[insert_pos] = sentences[insert_pos] + ' ' + insertion
    return '. '.join(sentences), 1

def corrupt_genre_shift(text):
    """Add casual/slang language"""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return text, 0

    insert_pos = random.randint(1, len(sentences))
    insertion = random.choice(GENRE_SHIFT_INSERTIONS)
    corrupted = '. '.join(sentences[:insert_pos]) + '. ' + insertion + ' ' + '. '.join(sentences[insert_pos:])
    return corrupted, 1

def corrupt_category_violation(text):
    """Insert semantic category errors"""
    sentences = text.split('. ')
    if len(sentences) < 2:
        return text, 0

    insert_pos = random.randint(1, len(sentences))
    insertion = random.choice(CATEGORY_VIOLATION_INSERTIONS)
    corrupted = '. '.join(sentences[:insert_pos]) + '. ' + insertion + ' ' + '. '.join(sentences[insert_pos:])
    return corrupted, 1

# Data paths
PROJECT_ROOT = Path("/Users/jdonaldson/Projects/semantic-proprioception")
ARXIV_DATA = PROJECT_ROOT / "arxiv_demo_data"
ARXIV_TEXT = PROJECT_ROOT / "arxiv_data" / "arxiv_papers.csv"
MODEL_NAME = "MiniLM-L6"

print("=" * 70)
print("HALLUCINATION TYPE COMPARISON")
print("=" * 70)
print()
print("Testing which hallucination types LSH density can detect")
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
print()

# Compute bucket density distribution
bucket_counts = (df_index
    .group_by('bucket_id')
    .agg(pl.count('row_id').alias('count'))
)

dense_threshold = 5

print(f"Bucket distribution:")
print(f"  Dense buckets (≥{dense_threshold}):     {len(bucket_counts.filter(pl.col('count') >= dense_threshold))}")
print()

# Load models
print("Loading models...")
embedding_model = SentenceTransformer(f"sentence-transformers/all-{MODEL_NAME}-v2")
print(f"  ✓ Loaded embedding model: {MODEL_NAME}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Loading GPT-2 for perplexity (device: {device})...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()
print(f"  ✓ Loaded GPT-2")
print()

# ============================================================================
# Sample ML/AI papers
# ============================================================================
print("Selecting ML/AI papers...")
sample_size = 30
random.seed(42)

ml_papers = df_texts.filter(
    (pl.col('category') == 'cs.LG') |    # Machine Learning
    (pl.col('category') == 'stat.ML') |  # Machine Learning (stats)
    (pl.col('category') == 'cs.AI') |    # Artificial Intelligence
    (pl.col('category') == 'cs.CV') |    # Computer Vision
    (pl.col('category') == 'cs.CL')      # Computational Linguistics
)

sample = ml_papers.sample(n=min(sample_size, len(ml_papers)), seed=42)
print(f"  Selected {len(sample)} papers")
print()

# ============================================================================
# Test each hallucination type
# ============================================================================
hallucination_types = [
    ('Topic Drift', corrupt_topic_drift),
    ('Domain Mismatch', corrupt_domain_mismatch),
    ('Genre Shift', corrupt_genre_shift),
    ('Category Violation', corrupt_category_violation),
]

results_by_type = {}

for hal_name, corrupt_fn in hallucination_types:
    print("=" * 70)
    print(f"HALLUCINATION TYPE: {hal_name}")
    print("=" * 70)
    print()

    original_densities = []
    corrupted_densities = []
    original_perplexities = []
    corrupted_perplexities = []
    examples = []

    for row in sample.iter_rows(named=True):
        abstract = row['abstract']

        # Corrupt the abstract
        corrupted, num_changes = corrupt_fn(abstract)

        if num_changes == 0:
            continue

        # Embed both versions
        original_emb = embedding_model.encode([abstract], show_progress_bar=False)[0]
        corrupted_emb = embedding_model.encode([corrupted], show_progress_bar=False)[0]

        # LSH density
        orig_bucket_id = lsh_hash(original_emb, num_bits=8, seed=12345)
        corr_bucket_id = lsh_hash(corrupted_emb, num_bits=8, seed=12345)

        orig_matches = bucket_counts.filter(pl.col('bucket_id') == orig_bucket_id)
        orig_size = orig_matches['count'][0] if len(orig_matches) > 0 else 0

        corr_matches = bucket_counts.filter(pl.col('bucket_id') == corr_bucket_id)
        corr_size = corr_matches['count'][0] if len(corr_matches) > 0 else 0

        original_densities.append(orig_size)
        corrupted_densities.append(corr_size)

        # Perplexity
        orig_ppl = compute_perplexity(abstract, gpt2_model, gpt2_tokenizer, device)
        corr_ppl = compute_perplexity(corrupted, gpt2_model, gpt2_tokenizer, device)
        original_perplexities.append(orig_ppl)
        corrupted_perplexities.append(corr_ppl)

        # Save first 2 examples
        if len(examples) < 2:
            examples.append({
                'original': abstract[:150] + "...",
                'corrupted': corrupted[:150] + "...",
                'orig_density': orig_size,
                'corr_density': corr_size,
                'orig_ppl': orig_ppl,
                'corr_ppl': corr_ppl,
            })

    if len(original_densities) == 0:
        print("  No abstracts corrupted")
        continue

    # Statistics
    orig_avg_density = np.mean(original_densities)
    corr_avg_density = np.mean(corrupted_densities)
    density_change_pct = (corr_avg_density - orig_avg_density) / orig_avg_density * 100

    orig_avg_ppl = np.mean(original_perplexities)
    corr_avg_ppl = np.mean(corrupted_perplexities)
    ppl_change_pct = (corr_avg_ppl - orig_avg_ppl) / orig_avg_ppl * 100

    results_by_type[hal_name] = {
        'n': len(original_densities),
        'density_change': density_change_pct,
        'ppl_change': ppl_change_pct,
    }

    print(f"Corrupted {len(original_densities)} abstracts")
    print()
    print(f"{'Method':<30s} {'Original':<15s} {'Corrupted':<15s} {'Change':<15s}")
    print("-" * 70)
    print(f"{'LSH Bucket Density':<30s} {orig_avg_density:>12.2f}  {corr_avg_density:>12.2f}  {density_change_pct:>11.1f}%")
    print(f"{'Perplexity (GPT-2)':<30s} {orig_avg_ppl:>12.2f}  {corr_avg_ppl:>12.2f}  {ppl_change_pct:>11.1f}%")
    print()

    # Show examples
    print("Examples:")
    print("-" * 70)
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Original:  {ex['original']}")
        print(f"  Corrupted: {ex['corrupted']}")
        print(f"  Density:     {ex['orig_density']:6.1f} → {ex['corr_density']:6.1f} ({ex['corr_density']-ex['orig_density']:+.1f})")
        print(f"  Perplexity:  {ex['orig_ppl']:6.1f} → {ex['corr_ppl']:6.1f} ({ex['corr_ppl']-ex['orig_ppl']:+.1f})")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY: HALLUCINATION DETECTION BY TYPE")
print("=" * 70)
print()

print(f"{'Hallucination Type':<25s} {'N':<6s} {'LSH Δ':<12s} {'Perplexity Δ':<15s} {'Winner':<10s}")
print("-" * 70)

for hal_name in results_by_type:
    r = results_by_type[hal_name]
    winner = 'LSH' if abs(r['density_change']) > abs(r['ppl_change']) else 'Perplexity'
    if abs(r['density_change']) < 5 and abs(r['ppl_change']) < 5:
        winner = 'Neither'

    print(f"{hal_name:<25s} {r['n']:<6d} {r['density_change']:>9.1f}%  {r['ppl_change']:>12.1f}%  {winner:<10s}")

print()
print("Interpretation:")
print()
print("LSH Density:")
print("  - Effective when semantic structure fundamentally changes")
print("  - Best for: topic drift, domain mismatch (different embedding regions)")
print("  - Weak for: subtle errors within same semantic space")
print()
print("Perplexity:")
print("  - Effective at detecting grammatically/statistically odd text")
print("  - Best for: genre shifts, category violations (violate language model)")
print("  - Consistent across all hallucination types")
print()
print("Conclusion:")
print("  - LSH density: Fast (O(1)) but only catches major semantic shifts")
print("  - Perplexity: Slow (O(n)) but catches more hallucination types")
print("  - Combining both may provide complementary signals")
print()
print("=" * 70)
