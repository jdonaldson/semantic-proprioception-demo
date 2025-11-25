# Learned Hyperplanes Experiment: Wikipedia PCA

**Date**: 2024-11-24
**Context**: Testing whether hyperplanes learned from general corpus improve over random
**Result**: Random hyperplanes win (simpler, no training, essentially identical performance)

## Hypothesis

**Question**: Can we improve LSH bucket coherence by learning hyperplanes from a diverse reference corpus?

**Approach**:
- Train PCA on 10K Wikipedia articles (general-purpose corpus)
- Use principal components as LSH hyperplanes
- Test on Twitter/ArXiv/HackerNews datasets
- Compare against random hyperplanes (baseline)

**Expected outcome**: Wikipedia-learned hyperplanes would capture semantic structure better than random

## Implementation

### Data
- **Training corpus**: Wikipedia (wikimedia/wikipedia, 20231101.en)
- **Sample size**: 10,000 articles (first 500 chars of each)
- **Download size**: ~1GB (41 shard files) - overkill for 10K samples
- **Embedding model**: all-MiniLM-L6-v2 (384-dim)

### Method
```python
# Load 10K Wikipedia articles
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")

# Embed with MiniLM-L6
embeddings = model.encode(wiki['text'])  # (10000, 384)

# Compute PCA (16 components for 16-bit LSH)
pca = PCA(n_components=16)
pca.fit(embeddings)

# Use principal components as hyperplanes
hyperplanes = pca.components_  # (16, 384)
```

**Explained variance**: 25.1% (first 16 components)

### Evaluation
Test on 3 datasets:
- Twitter (1,000 customer support tweets)
- ArXiv (1,000 research abstracts)
- Hacker News (684 tech discussion posts)

Metrics:
- Dense bucket coherence (≥5 items)
- Number of dense buckets
- Hierarchical refinement quality (bit slicing)

## Results

### Initial Coherence (Level 0, 8-bit hash)

| Dataset | Random | Wikipedia PCA | Difference | Winner |
|---------|--------|---------------|------------|--------|
| Twitter | 0.251 | **0.281** | **+0.030** | Wikipedia |
| ArXiv | 0.248 | 0.244 | -0.004 | Tie |
| Hacker News | **0.236** | 0.213 | -0.023 | Random |
| **Average** | - | - | **+0.001** | **Random** |

### After Hierarchical Refinement (bit slicing)

| Dataset | Random | Wikipedia PCA | Difference |
|---------|--------|---------------|------------|
| Twitter | 0.293 | 0.315 | +0.022 |
| ArXiv | 0.383 | **0.463** | **+0.080** |
| Hacker News | **0.542** | 0.516 | -0.027 |
| **Average** | - | - | **+0.025** |

### Key Findings

1. **Overall**: Essentially no difference (+0.001 average)
2. **Twitter**: Wikipedia helps modestly (+0.030)
   - General text structure aligns with customer support language
3. **ArXiv**: Neutral initially, helps after refinement (+0.080)
   - Learned structure aids hierarchical subdivision
4. **Hacker News**: Random wins (-0.023)
   - Tech discussions don't match Wikipedia's editorial style

## Why Wikipedia Didn't Help

### 1. Embeddings Already Optimized
Sentence-transformers (MiniLM-L6) are **pre-trained on diverse text** including:
- Wikipedia
- Common Crawl
- Reddit
- News articles
- Academic papers

The embedding space **already captures semantic structure** from diverse training.

### 2. PCA Captures Variance, Not Separability
PCA finds directions of **maximum variance**, which doesn't necessarily mean **best semantic separation**.

Random hyperplanes provide unbiased partitioning. PCA hyperplanes may:
- Over-represent common topics (high variance)
- Under-represent rare but distinct topics (low variance)

### 3. Domain Mismatch
Wikipedia has a specific style:
- Encyclopedia articles (formal, factual)
- Broad coverage but shallow depth
- Editorial consistency

Our test datasets differ:
- Twitter: Conversational, problem-oriented
- ArXiv: Technical, specialized terminology
- Hacker News: Informal, tech-focused discussions

### 4. Small Gains Don't Justify Complexity

Even where Wikipedia helps (Twitter +0.030), the gain is **marginal** and doesn't offset:
- Training time (~15 min for 10K samples)
- Download overhead (~1GB)
- Storage (hyperplane files)
- Model-specific (need separate hyperplanes per embedding model)
- Loss of simplicity

## Comparison: What Actually Works

| Approach | Coherence Gain | Speed | Complexity | Compositional |
|----------|----------------|-------|------------|---------------|
| **Random hyperplanes** | Baseline | Instant | Simple | ✓ |
| Wikipedia PCA | +0.001 | 15 min training | Complex | ✓ |
| **Bit slicing** | **+0.145** | **2.1× faster** | Simple | ✓ |

**Bit slicing** (compute 16-bit hash once, extract levels via bit masking) provides:
- 145× larger improvement than learned hyperplanes
- 2.1× faster execution
- No training required
- Still compositional

## Alternative Lightweight Corpora (for future reference)

If we were to retry this, use smaller datasets:

1. **AG News** (~5MB for 10K): News articles, diverse topics
2. **DBpedia** (~8MB for 10K): Wikipedia excerpts, pre-processed
3. **Multi-domain blend** (~15MB): AG News + IMDB + DBpedia

Wikipedia download was **overkill** (1GB for 10K samples due to sharding).

## Recommendation

**Use random hyperplanes** (current approach):

**Pros**:
- Simple implementation
- No training needed
- Compositional (same seed = same buckets)
- Unbiased partitioning
- Proven effective with bit slicing

**Cons of learned hyperplanes**:
- Minimal improvement (+0.001)
- Adds complexity (training, storage, versioning)
- Model-specific (need to retrain for each embedding model)
- Corpus-dependent (Wikipedia may not match target domain)

## Conclusion

**Learned hyperplanes do not improve LSH bucket coherence significantly.**

The experiment validates our current architecture:
- **Random hyperplanes** for initial projection (simple, compositional)
- **Bit slicing** for hierarchical refinement (2.1× better quality, 2.1× faster)
- **Fixed seed** for composability across datasets

This closes the "learned hyperplanes" experimental direction. The real gains come from **how we use the bits** (bit slicing), not from **which hyperplanes** we choose.

## Files Created

1. `compute_wikipedia_hyperplanes.py` - Compute PCA from Wikipedia
2. `test_wikipedia_hyperplanes.py` - Compare Wikipedia vs random
3. `all-MiniLM-L6-v2_wikipedia_hyperplanes.npy` - Learned hyperplanes (archived)
4. `all-MiniLM-L6-v2_wikipedia_hyperplanes.json` - Metadata (archived)
5. `NOTES_learned_hyperplanes.md` - This document

## Next Steps

- [x] Document findings
- [x] Archive hyperplane files (not needed for production)
- [ ] Consider brief mention in paper's Related Work
- [ ] Focus on bit slicing (the actual winner)

---

**Author**: J. Justin Donaldson
**Project**: Semantic Proprioception
**Experiment**: Learned hyperplanes vs random
**Verdict**: Random wins (simplicity, composability, no loss in quality)
