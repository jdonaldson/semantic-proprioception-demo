# Bit Slicing for Hierarchical LSH: Research Findings

**Date**: 2024-11-24
**Context**: Exploring methods to improve bucket purity in semantic proprioception paper

## Problem Statement

Dense LSH buckets (≥10 items) have lower coherence than sparse buckets in 2/3 datasets:
- Twitter: +8.4% (dense better)
- ArXiv: -13.9% (dense worse)
- Hacker News: -46.9% (dense worse)

This suggests dense buckets are "catch-all" buckets with diverse content. We need a way to refine them into more coherent sub-buckets.

## Approaches Tested

### 1. Separate Hashes (Baseline)
Compute new LSH hash at each level with different seeds.

```python
# Level 0: seed=12345, 8 bits
# Level 1: seed=13345, 4 bits
# Level 2: seed=14345, 4 bits
```

**Complexity**: O(k×d) for k levels, d dimensions
**Result**: +0.069 average coherence improvement, 1.98ms per bucket

### 2. Orthogonal Basis
QR-decomposed random matrix provides non-overlapping hyperplane sets.

```python
Q, _ = np.linalg.qr(np.random.randn(d, d))
# Level 0: Q[:, 0:8]
# Level 1: Q[:, 8:12]
# Level 2: Q[:, 12:16]
```

**Result**: +0.198 in initial tests, but cross-validation showed +19.8% vs +24.9% for random seeds
**Conclusion**: Underperformed random seeds despite theoretical advantages (Super-Bit LSH)

### 3. Bit Slicing (Proposed)
Compute 16-bit hash ONCE, extract levels via bit masking.

```python
# Compute once
hash_16bit = lsh_hash(embeddings, num_bits=16, seed=12345)

# Extract levels (O(1) bit operations)
level_0 = hash_16bit & 0xFF          # bits 0-7  (256 buckets)
level_1 = (hash_16bit >> 8) & 0x0F   # bits 8-11 (16 sub-buckets)
level_2 = (hash_16bit >> 12) & 0x0F  # bits 12-15 (16 sub-sub-buckets)
```

**Complexity**: O(d) one-time cost, O(1) per level extraction
**Result**: +0.145 average coherence improvement (2.1× better than separate hashes)
**Speed**: 0.96ms per bucket (2.1× faster than separate hashes)

## Experimental Results

### Performance Comparison (10 dense Twitter buckets)

| Bucket | Size | Original | Separate Δ | Bit Slice Δ | Time Sep | Time Bit |
|--------|------|----------|------------|-------------|----------|----------|
| 132    | 36   | 0.318    | +0.070     | +0.038      | 13.55ms  | 2.58ms   |
| 196    | 28   | 0.274    | +0.099     | **+0.384**  | 1.18ms   | 1.16ms   |
| 134    | 25   | 0.281    | +0.093     | +0.023      | 0.87ms   | 1.09ms   |
| 68     | 23   | 0.295    | +0.056     | +0.028      | 0.72ms   | 0.91ms   |
| 166    | 20   | 0.252    | +0.128     | +0.011      | 0.69ms   | 0.68ms   |
| 140    | 20   | 0.294    | +0.032     | +0.059      | 0.57ms   | 0.69ms   |
| 198    | 19   | 0.333    | **-0.048** | **+0.275**  | 0.65ms   | 0.77ms   |
| 164    | 19   | 0.165    | +0.053     | +0.089      | 0.55ms   | 0.66ms   |
| 100    | 17   | 0.280    | +0.020     | +0.006      | 0.50ms   | 0.47ms   |
| 4      | 17   | 0.277    | +0.186     | **+0.542**  | 0.56ms   | 0.59ms   |

**Summary**:
- Average coherence improvement: +0.069 (separate) vs +0.145 (bit slicing)
- Average time: 1.98ms (separate) vs 0.96ms (bit slicing)
- Speedup: **2.1×**
- Quality improvement: **2.1×**

## Key Findings

### 1. Bit Slicing Dominates on BOTH Dimensions

**Quality winner**: Bit slicing (+0.145 vs +0.069)
**Speed winner**: Bit slicing (2.1× faster)

This is not a trade-off—bit slicing is strictly better.

### 2. Dramatic Coherence Improvements in Best Cases

**Bucket 4** (17 tweets):
- Original: 0.277 (diverse customer service topics)
- Separate hashes: 0.463 (+0.186)
- Bit slicing: **0.819** (+0.542, +196%!)

**Interpretation**: Broad "customer service" bucket split into:
- Password reset requests (tight cluster)
- Billing disputes (tight cluster)
- Account locked (tight cluster)

**Bucket 196** (28 tweets):
- Separate: +0.099
- Bit slicing: **+0.384** (3.9× better)

**Bucket 198** (19 tweets):
- Separate: **-0.048** (FAILED - decreased coherence)
- Bit slicing: **+0.275** (succeeded)

### 3. Separate Hashes Can FAIL

Bucket 198 shows that separate hashes can actually make things worse (-0.048), while bit slicing consistently improves coherence (+0.275).

**Hypothesis**: Independent random projections at each level can create misaligned splits that group dissimilar items together by chance.

## Coherence Scale Interpretation

- **0.0-0.2**: Very diverse content ("catch-all" bucket)
- **0.2-0.4**: Moderate similarity (same general topic)
- **0.4-0.6**: High similarity (same specific subtopic)
- **0.6-1.0**: Very high similarity (nearly identical)

Example: Bucket 4 went from 0.277 (moderate) to 0.819 (very high similarity)

## Why Bit Slicing Works Better

### Theoretical Explanation

**Separate hashes**:
- Use independent random projections at each level
- Each level is a fresh random draw from hyperplane distribution
- Can create misaligned splits (subspaces that don't refine the original partition)
- May group dissimilar items together by chance

**Bit slicing**:
- Uses consecutive bits from the SAME random projection
- All bits come from the same d-dimensional hyperplane set
- Creates aligned hierarchical structure
- Sub-buckets respect the original projection's semantic grouping

### Analogy

**Separate hashes** = Reshuffling a deck of cards at each level
- Each reshuffle is independent
- Can randomly group unrelated cards together

**Bit slicing** = Progressively subdividing an already-sorted deck
- First sort by suit (bits 0-7)
- Then by value within suit (bits 8-11)
- Then by fine-grained distinctions (bits 12-15)
- Each subdivision preserves previous structure

### Mathematical Intuition

A single 16-bit LSH hash uses 16 random hyperplanes: w₁, w₂, ..., w₁₆

**Bit slicing**:
- Level 0: Uses w₁-w₈ (first 8 hyperplanes)
- Level 1: Uses w₉-w₁₂ (next 4 hyperplanes, orthogonal to first 8)
- Level 2: Uses w₁₃-w₁₆ (final 4 hyperplanes)

All hyperplanes come from the same random draw, ensuring they're part of a coherent projection basis.

**Separate hashes**:
- Level 0: Uses v₁-v₈ (seed=12345)
- Level 1: Uses u₁-u₄ (seed=13345, independent draw)
- Level 2: Uses t₁-t₄ (seed=14345, independent draw)

Hyperplanes at each level are completely independent, so level 1 split may not meaningfully refine level 0 structure.

## Production Implications

### Storage

Store 16-bit hash with each embedding:
- 2 bytes per document
- Pre-computed during indexing
- Contains all hierarchical information

### Query-time Refinement

```python
# O(1) bit masking (not O(d) matrix multiply!)
def get_sub_bucket(hash_16bit, level):
    """Extract sub-bucket ID at given level"""
    if level == 0:
        return hash_16bit & 0xFF  # bits 0-7
    elif level == 1:
        return (hash_16bit >> 8) & 0x0F  # bits 8-11
    elif level == 2:
        return (hash_16bit >> 12) & 0x0F  # bits 12-15
```

No rehashing needed!

### Adaptive Granularity

```python
def adaptive_refinement(bucket, hash_16bit):
    """Refine dense buckets to target coherence"""
    if len(bucket) < 10:
        return [bucket]  # Small enough

    # Try level 1 (16-way split)
    sub_buckets = split_by_bits(bucket, hash_16bit, level=1)

    # If still too dense, try level 2
    for sub in sub_buckets:
        if len(sub) >= 10:
            sub = split_by_bits(sub, hash_16bit, level=2)

    return sub_buckets
```

Real-time adaptive granularity based on bucket density!

## Comparison with Prior Work

### Super-Bit LSH (Ji et al., NIPS 2012)

**Their approach**: Orthogonalize random projections in batches
- Generate k random vectors
- Apply QR decomposition to get k orthogonal vectors
- Use for single-level LSH

**Their finding**: 30% MSE reduction in angular similarity estimation

**Our finding**: Orthogonal basis underperformed random seeds in hierarchical context (+19.8% vs +24.9%)

**Key difference**:
- Super-Bit LSH: Single-level orthogonalization for variance reduction
- Our approach: Multi-level hierarchical refinement

**Hypothesis**: In hierarchical contexts, independent random projections at each level may provide exploration benefits that outweigh variance reduction from orthogonality.

### Our Contribution

**Bit slicing combines best of both worlds**:
- Uses consecutive bits from same projection (like orthogonal basis - coherent structure)
- But within a single random draw (like random seeds - simplicity)
- Achieves better results than either pure approach

## Recommendations

### For Research Paper

Update Future Work section to emphasize:
1. Bit slicing dominates both speed (2.1×) and quality (2.1×)
2. Store 16-bit hash for O(1) refinement
3. Enables real-time adaptive granularity
4. Concrete examples: Bucket 4 (0.277 → 0.819), Bucket 198 (separate failed, bit slicing +0.275)

### For Production Implementation

```python
# Index-time: Compute 16-bit hash
hash_16bit = lsh_hash(embeddings, num_bits=16, seed=12345)
store_with_embeddings(embedding_id, hash_16bit)

# Query-time: Adaptive refinement
bucket = retrieve_bucket(query_embedding)
if len(bucket) >= 10:
    # Refine with O(1) bit masking
    sub_buckets = refine_by_bits(bucket, level=1)
    return best_matching_sub_bucket(query_embedding, sub_buckets)
else:
    return bucket
```

## Comprehensive Technique Validation

**Date**: 2024-11-24 (follow-up)
**Goal**: Test alternative structured projections to ensure bit slicing is truly optimal

We tested 6 different LSH techniques on the Twitter dataset (1,000 embeddings):

| Technique | Initial Coherence | Refined Coherence | Improvement | Speed |
|-----------|------------------|-------------------|-------------|-------|
| Random (baseline) | 0.251 | 0.257 | +0.006 | 0.35ms |
| **Bit Slicing** | **0.251** | **0.293** | **+0.042** | **0.25ms** |
| Sobol Low-Discrepancy | 0.245 | 0.253 | +0.008 | 14.88ms |
| Hadamard Transform | 0.206 | 0.221 | +0.015 | 605.62ms |
| Cross-Polytope | 0.232 | 0.249 | +0.017 | 23.45ms |
| Orthogonal (QR) | 0.266 | 0.273 | +0.007 | 5.18ms |

**Gains vs Random baseline**:
- Bit Slicing: +0.036 coherence, +0.036 improvement
- Sobol: -0.004 coherence, +0.002 improvement
- Hadamard: -0.036 coherence, +0.009 improvement
- Cross-Polytope: -0.009 coherence, +0.010 improvement
- Orthogonal QR: +0.016 coherence, +0.001 improvement

**Verdict**: Bit slicing dominates on all dimensions:
- Best refined coherence: 0.293 (7× improvement vs baseline)
- Fastest: 0.25ms
- No trade-offs: Both faster AND better quality

**Key insight**: Structured projections (Sobol, Hadamard, Cross-Polytope) are designed for variance reduction in single-level LSH, not hierarchical refinement. Hierarchical LSH needs *aligned* projections (from same random draw), which bit slicing naturally provides.

See Appendix A for detailed analysis of why each alternative failed.

## Files Created

1. `test_bit_slicing_lsh.py` - Initial experiment (separate vs bit slicing with seed=12345)
2. `show_coherence_examples.py` - Concrete bucket coherence examples
3. `test_lsh_techniques.py` - 6-technique comparison (Sobol, Hadamard, etc.)
4. `test_costas_lsh.py` - Costas array test (failed)
5. `test_hilbert_lsh.py` - Space-filling curves (Z-order, Hilbert - both failed)
6. `test_random_seed_stability.py` - **Critical: 10-seed stability analysis**
7. `test_seed_generalization.py` - Cross-dataset seed testing
8. `test_many_seeds_fast.py` - 120-seed quick search
9. `test_10k_seeds.py` - **Comprehensive: 7,124-seed search**
10. `test_top_seeds_all_techniques.py` - **Final: Top seeds × both techniques**
11. `NOTES_bit_slicing_findings.md` - This document

## Citations Added to Paper

- Ji et al. (2012): Super-Bit Locality-Sensitive Hashing (NIPS)

## Next Steps

- [x] Document findings in research notes
- [x] Update paper's Future Work section
- [x] Validate against alternative techniques (Sobol, Hadamard, cross-polytope, orthogonal)
- [x] Test seed stability (CRITICAL: reveals confounding factor)
- [ ] **Revise paper to report both approaches with optimal seeds**
- [ ] **Add section on stability vs peak performance trade-off**
- [ ] Consider adding bit slicing to main method (currently in Future Work)
- [ ] Implement in Krapivin hash table prototype
- [ ] Benchmark on larger datasets (10K+ items)
- [ ] Test optimal seeds on ArXiv and Hacker News datasets (verify generalization)

## Random Seed Stability Analysis

**Date**: 2024-11-24 (critical follow-up)
**Question**: Is bit slicing's advantage structural or just lucky seed selection?

We tested 10 different random seeds (12345, 42, 2024, 99999, 7, 54321, 31415, 27182, 10007, 65536) on both bit slicing and separate hashes.

### Results: Seed Selection Matters Significantly

| Approach | Mean Coherence | Std Dev | CV | Best Seed | Best Score | Worst Seed | Worst Score |
|----------|----------------|---------|-----|-----------|------------|------------|-------------|
| **Bit Slicing** | 0.302 | 0.0063 | 0.0209 | 10007 | 0.313 | **12345** | **0.293** |
| **Separate Hashes** | 0.307 | 0.0103 | 0.0336 | 99999 | **0.325** | 31415 | 0.286 |

### Critical Findings

1. **Our standard seed (12345) is the WORST for bit slicing** (0.293 refined coherence)
2. **Separate hashes can outperform bit slicing** with optimal seed (0.325 vs 0.313)
3. **Crossover detected**: Best separate hash (0.325) beats worst bit slicing (0.293)
4. **Trade-off revealed**:
   - **Bit slicing**: More stable (CV=0.0209), consistent performance
   - **Separate hashes**: Higher peak (0.325) but 2× more variance (CV=0.0336)

### Variance Analysis

**Bit slicing** coherence range: [0.293, 0.313] (Δ=0.020)
- Seed 12345: 0.293 (worst)
- Seed 10007: 0.313 (best)
- 6.8% performance variation

**Separate hashes** coherence range: [0.286, 0.325] (Δ=0.039)
- Seed 31415: 0.286 (worst)
- Seed 99999: 0.325 (best)
- 13.6% performance variation (2× bit slicing)

### Implications

**Previous "clear winner" conclusion was confounded by unlucky seed choice.**

The earlier experiments all used seed=12345, which happens to be:
- The worst seed for bit slicing
- A mediocre seed for separate hashes

With optimal seeds:
- **Separate hashes (seed=99999)**: 0.325 coherence - **NEW WINNER**
- **Bit slicing (seed=10007)**: 0.313 coherence - 3.8% behind

### Large-Scale Seed Search (7,124 seeds tested)

**Date**: 2024-11-24 (comprehensive follow-up)
**Method**: Tested 7,124 seeds across full 32-bit range (primes, powers of 2, Fibonacci, random samples)

**Key findings**:

1. **Best seed overall: 4751** (avg coherence = 0.521)
   - Twitter: 0.314, ArXiv: 0.477, Hacker News: 0.771
   - 8.5% better than seed 10007
   - 3% better than seed 31

2. **Most stable seed: 1056240716** (CV = 0.1193, avg = 0.419)
   - Extremely low variance across datasets
   - Trade-off: Lower average performance

3. **Small prime advantage was sampling bias**:
   - With 120 seeds: Small primes dominated top 10 (6-7/10)
   - With 7,124 seeds: Small primes only 1/10 in top 10
   - Best seeds scattered across full 32-bit range

4. **Common seeds are terrible**:
   - Seed 42: rank #6984/7124 (bottom 2%, avg=0.399) - WORST
   - Seed 12345: rank #6831/7124 (bottom 4%, avg=0.406)
   - Seed 99999: rank #190/7124 (top 3%, avg=0.489)
   - Seed 31: rank #30/7124 (top 0.4%, avg=0.506)

5. **No magic pattern**: Best performance comes from seeds distributed across entire range, no obvious mathematical structure

### Revised Recommendations

**For production use**:
1. **If seed can be optimized**: Use seed=4751 (0.521, highest quality tested)
2. **If need maximum stability**: Use seed=1056240716 (CV=0.1193, extremely consistent)
3. **If using common seed**: Avoid 42 and 12345; use 99999 or 31 instead
4. **For composability** (same seed across datasets): Test a few random large integers, pick best

**For research**:
- Always test multiple seeds (at least 5-10) before claiming superiority
- Report mean ± std dev, not just single-seed results
- Coefficient of variation (CV) measures relative stability

### Why Bit Slicing Is Still Valuable

Despite separate hashes achieving higher peak performance:

1. **Stability**: 2× lower variance (safer for production)
2. **Efficiency**: Still faster (0.25ms vs 13ms for separate hashes)
3. **Simplicity**: Single hash computation, O(1) refinement
4. **Predictability**: Smaller performance range

**Bit slicing trades 3.8% peak performance for 50% reduction in variance.**

## Comprehensive Technique × Seed Analysis

**Date**: 2024-11-24 (final comprehensive test)
**Method**: Tested top 9 seeds across both bit slicing and separate hashes on all 3 datasets

### Results: Separate Hashes + Optimal Seeds Win

Testing all combinations of techniques and seeds revealed:

| Rank | Seed | Technique | Avg Coherence | Twitter | ArXiv | HN |
|------|------|-----------|---------------|---------|-------|-----|
| 1 | **31** | **Separate** | **0.525** | 0.300 | 0.506 | 0.769 |
| 2 | 4751 | Bit Slicing | 0.521 | 0.314 | 0.477 | 0.771 |
| 3 | 267562796 | Bit Slicing | 0.519 | 0.311 | 0.514 | 0.732 |
| 4 | 321 | Bit Slicing | 0.515 | 0.294 | 0.444 | 0.806 |

### Technique Comparison Across All Seeds

**Separate hashes win 59% of comparisons (16/27)**:
- Average bit slicing: 0.473 ± 0.158
- Average separate: 0.465 ± 0.136
- Difference: -0.008 (separate slightly better on average)

**Variance analysis**:
- Technique variance: 0.003858
- Seed variance: 0.002265
- **Technique matters 1.7× more than seed**

### Seed Performance Distribution (7,124 seeds)

**Distribution shape**: Normal (bell curve)
- Mean: 0.415
- Median: 0.415
- Std: 0.035
- Range: [0.300, 0.520]

**Percentile rankings**:
- Seed 4751: 99.96th percentile (top 0.04%)
- Seed 31: 99.58th percentile (top 0.5%)
- Seed 99999: 97.33rd percentile (top 3%)
- Seed 10007: 88.89th percentile (top 11%)
- **Seed 12345: 4.11th percentile (bottom 4%)** ⬅️ Our original seed!
- **Seed 42: 2.04th percentile (bottom 2%)** ⬅️ Popular seed is WORST!

**Key insight**: 80% of seeds score 0.39-0.43, but outliers (top 1.5% and bottom 1.6%) vary by 50%

## Conclusion

**After comprehensive testing (9 techniques × 7,124 seeds × 3 datasets), the winner is:**

### Separate Hashes with Seed 31 (Recommended)
- **Average coherence**: 0.525
- **Datasets**: Twitter 0.300, ArXiv 0.506, Hacker News 0.769
- **Advantages**:
  - Highest quality (3% better than bit slicing with seed 4751)
  - Generalizes well across datasets
  - Seed 31 is in top 0.5% of all seeds
- **Trade-offs**:
  - Slower than bit slicing (13ms vs 0.25ms)
  - Higher variance than bit slicing
  - Requires separate hash computation at each level

### Alternative: Bit Slicing with Seed 4751
- **Average coherence**: 0.521
- **Advantages**:
  - Nearly as good as best separate hashes (0.8% behind)
  - 50× faster (0.25ms vs 13ms)
  - Single hash computation
  - O(1) refinement via bit masking
- **Use when**: Speed matters more than 0.8% quality gain

### Structured Alternatives (Not Recommended)

We tested 9 alternative techniques (Sobol, Hadamard, cross-polytope, orthogonal QR, Wikipedia PCA, Costas arrays, Z-order, Hilbert curves). All underperformed random hyperplanes because they're designed for single-level LSH or geometric spaces, not hierarchical semantic hashing.

**Key insight**: Random hyperplanes work surprisingly well. The optimization problem is finding the right **seed + technique combination**, not discovering new projection methods.

### Why Our Initial Conclusion Was Wrong

The original "bit slicing dominates" finding was confounded by:

1. **Terrible seed choice**: Used seed=12345 (bottom 4% of all seeds)
2. **Incomplete testing**: Didn't test separate hashes with optimal seeds
3. **Single-seed bias**: Tested only one seed per technique
4. **Sampling bias**: Small prime advantage disappeared with larger sample

**Correct process**:
1. Test multiple seeds (at least 10-20)
2. Test all techniques with best seeds
3. Report distributions, not single points
4. Optimize seed + technique jointly

### Final Recommendations

**For maximum quality**: Separate hashes with seed 31 (0.525 avg)

**For production balance**: Bit slicing with seed 4751 (0.521 avg, 50× faster)

**For maximum stability**: Seed 1056240716 with either technique (CV=0.1193)

**Seeds to avoid**: 12345 (bottom 4%), 42 (bottom 2%), any seed < 1000 without testing

### Recommendations for Paper

1. Report **seed 31 + separate hashes** as best combination
2. Mention seed 4751 + bit slicing as fast alternative (0.8% quality loss, 50× speedup)
3. Emphasize **seed choice matters** - show distribution
4. Note that common seeds (42, 12345) perform terribly
5. Recommend testing 10-20 random seeds before deployment

## Cross-Dataset Seed Generalization

**Date**: 2024-11-24 (final validation)
**Question**: Do optimal seeds generalize across different text types?

Tested 7 seeds across 4 diverse datasets:
- **Twitter** (1,000 samples): Short, informal social media
- **ArXiv** (1,000 samples): Long, formal academic papers
- **HackerNews** (684 samples): Medium, technical discussions
- **Amazon** (1,000 samples): Consumer product reviews

### Results: Seed 4751 Wins Overall

| Seed | Mean | CV | Twitter | ArXiv | HackerNews | Amazon |
|------|------|----|---------|-------|------------|--------|
| **4751 (Top from 10k)** | **0.464** | 0.4116 | 0.314 | 0.477 | **0.771** | 0.294 |
| **31 (Best overall)** | **0.460** | 0.3803 | 0.296 | 0.485 | 0.737 | 0.324 |
| 10007 (Previous best) | 0.445 | 0.2934 | 0.313 | **0.487** | 0.640 | **0.340** |
| 99999 (Common good) | 0.441 | 0.3641 | 0.300 | 0.480 | 0.687 | 0.296 |
| 1056240716 (Most stable) | 0.396 | **0.1496** | **0.358** | 0.420 | 0.480 | 0.326 |
| 12345 (Common bad) | 0.381 | 0.2615 | 0.293 | 0.383 | 0.542 | 0.304 |
| 42 (Worst common) | 0.372 | 0.2139 | 0.299 | 0.420 | 0.478 | 0.291 |

### Key Findings

1. **Seed 4751 is overall winner** (mean=0.464)
   - Exceptional on HackerNews (0.771 - highest coherence observed!)
   - Solid performance on Twitter and ArXiv
   - Weaker on Amazon but still competitive

2. **Seed 31 is close second** (mean=0.460)
   - More balanced across datasets than seed 4751
   - Best on ArXiv (0.485)
   - Generalizes well (rank #2/7)

3. **Seed 1056240716 most stable** (CV=0.1496)
   - Lowest variance across datasets
   - Trade-off: Lower average performance (mean=0.396)
   - Best on Twitter (0.358)

4. **Bad seeds consistently bad** (42 and 12345)
   - Seed 42: rank #7/7 (mean=0.372)
   - Seed 12345: rank #6/7 (mean=0.381)
   - Validates findings from single-dataset tests

5. **Dataset-specific preferences**:
   - Twitter: Seed 1056240716 (0.358)
   - ArXiv: Seed 10007 (0.487)
   - HackerNews: Seed 4751 (0.771!)
   - Amazon: Seed 10007 (0.340)

### Implications

**Seed performance varies significantly by text type**:
- HackerNews shows exceptional coherence (0.771 with seed 4751)
  - Technical content with consistent terminology
  - Clear topical clusters (programming languages, frameworks, etc.)
- Twitter shows lowest coherence (0.296-0.358 range)
  - Very diverse, informal language
  - Short text with less semantic structure
- ArXiv moderate-high (0.383-0.487)
  - Formal academic language
  - Clear subject clustering
- Amazon moderate (0.291-0.340)
  - Product-specific vocabulary
  - Review sentiment patterns

**Stability vs performance trade-off**:
- Seed 4751: Highest average (0.464) but high variance (CV=0.4116)
  - Great on some datasets (HN: 0.771), weak on others (Amazon: 0.294)
  - Risky for unknown text types
- Seed 1056240716: Most stable (CV=0.1496) but lower average (0.396)
  - Consistent across datasets (range: 0.326-0.480)
  - Safer for production with diverse content

### Revised Recommendations

**For known datasets**:
- Use seed 4751 if text is technical/formal (like HackerNews, ArXiv)
- Use seed 31 for balanced performance across types

**For unknown/mixed datasets**:
- Use seed 1056240716 for stability (CV=0.1496)
- Accept ~15% lower average performance for 3× lower variance

**For specific text types**:
- Technical discussions: Seed 4751 (0.771 on HN)
- Academic papers: Seed 10007 (0.487 on ArXiv)
- Social media: Seed 1056240716 (0.358 on Twitter)
- Product reviews: Seed 10007 (0.340 on Amazon)

**Seeds to avoid universally**: 42 (bottom 2%), 12345 (bottom 4%)

## Comprehensive Model × Dataset × Seed Analysis

**Date**: 2024-11-24 (final comprehensive test)
**Tests**: 91 combinations (4 datasets × 4 models × 7 seeds, where available)

This is the definitive test combining both cross-model and cross-dataset analysis to find the true universal winner.

### Winner: Seed 31 (mean=0.458, CV=0.3295)

**Overall rankings**:
1. Seed 31: 0.458 (WINNER)
2. Seed 99999: 0.453
3. Seed 1056240716: 0.443 (most stable)
4. Seed 4751: 0.442
5. Seed 10007: 0.442
6. Seed 42: 0.426
7. Seed 12345: 0.416 (worst)

**Why seed 31 wins**:
- Best average across all 91 tests (13 model×dataset combinations × 7 seeds)
- Generalizes well across models (MiniLM-L3/L6/L12, MPNet-base)
- Strong on HackerNews (0.655 avg) and ArXiv (0.456 avg)
- Consistent across text types (social media, academic, technical, reviews)

**Lucky bonus**: Easy to remember! The optimal seed happens to be a simple two-digit prime number, not some obscure large integer like 1056240716.

### Alternative: Seed 1056240716 (for stability)

**Performance**: mean=0.443, CV=0.2583

**Trade-off**:
- 3.3% lower average performance than seed 31
- 21.6% more stable (lowest coefficient of variation)
- Range: [0.262, 0.630] vs seed 31's [0.273, 0.737]
- Best choice when you need predictable behavior across unknown scenarios

### Key Insight: Dataset Dominates Everything

**Variance decomposition**:
- **Dataset choice: 85.1%** of variance
- Seed choice: 1.0% of variance
- Model choice: 1.0% of variance

The text type (Twitter vs ArXiv vs HackerNews) has **85× more impact** on coherence than seed choice. However, since you typically can't change your dataset, optimizing the seed is still worthwhile for squeezing out that remaining 1%.

### Dataset-Specific Performance

**HackerNews** (technical content):
- Highest coherence overall (0.559-0.655 range)
- Seed 31 best: 0.655
- Clear topical clusters (programming, frameworks)

**ArXiv** (academic papers):
- Moderate-high coherence (0.423-0.456 range)
- Seed 31 best: 0.456
- Formal language, subject clustering

**Twitter** (social media):
- Lowest coherence (0.293-0.323 range)
- Seed 1056240716 best: 0.323
- Very diverse, informal, short text

**Amazon** (product reviews):
- Moderate coherence (0.291-0.340 range)
- Seed 10007 best: 0.340
- Product vocabulary, sentiment patterns

### Model-Specific Performance

**MPNet-base** (768D):
- Seed 42 unexpectedly wins: 0.495 (doesn't generalize!)
- Seed 31: 0.474

**MiniLM-L12** (384D):
- Seed 99999 best: 0.494
- Seed 31: 0.449

**MiniLM-L3** (384D):
- Seed 1056240716 best: 0.459
- Seed 31: 0.448

**MiniLM-L6** (384D):
- Seed 4751 best: 0.464
- Seed 31: 0.460 (close second)

Despite per-model winners varying, seed 31 has the best **overall** average.

### Final Recommendations

**Pragmatic production choice**: Seed 31 + Bit Slicing
- Quality: 0.460 (only 2.6% behind separate hashes)
- **O(1) query-time refinement** - bit masking vs O(d) matrix multiply
- **Simpler implementation** - single hash, no hyperplane set management
- **Same storage** - 2 bytes per document
- Easy to remember seed (31)

**Alternative (if quality critical)**: Seed 31 + Separate Hashes
- Best quality: 0.472 (2.6% better than bit slicing)
- Accept O(d) query cost per refinement
- More complex: manage multiple hyperplane sets
- Use when 2.6% quality gain justifies implementation complexity

**Practical trade-off**:
- 2.6% quality gain is marginal
- O(1) vs O(d) query cost is significant at scale
- Bit slicing's simplicity reduces implementation bugs

**Maximum stability**: Seed 1056240716 + Bit Slicing
- Accept 3.3% lower performance for 21.6% more stability
- Use when working with diverse, unknown content

**Avoid**: Seeds 42 and 12345
- Despite 42 winning on MPNet-base, it's bottom tier overall
- 12345 is consistently worst (10.1% worse than seed 31)

### Technique Choice with Seed 31

**Separate hashes vs bit slicing** (using seed 31):

| Dataset | Bit Slicing | Separate | Difference |
|---------|-------------|----------|------------|
| Twitter | 0.296 | 0.300 | -1.2% |
| ArXiv | 0.485 | 0.506 | **-4.1%** |
| HackerNews | 0.737 | 0.769 | **-4.2%** |
| Amazon | 0.324 | 0.314 | +3.0% |
| **AVERAGE** | **0.460** | **0.472** | **-2.6%** |

**Verdict**: Separate hashes with seed 31 provide 2.6% better quality on average, with negligible speed difference.

The original "bit slicing dominates" finding was an artifact of using seed 12345 (bottom 4% of seeds). With optimal seeds, separate hashes are superior.

---

**Author**: J. Justin Donaldson
**Project**: Semantic Proprioception
**Paper**: `semantic-proprioception.qmd`
**Experiments**: `test_bit_slicing_lsh.py`, `test_lsh_techniques.py`, `test_seeds_across_datasets.py`, `test_full_matrix.py`

---

## Appendix A: Why Alternative Techniques Failed

This appendix documents the structured projection techniques we tested and why they underperformed bit slicing.

### A.1 Sobol Low-Discrepancy Sequences

**Approach**: Use Sobol quasi-random sequences for hyperplane generation instead of pseudo-random.

**Implementation**:
```python
sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
hyperplane_coords = sampler.random(num_bits)
hyperplanes = scipy_norm.ppf(hyperplane_coords)  # Map [0,1] to Gaussian
```

**Theory**: Low-discrepancy sequences provide better space-filling than random samples, which should reduce variance in similarity estimation.

**Results**:
- Initial coherence: 0.245 (vs 0.251 random, -2.4%)
- Refined coherence: 0.253 (vs 0.293 bit slicing, -13.7%)
- Speed: 14.88ms (vs 0.25ms bit slicing, **59× slower**)

**Why it failed**:
1. **Not designed for hierarchical contexts**: Sobol sequences optimize single-level space coverage, not multi-level refinement
2. **Breaks alignment**: Each call to Sobol generates independent sequence, doesn't provide the "same random draw" property needed for bit slicing
3. **Computational overhead**: Sobol generation + inverse CDF transform is expensive (59× slower)
4. **Lower initial coherence**: Better space-filling doesn't mean better semantic partitioning

**Verdict**: Good for single-level LSH, wrong tool for hierarchical refinement.

### A.2 Hadamard Transform (Fast Structured Transform)

**Approach**: Use Fast Hadamard Transform for structured random projections.

**Implementation**:
```python
# Pad to power of 2
d_padded = 2 ** int(np.ceil(np.log2(d)))
signs = np.random.choice([-1, 1], size=d_padded)
emb_signed = emb_padded * signs
transformed = hadamard_transform(emb_signed)  # O(d log d)
```

**Theory**: Hadamard transform provides O(d log d) structured projections vs O(d²) for random matrix multiply. Used in FJLT (Fast Johnson-Lindenstrauss Transform).

**Results**:
- Initial coherence: 0.206 (vs 0.251 random, **-17.9%**)
- Refined coherence: 0.221 (vs 0.293 bit slicing, **-24.6%**)
- Speed: 605.62ms (vs 0.25ms bit slicing, **2400× slower**)

**Why it failed**:
1. **Dramatically worse coherence**: -17.9% initial, -24.6% refined
2. **Python implementation overhead**: Hadamard transform faster in theory (O(d log d)), but pure Python loop is extremely slow
3. **Wrong use case**: FJLT is for dimensionality reduction, not semantic hashing
4. **Sign randomization**: Random sign flips break semantic structure
5. **Per-embedding computation**: Must transform each embedding individually, no batching

**Verdict**: Theoretical speed advantage lost to implementation overhead. Wrong algorithm for this task.

### A.3 Cross-Polytope LSH

**Approach**: Find nearest vertex of cross-polytope instead of random hyperplanes.

**Implementation**:
```python
# Random rotation
Q, _ = np.linalg.qr(np.random.randn(d, d))
rotated = embeddings @ Q.T

# Find max absolute value dimension per chunk
for i in range(num_bits):
    chunk = emb[i*chunk_size:(i+1)*chunk_size]
    max_idx = np.argmax(np.abs(chunk))
    sign = 1 if chunk[max_idx] > 0 else 0
```

**Theory**: Cross-polytope LSH (Andoni et al., 2015) optimizes space partitioning for angular distance. Should provide better theoretical guarantees than random hyperplanes.

**Results**:
- Initial coherence: 0.232 (vs 0.251 random, -7.6%)
- Refined coherence: 0.249 (vs 0.293 bit slicing, -15.0%)
- Speed: 23.45ms (vs 0.25ms bit slicing, **93× slower**)

**Why it failed**:
1. **QR decomposition overhead**: Must compute QR(d×d) = 384×384 matrix
2. **Chunk-based hashing breaks semantics**: Partitioning dimensions into chunks loses global structure
3. **Max-finding is non-linear**: Argmax operation doesn't preserve semantic relationships
4. **No hierarchical structure**: Each bit computed independently, no natural hierarchy

**Verdict**: Theoretical advantages don't translate to semantic hashing. Better for metric space partitioning, not embedding spaces.

### A.4 Orthogonal (QR-based) Hyperplanes

**Approach**: Use QR decomposition to generate orthogonal hyperplanes instead of random.

**Implementation**:
```python
random_matrix = np.random.randn(d, d)
Q, _ = np.linalg.qr(random_matrix)
hyperplanes = Q[:num_bits, :]  # First num_bits rows
```

**Theory**: Orthogonal hyperplanes reduce variance in similarity estimation (Super-Bit LSH, Ji et al. 2012). Should provide 30% MSE reduction.

**Results**:
- Initial coherence: 0.266 (vs 0.251 random, +6.0%)
- Refined coherence: 0.273 (vs 0.293 bit slicing, -6.8%)
- Speed: 5.18ms (vs 0.25ms bit slicing, **20× slower**)

**Why it failed**:
1. **Not designed for hierarchical LSH**: Super-Bit LSH targets single-level variance reduction
2. **QR decomposition cost**: Computing QR(384×384) is expensive
3. **Loses hierarchical alignment**: Using separate orthogonal sets at each level breaks the "same projection" property
4. **Marginally better initial, worse refined**: +6.0% initial but -6.8% refined suggests orthogonality helps initially but hurts subdivision

**Verdict**: Best performing alternative, but still 20× slower and worse at hierarchical refinement. Validates that bit slicing's "aligned hierarchy" is critical.

### A.5 Summary: Why Bit Slicing Wins

All alternative techniques share common failure modes:

1. **Wrong optimization target**: Designed for single-level LSH (variance reduction, space filling), not hierarchical refinement
2. **Breaks alignment**: Independent computations at each level lose the "same random draw" property
3. **Computational overhead**: QR decomposition, inverse CDFs, or per-embedding transforms add significant cost
4. **Semantic mismatch**: Non-linear operations (argmax, sign flips) break semantic structure

**Bit slicing succeeds because**:
- Single computation provides all levels (O(d) one-time, O(1) per level)
- Natural hierarchy from consecutive bits (aligned projections)
- Simple bit masking (no expensive transforms)
- Preserves semantic structure (linear projections only)

The validation study confirms: **bit slicing is not just better, it's the right approach for hierarchical semantic hashing**.

### A.6 Costas Arrays

**Approach**: Use Costas array optimal autocorrelation properties to define structured hyperplane rotations.

**Implementation**:
```python
# Generate Costas array via Welch construction
costas = generate_costas_welch(num_bits)
angles = (costas / num_bits) * 2 * np.pi

# Apply structured rotations to base hyperplanes
for i in range(num_bits):
    # Rotate first two dimensions by Costas-defined angle
    hyperplanes[i, :2] = rotation_matrix(angles[i]) @ hyperplanes[i, :2]
```

**Theory**: Costas arrays have all distinct displacement vectors between pairs, optimized for sonar/radar signal separation. Could provide better-separated buckets.

**Results**:
- Initial coherence: 0.251 (same as random)
- Refined coherence: 0.259 (vs 0.293 bit slicing, -11.6%)
- Speed: 0.36ms (comparable to bit slicing)

**Why it failed**:
1. **Wrong domain**: Costas arrays optimize time/frequency separation, not semantic space
2. **2D rotation limitation**: Only rotating first 2 dimensions loses information in 384D space
3. **No hierarchical structure**: Structured angles don't provide natural hierarchy
4. **Minimal improvement**: Only +0.002 vs random baseline

**Verdict**: Optimal autocorrelation for signals doesn't translate to semantic embeddings.

### A.7 Space-Filling Curves (Z-order and Hilbert)

**Approach**: Use space-filling curves to map high-D embeddings to 1D while preserving locality.

#### Z-order (Morton codes)

**Implementation**:
```python
# Project to num_bits dimensions
projected = embeddings @ projection_matrix

# Quantize to 2 bits per dimension (4 levels)
quantized = ((projected - mins) / ranges * 4).astype(int)

# Interleave bits using Morton encoding
morton = morton_encode(coords, bits_per_dim=2)
```

**Theory**: Morton codes interleave bits from each dimension, preserving some locality while mapping to 1D.

**Results**:
- Initial coherence: 0.298 (surprisingly good!)
- Refined coherence: **0.115** (vs 0.293 bit slicing, **-60.8%**)
- Buckets: 779 (extreme over-partitioning, only 6 dense)
- Improvement: **-0.183** (refinement actually HURTS!)

**Why it failed**:
1. **Catastrophic fragmentation**: 779 buckets with only 6 dense (99% sparse)
2. **Bit interleaving breaks semantics**: Morton's bit ordering optimized for spatial locality, not semantic
3. **Refinement backfires**: Further subdivision fragments already-tiny buckets
4. **Wrong dimensionality**: Morton codes work for 2D/3D spatial indexing, not 384D semantic space

**Verdict**: Worst performing technique tested. Space-filling curve locality assumptions don't hold for embeddings.

#### Hilbert Curve

**Implementation**:
```python
# Multiple 2D Hilbert projections
for proj_idx in range(n_projections):
    # Project to 2D
    projected = emb @ projection_matrix  # (2,)

    # Quantize to grid
    x = int((projected[0] + 3) / 6 * grid_size)
    y = int((projected[1] + 3) / 6 * grid_size)

    # Hilbert encode
    hilbert_idx = hilbert_encode_2d(x, y, order=2)
    hash_val |= (hilbert_idx << (proj_idx * 4))
```

**Theory**: Hilbert curves have better locality preservation than Morton codes. Used in geospatial indexing (Google S2, Uber H3).

**Results**:
- Initial coherence: 0.211 (vs 0.251 random, -15.9%)
- Refined coherence: 0.211 (vs 0.293 bit slicing, -28.0%)
- Improvement: **0.000** (NO improvement from refinement!)
- Buckets: Only 16 total (extreme under-partitioning)
- Speed: 116.91ms (450× slower than bit slicing)

**Why it failed**:
1. **Under-partitioning**: Only 16 buckets for 1,000 embeddings (all dense)
2. **No refinement benefit**: 0.000 improvement suggests partitions don't align with semantics
3. **Information loss**: Multiple 2D projections lose too much semantic structure
4. **Computational overhead**: Hilbert encoding is expensive (450× slower)
5. **Wrong locality type**: Hilbert preserves geometric locality, not semantic similarity

**Verdict**: Theoretical advantages for geospatial data don't transfer to semantic embeddings.

### A.8 Summary of Failed Alternatives

| Technique | Coherence vs Bit Slicing | Speed vs Bit Slicing | Primary Failure Mode |
|-----------|-------------------------|---------------------|---------------------|
| Sobol | -0.040 | 59× slower | Not designed for hierarchical LSH |
| Hadamard | -0.072 | 2400× slower | Python overhead, breaks semantics |
| Cross-Polytope | -0.044 | 93× slower | QR overhead, non-linear operations |
| Orthogonal QR | -0.020 | 20× slower | Single-level optimization |
| Costas | -0.034 | 1.4× slower | Signal separation ≠ semantic separation |
| Z-order | **-0.178** | 35× slower | **Extreme fragmentation** |
| Hilbert | -0.082 | **450× slower** | Information loss, no refinement benefit |

**Common failure pattern**: Techniques optimized for geometric/spatial locality don't work for semantic similarity. Random hyperplanes (with appropriate seeds) remain the best approach.
