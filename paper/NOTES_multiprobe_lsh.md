# Multi-Probe LSH Findings

**Date**: 2025-01-24
**Dataset**: Twitter (1,000 tweets), MiniLM-L6 (384D embeddings)
**LSH Config**: 8-bit buckets, seed 31, bit slicing

## Problem

Standard LSH has low recall (~4%) because similar items don't always hash to the same bucket. Query "stolen" finds only 1 of 4 relevant tweets in primary bucket 0xe8.

## Solution: Multi-Probe LSH

**Idea**: Check nearby buckets by flipping hash bits
- Primary bucket: 0xe8 = `11101000`
- 1-bit flip: Flip each bit individually (9 buckets total)
- 2-bit flip: Flip all pairs of bits (37 buckets total)
- 3-bit flip: Flip all triples of bits (93 buckets total)

**Example**: "stolen" query
- Doc 747 at bucket 0xe8 (Hamming distance 0) - same bucket
- Doc 37 at bucket 0xca (Hamming distance 2) - bits 1,5 differ
- Doc 671 at bucket 0xfc (Hamming distance 2) - bits 2,4 differ
- Doc 713 at bucket 0x64 (Hamming distance 3) - bits 2,3,7 differ

## Recall vs Cost Trade-off

Tested on 5 diverse queries, averaged:

| Strategy | Recall@10 | Candidates | Speed vs Brute Force |
|----------|-----------|------------|----------------------|
| Standard LSH | 4% | 5 | 83× faster |
| 1-bit flip | 20% | 40 | 43× faster |
| **2-bit flip** | **46%** | **172** | **14× faster** |
| 3-bit flip | 64% | 419 | 6× faster |
| Brute force | 100% | 1000 | 1× (baseline) |

**Sweet spot**: 2-bit flip
- 46% recall (finds nearly half of top-10 results)
- Scans only 17% of dataset
- 14× faster than brute force
- 60% precision@1 (top result is correct 60% of time)

## Bit Density Analysis

### Popcount Effects

**Finding**: Bit density (popcount) significantly affects bucket quality
- Buckets with popcount=4 have highest coherence (0.267)
- Buckets with popcount=0,2 have lower coherence (~0.21)
- Popcount explains **34.9% of bucket size variance**
- **Verdict**: Popcount has SIGNIFICANT effect on bucket coherence

### Discriminative Power

Measured coherence difference when bit=0 vs bit=1:

| Rank | Bit | Discriminative Power | Coh(0) | Coh(1) |
|------|-----|---------------------|--------|--------|
| 1 | 6 | 0.0293 | 0.229 | 0.258 |
| 2 | 4 | 0.0284 | 0.261 | 0.233 |
| 3 | 5 | 0.0201 | 0.239 | 0.259 |
| ... | | | | |
| 8 | 3 | 0.0053 | 0.248 | 0.253 |

### Bit Balance

Most imbalanced bits (create skewed partitions):
- Bit 1: 692 zeros, 308 ones (balance=0.445)
- Bit 6: 327 zeros, 673 ones (balance=0.486)

Most balanced (split data evenly):
- Bit 7: 451 zeros, 549 ones (balance=0.821)
- Bit 3: 564 zeros, 436 ones (balance=0.773)

## Does Discriminative Power Help Multi-Probe?

**Hypothesis**: Prioritizing discriminative bits in multi-probe should improve recall under budget constraints.

**Test**: Budget-limited probing with 3 strategies:
1. **Most discriminative first**: [6, 4, 5, 2, 7, 1, 0, 3]
2. **Least discriminative first**: [3, 0, 1, 7, 2, 5, 4, 6]
3. **Uniform (sequential)**: [0, 1, 2, 3, 4, 5, 6, 7]

### Results

| Budget | Uniform | Most Disc | Least Disc |
|--------|---------|-----------|------------|
| 5 buckets | 12.0% | 8.0% | 16.0% |
| 10 buckets | 24.0% | 20.0% | 24.0% |
| 15 buckets | 34.0% | 20.0% | 26.0% |
| 20 buckets | 42.0% | 22.0% | 38.0% |
| **Average** | **28.0%** | **17.5%** | **26.0%** |

**Verdict**: ✓ **Uniform strategy wins**
- Simple sequential bit order (0,1,2,...) best overall
- Most discriminative bits are WORST (-37.5% vs uniform)
- Least discriminative bits are better but still lose to uniform

### Why Discriminative Power Doesn't Help

1. **Hamming distance matters more**: Items at distance 1 are more semantically similar than distance 2, regardless of which specific bits differ

2. **Discriminative power measures wrong thing**: High discriminative power means a bit separates different clusters well, but for search we want to explore NEARBY semantic space, not jump to distant clusters

3. **Sequential ordering explores systematically**: Bits 0,1,2,... explores hash space structure more evenly than jumping around based on data-dependent properties

**Conclusion**: Don't bother computing bit discriminative power at index time - it doesn't help query-time decisions.

## Query Complexity Analysis

Breakdown for 2-bit flip (sweet spot):

| Operation | Time (ms) | % of Total | Complexity |
|-----------|-----------|------------|------------|
| 1. Hash query | 0.004 | 5.8% | O(d) |
| 2. **Generate probes** | **0.004** | **6.0%** | **O(C(b,k))** |
| 3. Lookup buckets | 0.004 | 4.8% | O(probes) |
| 4. Deduplicate | 0.007 | 9.1% | O(C) |
| 5. **Compute similarities** | **0.051** | **69.7%** | **O(C × d)** |
| 6. Sort top-k | 0.004 | 4.8% | O(C log C) |
| **Total** | **0.073** | | |

Where:
- b = 8 bits (constant)
- k = number of flips (1, 2, or 3)
- C = candidates found (~172 for 2-bit flip)
- d = 384 (embedding dimension)

**Key insights**:
- ✓ Bit flipping is CHEAP: < 6% of query time, O(1) in practice
- ✓ Bucket lookup is CHEAP: < 5% of query time
- ✗ Similarity computation DOMINATES: 70% of query time, O(C × d)

**Bottom line**: Don't worry about bit flipping cost. The real trade-off is recall vs similarity computation cost (which scales with number of candidates).

## Recommendations for Production

1. **Use 2-bit flip as default**
   - Best balance: 46% recall, 14× speedup
   - Scans only 17% of dataset
   - Dramatically better than standard LSH (4% recall)

2. **Use uniform bit ordering (0,1,2,...)**
   - No need to compute discriminative power at index time
   - Simple sequential flipping works best
   - Avoid premature optimization based on bit statistics

3. **Tune based on use case**:
   - **High recall needed**: Use 3-bit flip (64% recall, 6× speedup)
   - **Low latency critical**: Use 1-bit flip (20% recall, 43× speedup)
   - **Balanced**: Use 2-bit flip (46% recall, 14× speedup)

4. **For Krapivin index**:
   - Store bit flip level as query-time parameter
   - No need to store discriminative power in index
   - Keep index simple and compact

## Theoretical Foundation

Multi-probe LSH explores buckets at increasing Hamming distances from the query bucket:
- **Hamming distance 0**: Primary bucket (1 bucket)
- **Hamming distance 1**: 1-bit flips (8 buckets)
- **Hamming distance 2**: 2-bit flips (28 buckets)
- **Hamming distance 3**: 3-bit flips (56 buckets)

Formula: C(b, k) = b! / (k!(b-k)!) where b=8 bits

This is equivalent to searching a neighborhood in the hash space, with the assumption that semantically similar items hash to nearby buckets (low Hamming distance).

**LSH fundamental trade-off**: Precision/recall vs speed
- More probes → Better recall but higher cost
- Multi-probe smooths this trade-off curve significantly

## References

- Lv et al. (2007): "Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search"
- Our seed analysis: Seed 31 optimal across models and datasets
- Our bit slicing: O(1) hierarchical refinement via bit masking

## Files

- `test_multiprobe_lsh.py`: Basic multi-probe implementation and recall tests
- `test_hamming_distance.py`: Hamming distance analysis for "stolen" query
- `test_multiprobe_queries.py`: Comprehensive recall evaluation across queries
- `test_bit_flip_pattern.py`: Visualization of bit flipping patterns
- `test_bucket_bit_density.py`: Bit density and discriminative power analysis
- `test_discriminative_multiprobe.py`: Test if discriminative power helps
- `test_budget_limited_multiprobe.py`: Budget-limited probing comparison
- `test_inverse_discriminative.py`: Test least discriminative first hypothesis
- `test_multiprobe_complexity.py`: Query complexity breakdown

## Next Steps

1. Integrate 2-bit flip multi-probe into Streamlit demo
2. Test on larger datasets (10K, 100K, 1M documents)
3. Benchmark against vector databases (Qdrant, Pinecone)
4. Document in Krapivin hash table design
