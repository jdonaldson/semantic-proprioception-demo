# Semantic Proprioception - Datasets

This demo showcases semantic theme discovery across three diverse datasets using 4 different embedding models and LSH-based clustering.

## Datasets

### 1. Twitter Customer Support (1,000 tweets)
- **Source**: Twitter customer service conversations
- **Size**: ~8.6 MB (all models + indexes)
- **Content**: Customer support tweets covering common issues (billing, technical problems, account access)
- **Key Themes**: Password resets, billing issues, technical support, service outages
- **Directory**: `semantic_proprioception_data/`

### 2. ArXiv Research Papers (1,000 papers)
- **Source**: ArXiv academic papers
- **Size**: ~9.5 MB (all models + indexes)
- **Categories**:
  - Artificial Intelligence (cs.AI)
  - Machine Learning (cs.LG, stat.ML)
  - Computer Vision (cs.CV)
  - Natural Language Processing (cs.CL)
  - General Physics (physics.gen-ph)
  - Combinatorics (math.CO)
  - Genomics (q-bio.GN)
  - Econometrics (econ.EM)
  - Astrophysics (astro-ph.GA)
- **Content**: Paper titles and abstracts (~1,400 chars avg)
- **Key Themes**: Deep learning, neural networks, optimization, quantum mechanics, genomics
- **Directory**: `arxiv_demo_data/`

### 3. Hacker News (684 posts)
- **Source**: Hacker News posts and discussions
- **Size**: ~5.6 MB (all models + indexes)
- **Categories**:
  - Top Stories
  - Best Stories
  - New Stories
  - Ask HN
  - Show HN
- **Content**: Post titles and text (~255 chars avg)
- **Key Themes**: Startups, technology, programming, AI/ML, privacy, open source
- **Directory**: `hackernews_demo_data/`

## Total Size
- Twitter: 8.6 MB
- ArXiv: 9.5 MB
- Hacker News: 5.6 MB
- **Combined: ~23.7 MB across all datasets and models**
- Well within free hosting limits (< 100 MB)

## Embedding Models

All datasets use the same 4 models for consistent comparison:

1. **MiniLM-L3** (384-dim, 8 LSH bits)
   - Fastest, smallest (61MB model)
   - Good for quick similarity

2. **MiniLM-L6** (384-dim, 8 LSH bits)
   - Balanced speed/quality (90MB model)
   - General purpose

3. **MiniLM-L12** (384-dim, 8 LSH bits)
   - Better quality (120MB model)
   - More accurate clustering

4. **MPNet-base** (768-dim, 10 LSH bits)
   - Highest quality (420MB model)
   - Best semantic understanding

## LSH Configuration

All models use:
- **Fixed seed**: 12345 (for composability)
- **Delta**: 0.3 (collision probability)
- **Min bucket size**: 5 items (dense bucket threshold)

## Theme Discovery

The demo automatically discovers themes by:
1. Computing LSH bucket density (O(1) with Krapivin hash tables)
2. Identifying dense buckets (≥5 items with similar embeddings)
3. Generating semantic labels via LLM or keyword extraction
4. Merging similar themes using Jaccard similarity

## Running the Demo

```bash
# 1. Fetch data (if not already done)
python fetch_twitter_data.py      # Already included
python fetch_arxiv_data.py
python fetch_hackernews_data.py

# 2. Precompute embeddings
python precompute_embeddings.py            # Twitter
python precompute_arxiv_embeddings.py
python precompute_hackernews_embeddings.py

# 3. Run demo
streamlit run semantic_proprioception_demo.py
```

## Dataset Comparison

| Dataset | Items | Avg Length | Domain | Themes |
|---------|-------|------------|--------|--------|
| Twitter | 1,000 | ~200 chars | Customer Support | Concrete, action-oriented |
| ArXiv | 1,000 | ~1,400 chars | Academic Research | Abstract, technical concepts |
| Hacker News | 684 | ~255 chars | Tech Discussion | News, opinions, projects |

Each dataset reveals different semantic structures:
- **Twitter**: Short, focused on specific problems and solutions
- **ArXiv**: Longer, explores complex research topics
- **Hacker News**: Mix of short headlines and detailed discussions

The same embedding models produce different clustering patterns for each dataset, demonstrating how semantic proprioception adapts to domain-specific content.
