# Streamlit Demo Deployment

This document explains how to deploy the Semantic Proprioception Streamlit demo.

## Quick Deploy to Streamlit Cloud

1. **Push to GitHub**:
```bash
git add .
git commit -m "Add Hacker News dataset and prepare for Streamlit deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo: `semantic-proprioception`
   - Main file: `semantic_proprioception_demo.py`
   - Click "Deploy"

3. **Wait for deployment** (~2-3 minutes)

## Files Included

- `semantic_proprioception_demo.py` - Main app
- `requirements.txt` - Python dependencies
- `semantic_proprioception_data/` - Twitter embeddings (8.6 MB)
- `arxiv_demo_data/` - ArXiv embeddings (9.5 MB)
- `hackernews_demo_data/` - Hacker News embeddings (5.6 MB)
- `krapivin-python/` - Python bindings for LSH operations

**Total size**: ~24 MB (well under Streamlit's free tier limits)

## Requirements

All dependencies are in `requirements.txt`:
- streamlit==1.51.0
- polars==1.35.2
- numpy==2.3.5
- sentence-transformers==5.1.2

## App Features

- 3 datasets (Twitter, ArXiv, Hacker News)
- 4 embedding models
- Automatic theme discovery
- Interactive exploration

## Troubleshooting

**If deployment fails**:
1. Check GitHub repo is public
2. Verify all data files are committed
3. Check requirements.txt has correct versions
4. Look at Streamlit Cloud logs

**If app loads slowly**:
- First load caches data (~30 seconds)
- Subsequent loads are fast (<1 second)

## Post-Deployment

Update README.md with live demo URL:
```markdown
🔗 **[Live Demo](https://YOUR-APP-NAME.streamlit.app)**
```
