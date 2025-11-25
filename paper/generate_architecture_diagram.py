#!/usr/bin/env python3
"""
Generate architecture diagram for Krapivin + LSH system
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def figure_architecture():
    """Generate architecture diagram showing LSH + Krapivin pipeline"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Semantic Proprioception Architecture',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # ========== Stage 1: Text Input ==========
    y_start = 8.5

    # Text samples
    texts = [
        ("tweet1.txt", "Password reset\nnot working", '#e8f4f8'),
        ("tweet2.txt", "Cannot login\nto account", '#e8f4f8'),
        ("arxiv1.txt", "Deep learning\nfor NLP", '#fff4e6'),
    ]

    x_offset = 1
    for i, (filename, content, color) in enumerate(texts):
        box = FancyBboxPatch((x_offset + i*1.8, y_start - 0.8), 1.4, 0.7,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_offset + i*1.8 + 0.7, y_start - 0.45, content,
                ha='center', va='center', fontsize=8, style='italic')
        ax.text(x_offset + i*1.8 + 0.7, y_start - 0.1, filename,
                ha='center', va='center', fontsize=7, fontweight='bold')

    ax.text(7, y_start + 0.3, 'Stage 1: Text Documents',
            ha='center', fontsize=12, fontweight='bold')

    # ========== Stage 2: Embeddings ==========
    y_embed = 6.5

    # Arrow down
    ax.arrow(4, y_start - 0.9, 0, -0.8, head_width=0.2, head_length=0.15,
             fc='black', ec='black', linewidth=2)
    ax.text(4.5, y_start - 1.3, 'sentence-transformers', fontsize=9, style='italic')

    # Embedding vectors
    vectors = [
        (1.5, "v₁ = [0.2, -0.5, 0.8, ...]", '#e8f4f8'),
        (4.5, "v₂ = [0.3, -0.4, 0.9, ...]", '#e8f4f8'),
        (7.5, "v₃ = [-0.6, 0.7, -0.2, ...]", '#fff4e6'),
    ]

    for x, text, color in vectors:
        box = FancyBboxPatch((x - 0.5, y_embed - 0.3), 2.5, 0.5,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.75, y_embed - 0.05, text, ha='center', va='center',
                fontsize=8, family='monospace')

    ax.text(7, y_embed + 0.5, 'Stage 2: Embeddings (384-d or 768-d)',
            ha='center', fontsize=12, fontweight='bold')

    # ========== Stage 3: LSH Hashing ==========
    y_lsh = 4.5

    # Arrow down
    ax.arrow(4, y_embed - 0.4, 0, -1.3, head_width=0.2, head_length=0.15,
             fc='black', ec='black', linewidth=2)
    ax.text(4.5, y_embed - 1.0, 'LSH (fixed seed=12345)', fontsize=9, style='italic')
    ax.text(4.5, y_embed - 1.3, '8 random hyperplanes', fontsize=8, style='italic')

    # LSH hash computation visual
    box = FancyBboxPatch((1, y_lsh - 0.6), 11, 1.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='#2c5aa0', facecolor='#e3f2fd', linewidth=2)
    ax.add_patch(box)

    # Show hash computation
    ax.text(6.5, y_lsh + 0.2, 'h(v) = [sign(w₁·v), sign(w₂·v), ..., sign(w₈·v)]',
            ha='center', fontsize=9, family='monospace', fontweight='bold')

    hash_examples = [
        (2, "v₁ → 10110101 → bucket 181", '#e8f4f8'),
        (6.5, "v₂ → 10110101 → bucket 181", '#e8f4f8'),
        (10, "v₃ → 01101100 → bucket 108", '#fff4e6'),
    ]

    for x, text, color in hash_examples:
        ax.text(x, y_lsh - 0.25, text, ha='center', fontsize=8,
                family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    ax.text(7, y_lsh + 0.8, 'Stage 3: LSH Bucketing (Collision Detection)',
            ha='center', fontsize=12, fontweight='bold')

    # ========== Stage 4: Krapivin Index ==========
    y_krapivin = 2.0

    # Arrow down
    ax.arrow(4, y_lsh - 0.7, 0, -1.5, head_width=0.2, head_length=0.15,
             fc='black', ec='black', linewidth=2)

    # Krapivin hash table structure
    box = FancyBboxPatch((0.5, y_krapivin - 1.2), 13, 2.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c62828', facecolor='#ffebee', linewidth=2)
    ax.add_patch(box)

    ax.text(7, y_krapivin + 1.0, 'Stage 4: Krapivin Hash Table (O(1) Density Queries)',
            ha='center', fontsize=12, fontweight='bold')

    # Show hierarchical levels
    ax.text(1.5, y_krapivin + 0.5, 'Hierarchical Levels:', fontsize=9, fontweight='bold')

    # Dense buckets (high levels)
    dense_box = Rectangle((1, y_krapivin - 0.1), 3.5, 0.5,
                          facecolor='#ff6b6b', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(dense_box)
    ax.text(2.75, y_krapivin + 0.15, 'Level 7-9: Dense Buckets',
            ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(2.75, y_krapivin - 0.4, 'bucket 181: [tweet1, tweet2]',
            ha='center', fontsize=7, family='monospace')
    ax.text(2.75, y_krapivin - 0.65, 'count = 2, theme = "login issues"',
            ha='center', fontsize=7, style='italic')

    # Medium buckets (mid levels)
    medium_box = Rectangle((5, y_krapivin - 0.1), 3.5, 0.5,
                           facecolor='#ffd93d', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(medium_box)
    ax.text(6.75, y_krapivin + 0.15, 'Level 4-6: Medium',
            ha='center', fontsize=8, fontweight='bold')
    ax.text(6.75, y_krapivin - 0.4, 'bucket 142: [...]',
            ha='center', fontsize=7, family='monospace')

    # Sparse buckets (low levels)
    sparse_box = Rectangle((9, y_krapivin - 0.1), 3.5, 0.5,
                           facecolor='#95e1d3', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(sparse_box)
    ax.text(10.75, y_krapivin + 0.15, 'Level 0-3: Sparse',
            ha='center', fontsize=8, fontweight='bold')
    ax.text(10.75, y_krapivin - 0.4, 'bucket 108: [arxiv1]',
            ha='center', fontsize=7, family='monospace')

    # Key insight box
    insight_box = FancyBboxPatch((9.5, 0.2), 4, 1.2,
                                boxstyle="round,pad=0.1",
                                edgecolor='#1565c0', facecolor='#e3f2fd', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(11.5, 1.0, '💡 Key Insight', ha='center', fontsize=10, fontweight='bold')
    ax.text(11.5, 0.7, 'Query "buckets with ≥5 items"', ha='center', fontsize=8)
    ax.text(11.5, 0.45, '→ Returns dense buckets in O(1)', ha='center', fontsize=8)

    # Composability note
    comp_box = FancyBboxPatch((0.5, 0.2), 4, 1.2,
                             boxstyle="round,pad=0.1",
                             edgecolor='#6a1b9a', facecolor='#f3e5f5', linewidth=2)
    ax.add_patch(comp_box)
    ax.text(2.5, 1.0, '🔗 Composability', ha='center', fontsize=10, fontweight='bold')
    ax.text(2.5, 0.7, 'Fixed seed → same buckets', ha='center', fontsize=8)
    ax.text(2.5, 0.45, 'across all datasets', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_architecture.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure_architecture.png", dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to {OUTPUT_DIR / 'figure_architecture.pdf'}")
    plt.close()

def figure_krapivin_detail():
    """Generate detailed Krapivin hierarchical structure diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'Krapivin Hash Table: Hierarchical Structure',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Hierarchical levels
    levels = [
        (9, "Dense", "#d32f2f", 3),
        (7, "Dense", "#f44336", 5),
        (5, "Medium", "#ff9800", 8),
        (3, "Medium", "#ffc107", 12),
        (1, "Sparse", "#4caf50", 20),
    ]

    level_y = 6
    for level_num, label, color, bucket_count in levels:
        # Level label
        ax.text(1, level_y, f"Level {level_num}", fontsize=10, fontweight='bold', va='center')
        ax.text(2.5, level_y, f"({label})", fontsize=9, va='center', style='italic')

        # Buckets at this level
        bucket_width = 0.4
        spacing = 9 / bucket_count
        for i in range(min(bucket_count, 15)):  # Show max 15 buckets
            x = 3.5 + i * spacing
            height = 0.15 + (bucket_count - i) * 0.01  # Vary height slightly

            rect = Rectangle((x, level_y - height/2), bucket_width, height,
                           facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)

        if bucket_count > 15:
            ax.text(3.5 + 15*spacing + 0.3, level_y, f"... ({bucket_count} total)",
                   fontsize=7, va='center', style='italic')

        level_y -= 1.2

    # Arrows showing density query
    ax.annotate('', xy=(1, 6.5), xytext=(0.3, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(1, 5.3), xytext=(0.3, 5.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax.text(0.15, 5.9, 'Query:\nbuckets\n≥5 items', ha='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # O(1) access annotation
    ax.text(6, 0.8, 'O(1) access to density distribution',
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(6, 0.4, 'Traditional hash tables require O(n) scan of all buckets',
           ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_krapivin_detail.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure_krapivin_detail.png", dpi=300, bbox_inches='tight')
    print(f"Saved Krapivin detail diagram to {OUTPUT_DIR / 'figure_krapivin_detail.pdf'}")
    plt.close()

def main():
    print("Generating architecture diagrams...")
    figure_architecture()
    figure_krapivin_detail()
    print("Done!")

if __name__ == "__main__":
    main()
