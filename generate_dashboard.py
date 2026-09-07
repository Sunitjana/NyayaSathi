"""Generate performance and data distribution dashboard charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve() / "legal-rag"))

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "legal-rag", ".env"))

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

DARK_BG = "#0f0f1a"
CARD_BG = "#1a1a2e"
ACCENT = "#6366f1"
ACCENT2 = "#8b5cf6"
ACCENT3 = "#a78bfa"
TEXT = "#e4e4f0"
MUTED = "#6b7280"
GREEN = "#10b981"
AMBER = "#f59e0b"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#2a2a3e",
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def get_db_metrics():
    import psycopg2
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    cur.execute("SELECT collection, COUNT(*) FROM legal_chunks GROUP BY collection ORDER BY COUNT(*) DESC")
    collections = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("SELECT source, COUNT(*) FROM legal_chunks GROUP BY source ORDER BY COUNT(*) DESC LIMIT 15")
    sources = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) FROM legal_chunks")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT source) FROM legal_chunks")
    unique_sources = cur.fetchone()[0]

    conn.close()
    return collections, sources, total, unique_sources


def get_latency_metrics():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import psycopg2

    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    embed_times = []
    for _ in range(5):
        start = time.time()
        model.encode(["What is the punishment for theft under IPC?"] * 50, batch_size=128, show_progress_bar=False)
        embed_times.append((time.time() - start) * 1000 / 50)

    rerank_times = []
    pairs = [("What is punishment for theft?", "Section 379 IPC provides punishment")] * 20
    for _ in range(3):
        start = time.time()
        reranker.predict(pairs)
        rerank_times.append((time.time() - start) * 1000 / 20)

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    q = model.encode("What is the punishment for theft under IPC?").tolist()
    query_times = []
    for _ in range(3):
        start = time.time()
        cur.execute(
            "SELECT text, 1 - (embedding <=> %s::vector) AS sim FROM legal_chunks ORDER BY embedding <=> %s::vector LIMIT 10",
            (str(q), str(q)),
        )
        cur.fetchall()
        query_times.append((time.time() - start) * 1000)
    conn.close()

    return embed_times, rerank_times, query_times


def plot_dashboard():
    print("Collecting database metrics...")
    collections, sources, total, unique_sources = get_db_metrics()

    print("Running latency benchmarks...")
    embed_times, rerank_times, query_times = get_latency_metrics()

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Legal AI RAG Pipeline — Performance Dashboard",
                 fontsize=22, fontweight="bold", color=TEXT, y=0.98)
    fig.text(0.5, 0.955, f"Total: {total:,} chunks | {unique_sources} sources | 384-dim embeddings | 3 ML models",
             ha="center", fontsize=11, color=MUTED)

    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # 1. Collection Distribution (pie)
    ax1 = fig.add_subplot(gs[0, 0:2])
    labels = [k.replace("legal_", "").replace("_", " ").title() for k in collections.keys()]
    sizes = list(collections.values())
    colors = [ACCENT, ACCENT2, ACCENT3, "#c4b5fd"]
    explode = [0.05] * len(sizes)
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        explode=explode, startangle=90, textprops={"color": TEXT, "fontsize": 9}
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("#1a1a2e")
        t.set_fontweight("bold")
    ax1.set_title("Data Distribution by Collection")

    # 2. Top Sources (horizontal bar)
    ax2 = fig.add_subplot(gs[0, 2:4])
    src_names = [k.replace(".pdf", "").replace("_", " ")[:25] for k in reversed(list(sources.keys())[:10])]
    src_counts = list(reversed(list(sources.values())[:10]))
    bars = ax2.barh(src_names, src_counts, color=ACCENT, alpha=0.85, height=0.6)
    for bar, val in zip(bars, src_counts):
        ax2.text(bar.get_width() + max(src_counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=8, color=MUTED)
    ax2.set_title("Top 10 Data Sources")
    ax2.set_xlabel("Chunks")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 3. Embedding Latency
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(len(embed_times)), embed_times, color=GREEN, alpha=0.85, width=0.5)
    avg_embed = np.mean(embed_times)
    ax3.axhline(y=avg_embed, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.text(len(embed_times) - 0.5, avg_embed + 0.3, f"avg: {avg_embed:.1f}ms", color=AMBER, fontsize=9)
    ax3.set_title("Embedding Latency (per chunk)")
    ax3.set_ylabel("ms")
    ax3.set_xlabel("Run")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # 4. Reranker Latency
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(len(rerank_times)), rerank_times, color=ACCENT2, alpha=0.85, width=0.5)
    avg_rerank = np.mean(rerank_times)
    ax4.axhline(y=avg_rerank, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.text(len(rerank_times) - 0.5, avg_rerank + 0.5, f"avg: {avg_rerank:.1f}ms", color=AMBER, fontsize=9)
    ax4.set_title("Reranker Latency (per pair)")
    ax4.set_ylabel("ms")
    ax4.set_xlabel("Run")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # 5. Vector Search Latency
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(range(len(query_times)), query_times, color=ACCENT3, alpha=0.85, width=0.5)
    avg_query = np.mean(query_times)
    ax5.axhline(y=avg_query, color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7)
    ax5.text(len(query_times) - 0.5, avg_query + 50, f"avg: {avg_query:.0f}ms", color=AMBER, fontsize=9)
    ax5.set_title(f"pgvector Search ({total:,} vectors)")
    ax5.set_ylabel("ms")
    ax5.set_xlabel("Run")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # 6. Summary Stats
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis("off")
    stats_text = (
        f"Total Chunks:  {total:,}\n"
        f"Unique Sources:  {unique_sources}\n"
        f"Embedding Dim:  384\n"
        f"ML Models:  3\n\n"
        f"Embed Speed:  {1000/avg_embed:.0f} chunks/sec\n"
        f"Rerank Speed:  {1000/avg_rerank:.0f} pairs/sec\n"
        f"Search Latency:  {avg_query:.0f}ms\n\n"
        f"Collections:  {len(collections)}\n"
        f"PDF Documents:  28\n"
        f"Court Judgments:  146K+"
    )
    ax6.text(0.1, 0.95, "Summary", fontsize=14, fontweight="bold", color=ACCENT,
             transform=ax6.transAxes, va="top")
    ax6.text(0.1, 0.82, stats_text, fontsize=10, color=TEXT, family="monospace",
             transform=ax6.transAxes, va="top", linespacing=1.6)

    # 7. Pipeline Architecture
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 2)

    stages = [
        ("User Query", 0.5, GREEN),
        ("Embedding\n(MiniLM-L6)", 2.0, ACCENT),
        ("pgvector\nSearch", 3.5, ACCENT2),
        ("Cross-Encoder\nReranking", 5.2, ACCENT3),
        ("Gemini 2.5\nFlash", 7.0, "#c4b5fd"),
        ("Streamed\nAnswer", 8.7, GREEN),
    ]

    for label, x, color in stages:
        box = plt.Rectangle((x - 0.55, 0.5), 1.1, 1.0, facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=2, zorder=2, transform=ax7.transData)
        ax7.add_patch(box)
        ax7.text(x, 1.0, label, ha="center", va="center", fontsize=10,
                 fontweight="bold", color=TEXT, zorder=3)

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 0.55
        x2 = stages[i + 1][1] - 0.55
        ax7.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))

    ax7.set_title("RAG Pipeline Architecture", fontsize=14, fontweight="bold", pad=15)

    metrics_labels = [
        (2.0, f"{avg_embed:.1f}ms"),
        (3.5, f"{avg_query:.0f}ms"),
        (5.2, f"{avg_rerank:.1f}ms"),
        (7.0, "streaming"),
    ]
    for x, label in metrics_labels:
        ax7.text(x, 0.25, label, ha="center", fontsize=8, color=AMBER, style="italic")

    out_path = os.path.join(CHARTS_DIR, "00_dashboard.png")
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")

    # Individual charts
    # Data distribution bar chart
    fig2, ax = plt.subplots(figsize=(10, 5))
    coll_labels = [k.replace("legal_", "").replace("_", " ").title() for k in collections.keys()]
    coll_vals = list(collections.values())
    bars = ax.bar(coll_labels, coll_vals, color=[ACCENT, ACCENT2, ACCENT3, "#c4b5fd"], alpha=0.85)
    for bar, val in zip(bars, coll_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                f"{val:,}", ha="center", fontsize=10, color=TEXT)
    ax.set_title("Dataset Distribution by Collection")
    ax.set_ylabel("Number of Chunks")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig2.savefig(os.path.join(CHARTS_DIR, "01_dataset_distribution.png"), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: 01_dataset_distribution.png")

    # Latency comparison
    fig3, ax = plt.subplots(figsize=(8, 5))
    labels = ["Embedding\n(per chunk)", "Reranking\n(per pair)", "Vector Search\n(top 10)"]
    vals = [avg_embed, avg_rerank, avg_query]
    colors = [GREEN, ACCENT2, ACCENT3]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{val:.1f}ms", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_title("Pipeline Latency Breakdown")
    ax.set_ylabel("Latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig3.savefig(os.path.join(CHARTS_DIR, "02_latency_breakdown.png"), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: 02_latency_breakdown.png")

    # Top sources
    fig4, ax = plt.subplots(figsize=(10, 6))
    src_names_plot = [k.replace(".pdf", "").replace("_", " ")[:30] for k in list(sources.keys())[:10]]
    src_vals_plot = list(sources.values())[:10]
    colors_grad = [ACCENT] * len(src_names_plot)
    bars = ax.barh(list(reversed(src_names_plot)), list(reversed(src_vals_plot)),
                   color=colors_grad, alpha=0.85, height=0.6)
    for bar, val in zip(bars, reversed(src_vals_plot)):
        ax.text(bar.get_width() + max(src_vals_plot) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color=MUTED)
    ax.set_title("Top 10 Data Sources by Chunk Count")
    ax.set_xlabel("Number of Chunks")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig4.savefig(os.path.join(CHARTS_DIR, "03_top_sources.png"), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: 03_top_sources.png")

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    plot_dashboard()
