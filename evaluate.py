"""Evaluate the Legal RAG pipeline — retrieval metrics, classifier accuracy, and generate charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "legal-rag"))

import os
import time
import json

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
RED = "#ef4444"

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

RETRIEVAL_TEST_CASES = [
    {"query": "What is the punishment for murder under IPC?", "expected_source": "IPC.pdf", "category": "criminal"},
    {"query": "What is Section 302 of IPC?", "expected_source": "IPC.pdf", "category": "criminal"},
    {"query": "What is theft under Indian law?", "expected_source": "indian-law-dataset", "category": "criminal"},
    {"query": "Define robbery under IPC", "expected_source": "IPC.pdf", "category": "criminal"},
    {"query": "Bail provisions in India", "expected_source": "indian-law-dataset", "category": "criminal"},
    {"query": "What are fundamental rights?", "expected_source": "Constitution_of_India.pdf", "category": "civil"},
    {"query": "Right to equality Article 14", "expected_source": "Constitution_of_India.pdf", "category": "civil"},
    {"query": "Freedom of speech Article 19", "expected_source": "Constitution_of_India.pdf", "category": "civil"},
    {"query": "Right to life Article 21", "expected_source": "Constitution_of_India.pdf", "category": "civil"},
    {"query": "Directive principles of state policy", "expected_source": "Constitution_of_India.pdf", "category": "civil"},
    {"query": "Transfer of property by sale", "expected_source": "Transfer_of_Property_Act_1882.pdf", "category": "property"},
    {"query": "Lease agreement rights under property law", "expected_source": "Transfer_of_Property_Act_1882.pdf", "category": "property"},
    {"query": "Landlord tenant rights India", "expected_source": "indian-law-dataset", "category": "property"},
    {"query": "Consumer protection act complaint", "expected_source": "Consumer_Protection_Act_2019.pdf", "category": "consumer"},
    {"query": "Defective product consumer rights", "expected_source": "Consumer_Protection_Act_2019.pdf", "category": "consumer"},
    {"query": "Refund rights for defective goods", "expected_source": "indian-law-dataset", "category": "consumer"},
    {"query": "Employment termination laws India", "expected_source": "indian-law-dataset", "category": "labor"},
    {"query": "Gratuity payment eligibility", "expected_source": "indian-law-dataset", "category": "labor"},
    {"query": "Minimum wages act provisions", "expected_source": "indian-law-dataset", "category": "labor"},
    {"query": "Divorce procedure Hindu marriage act", "expected_source": "indian-law-dataset", "category": "family"},
    {"query": "Child custody rights after divorce", "expected_source": "indian-law-dataset", "category": "family"},
    {"query": "Domestic violence protection act", "expected_source": "indian-law-dataset", "category": "family"},
    {"query": "Maintenance alimony wife rights", "expected_source": "indian-law-dataset", "category": "family"},
    {"query": "Motor vehicle accident compensation", "expected_source": "Motor_Vehicles_Act_1988.pdf", "category": "civil"},
    {"query": "IT Act cyber crime punishment", "expected_source": "IT_Act_2000.pdf", "category": "criminal"},
    {"query": "Contract breach damages India", "expected_source": "Indian_Contract_Act_1872.pdf", "category": "civil"},
    {"query": "Arbitration dispute resolution India", "expected_source": "Arbitration_Act_1996.pdf", "category": "civil"},
    {"query": "NDPS Act drug offense punishment", "expected_source": "NDPS_Act_1985.pdf", "category": "criminal"},
    {"query": "Company directors liability", "expected_source": "Companies_Act_2013.pdf", "category": "civil"},
    {"query": "CPC civil suit procedure", "expected_source": "CPC_1908.pdf", "category": "civil"},
]


def evaluate_retrieval():
    from rag.retriever import _get_model, _get_conn, ALL_COLLECTIONS

    print(f"Evaluating retrieval on {len(RETRIEVAL_TEST_CASES)} test cases...")
    model = _get_model()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    per_query_results = []
    latencies = []

    conn = _get_conn()
    cur = conn.cursor()

    for tc in RETRIEVAL_TEST_CASES:
        query = tc["query"]
        expected = tc["expected_source"]

        q_emb = model.encode(query).tolist()

        start = time.time()
        cur.execute(
            """SELECT source, 1 - (embedding <=> %s::vector) AS sim
               FROM legal_chunks
               ORDER BY embedding <=> %s::vector
               LIMIT 10""",
            (str(q_emb), str(q_emb)),
        )
        rows = cur.fetchall()
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        sources_found = [r[0] for r in rows]
        scores = [r[1] for r in rows]

        found_at = -1
        for i, src in enumerate(sources_found):
            if src == expected or expected in src:
                found_at = i
                break

        if found_at == 0:
            hits_at_1 += 1
        if found_at >= 0 and found_at < 3:
            hits_at_3 += 1
        if found_at >= 0 and found_at < 5:
            hits_at_5 += 1
        if found_at >= 0:
            hits_at_10 += 1
            reciprocal_ranks.append(1.0 / (found_at + 1))
        else:
            reciprocal_ranks.append(0.0)

        per_query_results.append({
            "query": query,
            "expected": expected,
            "found_at": found_at,
            "top_source": sources_found[0] if sources_found else "none",
            "top_score": scores[0] if scores else 0,
        })

    conn.close()

    n = len(RETRIEVAL_TEST_CASES)
    metrics = {
        "precision_at_1": hits_at_1 / n,
        "precision_at_3": hits_at_3 / n,
        "precision_at_5": hits_at_5 / n,
        "recall_at_10": hits_at_10 / n,
        "mrr": np.mean(reciprocal_ranks),
        "avg_latency_ms": np.mean(latencies),
        "total_queries": n,
    }

    print(f"  Precision@1: {metrics['precision_at_1']:.1%}")
    print(f"  Precision@3: {metrics['precision_at_3']:.1%}")
    print(f"  Precision@5: {metrics['precision_at_5']:.1%}")
    print(f"  Recall@10:   {metrics['recall_at_10']:.1%}")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.0f}ms")

    return metrics, per_query_results


def evaluate_classifier():
    from assistant.legal_assistant import classify_problem

    print(f"\nEvaluating category classifier...")

    categories = ["criminal", "civil", "property", "consumer", "labor", "family"]
    true_labels = []
    pred_labels = []

    for tc in RETRIEVAL_TEST_CASES:
        true_labels.append(tc["category"])
        pred_labels.append(classify_problem(tc["query"]))

    n = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / n

    confusion = np.zeros((len(categories), len(categories)), dtype=int)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    for t, p in zip(true_labels, pred_labels):
        ti = cat_to_idx.get(t, 0)
        pi = cat_to_idx.get(p, 0)
        confusion[ti][pi] += 1

    per_class = {}
    for i, cat in enumerate(categories):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(len(categories))) - tp
        fn = sum(confusion[i][j] for j in range(len(categories))) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cat] = {"precision": precision, "recall": recall, "f1": f1, "support": int(sum(confusion[i]))}

    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    print(f"  Accuracy:        {accuracy:.1%}")
    print(f"  Macro Precision: {macro_precision:.3f}")
    print(f"  Macro Recall:    {macro_recall:.3f}")
    print(f"  Macro F1:        {macro_f1:.3f}")
    print(f"  Samples:         {n}")

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "categories": categories,
        "n_samples": n,
    }


def plot_evaluation(retrieval_metrics, classifier_metrics):
    categories = classifier_metrics["categories"]
    confusion = classifier_metrics["confusion_matrix"]
    per_class = classifier_metrics["per_class"]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Legal AI RAG — Evaluation Metrics Dashboard",
                 fontsize=22, fontweight="bold", color=TEXT, y=0.98)
    fig.text(0.5, 0.955,
             f"{retrieval_metrics['total_queries']} test queries | {classifier_metrics['n_samples']} classification samples | 6 categories",
             ha="center", fontsize=11, color=MUTED)

    gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.4,
                           left=0.07, right=0.95, top=0.92, bottom=0.06)

    # 1. Retrieval Precision@K
    ax1 = fig.add_subplot(gs[0, 0:2])
    k_vals = [1, 3, 5, 10]
    p_vals = [
        retrieval_metrics["precision_at_1"],
        retrieval_metrics["precision_at_3"],
        retrieval_metrics["precision_at_5"],
        retrieval_metrics["recall_at_10"],
    ]
    bars = ax1.bar([f"P@{k}" if k < 10 else "R@10" for k in k_vals], p_vals,
                   color=[ACCENT, ACCENT2, ACCENT3, GREEN], alpha=0.85, width=0.5)
    for bar, val in zip(bars, p_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Retrieval Precision@K & Recall@10")
    ax1.set_ylabel("Score")
    ax1.axhline(y=retrieval_metrics["mrr"], color=AMBER, linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.text(3.5, retrieval_metrics["mrr"] + 0.03, f"MRR: {retrieval_metrics['mrr']:.3f}",
             color=AMBER, fontsize=10, ha="right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 2. Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 2:4])
    short_cats = [c[:4].title() for c in categories]
    im = ax2.imshow(confusion, cmap="YlOrRd", aspect="auto", alpha=0.85)
    ax2.set_xticks(range(len(categories)))
    ax2.set_yticks(range(len(categories)))
    ax2.set_xticklabels(short_cats, fontsize=9)
    ax2.set_yticklabels(short_cats, fontsize=9)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"Category Classifier Confusion Matrix (Acc: {classifier_metrics['accuracy']:.1%})")
    for i in range(len(categories)):
        for j in range(len(categories)):
            val = confusion[i][j]
            color = "#1a1a2e" if val > 2 else TEXT
            ax2.text(j, i, str(val), ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    # 3. Per-class Precision, Recall, F1
    ax3 = fig.add_subplot(gs[1, 0:3])
    x = np.arange(len(categories))
    width = 0.25
    p_vals_cls = [per_class[c]["precision"] for c in categories]
    r_vals_cls = [per_class[c]["recall"] for c in categories]
    f1_vals_cls = [per_class[c]["f1"] for c in categories]
    ax3.bar(x - width, p_vals_cls, width, label="Precision", color=ACCENT, alpha=0.85)
    ax3.bar(x, r_vals_cls, width, label="Recall", color=GREEN, alpha=0.85)
    ax3.bar(x + width, f1_vals_cls, width, label="F1 Score", color=AMBER, alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.title() for c in categories], fontsize=10)
    ax3.set_ylim(0, 1.2)
    ax3.set_title("Per-Class Precision / Recall / F1")
    ax3.set_ylabel("Score")
    ax3.legend(loc="upper right", framealpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # 4. Summary stats card
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.axis("off")
    summary = (
        f"RETRIEVAL\n"
        f"─────────────────\n"
        f"Precision@1:   {retrieval_metrics['precision_at_1']:.1%}\n"
        f"Precision@3:   {retrieval_metrics['precision_at_3']:.1%}\n"
        f"Precision@5:   {retrieval_metrics['precision_at_5']:.1%}\n"
        f"Recall@10:     {retrieval_metrics['recall_at_10']:.1%}\n"
        f"MRR:           {retrieval_metrics['mrr']:.3f}\n"
        f"Avg Latency:   {retrieval_metrics['avg_latency_ms']:.0f}ms\n\n"
        f"CLASSIFIER\n"
        f"─────────────────\n"
        f"Accuracy:      {classifier_metrics['accuracy']:.1%}\n"
        f"Macro Prec:    {classifier_metrics['macro_precision']:.3f}\n"
        f"Macro Recall:  {classifier_metrics['macro_recall']:.3f}\n"
        f"Macro F1:      {classifier_metrics['macro_f1']:.3f}\n"
        f"Categories:    {len(categories)}\n"
        f"Samples:       {classifier_metrics['n_samples']}"
    )
    ax4.text(0.05, 0.95, summary, fontsize=10, color=TEXT, family="monospace",
             transform=ax4.transAxes, va="top", linespacing=1.5)

    # 5. Macro F1 gauge
    ax5 = fig.add_subplot(gs[2, 0])
    f1 = classifier_metrics["macro_f1"]
    color = GREEN if f1 >= 0.8 else AMBER if f1 >= 0.6 else RED
    ax5.barh(["Macro F1"], [f1], color=color, alpha=0.85, height=0.4)
    ax5.barh(["Macro F1"], [1.0], color="#2a2a3e", alpha=0.3, height=0.4)
    ax5.text(f1 + 0.02, 0, f"{f1:.3f}", va="center", fontsize=14, fontweight="bold", color=TEXT)
    ax5.set_xlim(0, 1.1)
    ax5.set_title("Macro F1 Score")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # 6. MRR gauge
    ax6 = fig.add_subplot(gs[2, 1])
    mrr = retrieval_metrics["mrr"]
    color = GREEN if mrr >= 0.7 else AMBER if mrr >= 0.5 else RED
    ax6.barh(["MRR"], [mrr], color=color, alpha=0.85, height=0.4)
    ax6.barh(["MRR"], [1.0], color="#2a2a3e", alpha=0.3, height=0.4)
    ax6.text(mrr + 0.02, 0, f"{mrr:.3f}", va="center", fontsize=14, fontweight="bold", color=TEXT)
    ax6.set_xlim(0, 1.1)
    ax6.set_title("Mean Reciprocal Rank")
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # 7. Recall@10 gauge
    ax7 = fig.add_subplot(gs[2, 2])
    r10 = retrieval_metrics["recall_at_10"]
    color = GREEN if r10 >= 0.8 else AMBER if r10 >= 0.6 else RED
    ax7.barh(["Recall@10"], [r10], color=color, alpha=0.85, height=0.4)
    ax7.barh(["Recall@10"], [1.0], color="#2a2a3e", alpha=0.3, height=0.4)
    ax7.text(r10 + 0.02, 0, f"{r10:.1%}", va="center", fontsize=14, fontweight="bold", color=TEXT)
    ax7.set_xlim(0, 1.1)
    ax7.set_title("Retrieval Recall@10")
    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)

    # 8. Accuracy gauge
    ax8 = fig.add_subplot(gs[2, 3])
    acc = classifier_metrics["accuracy"]
    color = GREEN if acc >= 0.8 else AMBER if acc >= 0.6 else RED
    ax8.barh(["Accuracy"], [acc], color=color, alpha=0.85, height=0.4)
    ax8.barh(["Accuracy"], [1.0], color="#2a2a3e", alpha=0.3, height=0.4)
    ax8.text(acc + 0.02, 0, f"{acc:.1%}", va="center", fontsize=14, fontweight="bold", color=TEXT)
    ax8.set_xlim(0, 1.1)
    ax8.set_title("Classifier Accuracy")
    ax8.spines["top"].set_visible(False)
    ax8.spines["right"].set_visible(False)

    out = os.path.join(CHARTS_DIR, "04_evaluation_dashboard.png")
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"\nSaved: {out}")

    # Standalone confusion matrix
    fig2, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="YlOrRd", aspect="auto", alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels([c.title() for c in categories], fontsize=11)
    ax.set_yticklabels([c.title() for c in categories], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Category Classification — Confusion Matrix\nAccuracy: {acc:.1%} | Macro F1: {classifier_metrics['macro_f1']:.3f}", fontsize=14)
    for i in range(len(categories)):
        for j in range(len(categories)):
            val = confusion[i][j]
            color = "#1a1a2e" if val > 2 else TEXT
            ax.text(j, i, str(val), ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    fig2.colorbar(im, ax=ax, shrink=0.8)
    fig2.savefig(os.path.join(CHARTS_DIR, "05_confusion_matrix.png"), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: 05_confusion_matrix.png")

    # Standalone precision/recall/f1
    fig3, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25
    ax.bar(x - width, p_vals_cls, width, label="Precision", color=ACCENT, alpha=0.85)
    ax.bar(x, r_vals_cls, width, label="Recall", color=GREEN, alpha=0.85)
    ax.bar(x + width, f1_vals_cls, width, label="F1 Score", color=AMBER, alpha=0.85)
    for i, (p, r, f) in enumerate(zip(p_vals_cls, r_vals_cls, f1_vals_cls)):
        ax.text(i - width, p + 0.03, f"{p:.2f}", ha="center", fontsize=8, color=TEXT)
        ax.text(i, r + 0.03, f"{r:.2f}", ha="center", fontsize=8, color=TEXT)
        ax.text(i + width, f + 0.03, f"{f:.2f}", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories], fontsize=11)
    ax.set_ylim(0, 1.25)
    ax.set_title(f"Per-Class Precision / Recall / F1\nMacro P: {classifier_metrics['macro_precision']:.3f} | Macro R: {classifier_metrics['macro_recall']:.3f} | Macro F1: {classifier_metrics['macro_f1']:.3f}")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", framealpha=0.3, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig3.savefig(os.path.join(CHARTS_DIR, "06_precision_recall_f1.png"), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"Saved: 06_precision_recall_f1.png")


if __name__ == "__main__":
    r_metrics, r_results = evaluate_retrieval()
    c_metrics = evaluate_classifier()
    plot_evaluation(r_metrics, c_metrics)
    print(f"\nAll evaluation charts saved to {CHARTS_DIR}/")
