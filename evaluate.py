from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LABEL_NAMES = ["negative", "positive", "neutral"]

# ── 1. Overall Metrics ────────────────────────────────────────────────────────
def evaluate_overall(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'='*40}")
    print(f"{model_name} — Overall Results")
    print(f"{'='*40}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro F1:  {f1:.3f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}")
    return {"accuracy": acc, "macro_f1": f1}


# ── 2. Per Bucket Metrics ─────────────────────────────────────────────────────
def evaluate_by_bucket(y_true, y_pred, buckets, model_name="Model"):
    results = {}
    print(f"\n{model_name} — Results by Length Bucket")
    print(f"{'='*40}")
    for bucket in ["short", "medium", "long"]:
        mask = buckets == bucket
        if mask.sum() == 0:
            print(f"{bucket}: no samples")
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
        print(f"{bucket:8s} → Accuracy: {acc:.3f} | Macro F1: {f1:.3f} | Samples: {mask.sum()}")
        results[bucket] = {"accuracy": acc, "macro_f1": f1}
    return results


# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/confusion_matrix_{model_name.lower()}.png", dpi=150)
    plt.show()


# ── 4. Comparison Table ───────────────────────────────────────────────────────
def build_results_table(all_results):
    """
    all_results = {
        "SVM":     {"overall": {...}, "short": {...}, "medium": {...}, "long": {...}},
        "LSTM":    {"overall": {...}, "short": {...}, "medium": {...}, "long": {...}},
        "FinBERT": {"overall": {...}, "short": {...}, "medium": {...}, "long": {...}},
    }
    """
    rows = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for bucket, metrics in results.items():
            row[f"{bucket}_accuracy"] = round(metrics["accuracy"], 3)
            row[f"{bucket}_f1"] = round(metrics["macro_f1"], 3)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ── 5. Per Bucket Bar Chart ───────────────────────────────────────────────────
def plot_bucket_comparison(all_results, metric="macro_f1"):
    models = list(all_results.keys())
    buckets = ["short", "medium", "long"]
    x = np.arange(len(buckets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        scores = [all_results[model][b][metric] for b in buckets]
        ax.bar(x + i * width, scores, width, label=model)

    ax.set_xlabel("Length Bucket")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Length Bucket")
    ax.set_xticks(x + width)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"results/figures/bucket_comparison_{metric}.png", dpi=150)
    plt.show()