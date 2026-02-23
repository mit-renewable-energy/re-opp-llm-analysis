"""
Generate comparison figures for GABRIEL vs Claude pipeline analysis.

Usage:
    python scripts/gabriel_comparison/generate_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for config imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paper_style import PaperColors, PaperFormat, save_paper_figure, setup_paper_style

GABRIEL_DIR = Path(__file__).resolve().parent / "outputs"
COMPARISON_DIR = GABRIEL_DIR / "comparison_results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def load_data():
    """Load all comparison datasets."""
    data = {}

    # Opposition comparison stats
    stats_path = COMPARISON_DIR / "opposition_comparison_stats.csv"
    if stats_path.exists():
        data["opp_stats"] = pd.read_csv(stats_path)

    # Opposition raw comparison
    raw_path = COMPARISON_DIR / "opposition_comparison_raw.csv"
    if raw_path.exists():
        data["opp_raw"] = pd.read_csv(raw_path, dtype={"plant_code": str})

    # GABRIEL content relevance
    cr_path = GABRIEL_DIR / "content_relevance" / "results.csv"
    if cr_path.exists():
        data["content_relevance"] = pd.read_csv(cr_path, dtype={"plant_code": str})

    return data


# Human-readable variable labels
VAR_LABELS = {
    "mention_support": "Support\nMention",
    "mention_opp": "Opposition\nMention",
    "physical_opp": "Physical",
    "policy_opp": "Policy",
    "legal_opp": "Legal",
    "opinion_opp": "Opinion",
    "environmental_opp": "Environmental",
    "participation_opp": "Participation",
    "tribal_opp": "Tribal",
    "health_opp": "Health",
    "intergov_opp": "Intergov.",
    "property_opp": "Property",
    "compensation": "Compensation",
    "delay": "Delay",
    "co_land_use": "Co-Land Use",
}

VAR_LABELS_SHORT = {
    "mention_support": "Support",
    "mention_opp": "Opposition",
    "physical_opp": "Physical",
    "policy_opp": "Policy",
    "legal_opp": "Legal",
    "opinion_opp": "Opinion",
    "environmental_opp": "Environ.",
    "participation_opp": "Particip.",
    "tribal_opp": "Tribal",
    "health_opp": "Health",
    "intergov_opp": "Intergov.",
    "property_opp": "Property",
    "compensation": "Compens.",
    "delay": "Delay",
    "co_land_use": "Co-Land",
}


def fig1_kappa_accuracy_bars(stats_df):
    """
    Figure 1: Cohen's Kappa and Accuracy by opposition variable.
    Grouped horizontal bar chart.
    """
    fig, ax = plt.subplots(figsize=(PaperFormat.FIG_WIDTH_DOUBLE, PaperFormat.FIG_HEIGHT_TALL))

    # Sort by kappa descending
    df = stats_df.sort_values("kappa", ascending=True).copy()
    labels = [VAR_LABELS_SHORT.get(v, v) for v in df["variable"]]
    y = np.arange(len(df))
    bar_height = 0.35

    bars_kappa = ax.barh(
        y + bar_height / 2, df["kappa"], bar_height,
        label="Cohen's Kappa", color=PaperColors.EASTERN_BLUE, edgecolor="white", linewidth=0.3
    )
    bars_acc = ax.barh(
        y - bar_height / 2, df["accuracy"], bar_height,
        label="Accuracy", color=PaperColors.CORNFLOWER, edgecolor="white", linewidth=0.3
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.4, color=PaperColors.SILVER, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axvline(x=0.6, color=PaperColors.SILVER, linestyle="--", linewidth=0.5, alpha=0.7)

    # Kappa interpretation zones
    ax.text(0.20, len(df) - 0.5, "Slight", fontsize=6, color=PaperColors.SILVER, ha="center")
    ax.text(0.50, len(df) - 0.5, "Moderate", fontsize=6, color=PaperColors.SILVER, ha="center")
    ax.text(0.70, len(df) - 0.5, "Substantial", fontsize=6, color=PaperColors.SILVER, ha="center")

    ax.legend(loc="lower right", frameon=True, edgecolor=PaperColors.SILVER)
    ax.set_title("Agreement Between GABRIEL (GPT-5-nano) and Claude (Opus)")

    plt.tight_layout()
    save_paper_figure("fig1_kappa_accuracy.png", str(FIGURES_DIR), fig)
    plt.close(fig)


def fig2_prevalence_comparison(stats_df):
    """
    Figure 2: Prevalence comparison — original vs GABRIEL predictions.
    Scatter plot with diagonal reference line.
    """
    fig, ax = plt.subplots(figsize=(PaperFormat.FIG_WIDTH_SINGLE + 1.0, PaperFormat.FIG_WIDTH_SINGLE + 1.0))

    df = stats_df.copy()

    ax.plot([0, 0.55], [0, 0.55], color=PaperColors.SILVER, linestyle="--", linewidth=0.8, zorder=1)

    ax.scatter(
        df["original_prevalence"], df["gabriel_prevalence"],
        color=PaperColors.GREEN_VOGUE, s=40, zorder=2, edgecolor="white", linewidth=0.5
    )

    # Label each point
    for _, row in df.iterrows():
        label = VAR_LABELS_SHORT.get(row["variable"], row["variable"])
        offset_x, offset_y = 0.005, 0.005
        # Adjust specific overlapping labels
        if row["variable"] == "opinion_opp":
            offset_y = -0.012
        elif row["variable"] == "physical_opp":
            offset_y = 0.008
        elif row["variable"] == "tribal_opp":
            offset_x = -0.04
            offset_y = -0.005
        ax.annotate(
            label,
            (row["original_prevalence"] + offset_x, row["gabriel_prevalence"] + offset_y),
            fontsize=6, color=PaperColors.GREEN_VOGUE
        )

    ax.set_xlabel("Original (Claude Opus) Prevalence")
    ax.set_ylabel("GABRIEL (GPT-5-nano) Prevalence")
    ax.set_title("Label Prevalence Comparison")
    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(-0.01, 0.55)
    ax.set_aspect("equal")

    plt.tight_layout()
    save_paper_figure("fig2_prevalence_scatter.png", str(FIGURES_DIR), fig)
    plt.close(fig)


def fig3_precision_recall(stats_df):
    """
    Figure 3: Precision vs Recall for each opposition variable.
    """
    fig, ax = plt.subplots(figsize=(PaperFormat.FIG_WIDTH_SINGLE + 1.0, PaperFormat.FIG_WIDTH_SINGLE + 1.0))

    df = stats_df.copy()

    ax.scatter(
        df["recall"], df["precision"],
        c=df["kappa"], cmap="YlGnBu", s=50, edgecolor="white", linewidth=0.5, zorder=2,
        vmin=0.1, vmax=0.55
    )

    for _, row in df.iterrows():
        label = VAR_LABELS_SHORT.get(row["variable"], row["variable"])
        offset_x, offset_y = 0.01, 0.01
        if row["variable"] == "mention_opp":
            offset_y = -0.03
        elif row["variable"] == "environmental_opp":
            offset_x = -0.07
            offset_y = 0.02
        elif row["variable"] == "health_opp":
            offset_y = -0.03
        ax.annotate(
            label,
            (row["recall"] + offset_x, row["precision"] + offset_y),
            fontsize=6, color=PaperColors.GREEN_VOGUE
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall by Variable")
    ax.set_xlim(0.15, 0.72)
    ax.set_ylim(0.05, 0.85)

    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cbar.set_label("Cohen's Kappa", fontsize=PaperFormat.FONT_SIZE_SMALL)

    plt.tight_layout()
    save_paper_figure("fig3_precision_recall.png", str(FIGURES_DIR), fig)
    plt.close(fig)


def fig4_confusion_heatmap(raw_df):
    """
    Figure 4: Heatmap showing agreement patterns across all 15 variables.
    Rows = variables, columns = (TP, FP, FN, TN) as proportions.
    """
    opp_vars = list(VAR_LABELS.keys())
    results = []

    for var in opp_vars:
        g_col = f"{var}_gabriel" if f"{var}_gabriel" in raw_df.columns else var
        o_col = f"{var}_original" if f"{var}_original" in raw_df.columns else var
        if g_col not in raw_df.columns or o_col not in raw_df.columns:
            continue

        g = raw_df[g_col].fillna(0).astype(int)
        o = raw_df[o_col].fillna(0).astype(int)
        n = len(g)

        tp = ((g == 1) & (o == 1)).sum() / n
        fp = ((g == 1) & (o == 0)).sum() / n
        fn = ((g == 0) & (o == 1)).sum() / n
        tn = ((g == 0) & (o == 0)).sum() / n

        results.append({
            "variable": var,
            "True Positive": tp,
            "False Positive": fp,
            "False Negative": fn,
            "True Negative": tn,
        })

    df = pd.DataFrame(results).set_index("variable")
    # Reorder to match paper variable ordering
    df = df.loc[[v for v in opp_vars if v in df.index]]
    labels = [VAR_LABELS_SHORT.get(v, v) for v in df.index]

    fig, ax = plt.subplots(figsize=(PaperFormat.FIG_WIDTH_SINGLE + 1.5, PaperFormat.FIG_HEIGHT_TALL))

    # Stacked horizontal bar chart
    left = np.zeros(len(df))
    colors = [PaperColors.EASTERN_BLUE, PaperColors.FLUSH_ORANGE, PaperColors.SELECTIVE_YELLOW, PaperColors.GALLERY]
    category_labels = ["True Positive", "False Positive", "False Negative", "True Negative"]

    for i, (cat, color) in enumerate(zip(category_labels, colors)):
        widths = df[cat].values
        ax.barh(np.arange(len(df)), widths, left=left, height=0.7,
                color=color, edgecolor="white", linewidth=0.3, label=cat)
        left += widths

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Proportion of Projects")
    ax.set_title("Classification Agreement Breakdown")
    ax.legend(loc="lower right", fontsize=7, frameon=True, edgecolor=PaperColors.SILVER)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()

    plt.tight_layout()
    save_paper_figure("fig4_agreement_breakdown.png", str(FIGURES_DIR), fig)
    plt.close(fig)


def fig5_content_relevance_distribution(cr_df):
    """
    Figure 5: Distribution of GABRIEL content relevance scores.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PaperFormat.FIG_WIDTH_DOUBLE, PaperFormat.FIG_HEIGHT_STANDARD - 0.5))

    # Left: 0-100 histogram
    ax1.hist(
        cr_df["content_relevance"], bins=20, color=PaperColors.EASTERN_BLUE,
        edgecolor="white", linewidth=0.5, alpha=0.9
    )
    ax1.set_xlabel("Content Relevance Score (0-100)")
    ax1.set_ylabel("Number of Projects")
    ax1.set_title("GABRIEL Score Distribution")
    ax1.axvline(cr_df["content_relevance"].median(), color=PaperColors.THUNDERBIRD,
                linestyle="--", linewidth=0.8, label=f"Median: {cr_df['content_relevance'].median():.0f}")
    ax1.legend(fontsize=7)

    # Right: mapped 1-5 bar chart
    counts = cr_df["content_relevance_1to5"].value_counts().sort_index()
    bars = ax2.bar(
        counts.index, counts.values,
        color=[PaperColors.GALLERY, PaperColors.CORNFLOWER, PaperColors.EASTERN_BLUE,
               PaperColors.BLUMINE, PaperColors.GREEN_VOGUE],
        edgecolor="white", linewidth=0.5
    )
    ax2.set_xlabel("Content Relevance (1-5 Scale)")
    ax2.set_ylabel("Number of Projects")
    ax2.set_title("Mapped to 1-5 Scale")

    # Add count labels
    for bar, count in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(count), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    save_paper_figure("fig5_content_relevance_dist.png", str(FIGURES_DIR), fig)
    plt.close(fig)


def main():
    setup_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    if "opp_stats" in data:
        print("Generating Figure 1: Kappa/Accuracy bars...")
        fig1_kappa_accuracy_bars(data["opp_stats"])

        print("Generating Figure 2: Prevalence scatter...")
        fig2_prevalence_comparison(data["opp_stats"])

        print("Generating Figure 3: Precision-Recall scatter...")
        fig3_precision_recall(data["opp_stats"])

    if "opp_raw" in data:
        print("Generating Figure 4: Agreement breakdown...")
        fig4_confusion_heatmap(data["opp_raw"])

    if "content_relevance" in data:
        print("Generating Figure 5: Content relevance distribution...")
        fig5_content_relevance_distribution(data["content_relevance"])

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
