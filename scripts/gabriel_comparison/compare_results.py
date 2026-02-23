"""
Compare GABRIEL pipeline results against original Claude-based pipeline.

Loads original per-plant JSON results and GABRIEL CSV outputs, then computes
agreement metrics for each stage.

Usage:
    python scripts/gabriel_comparison/compare_results.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# pCloud has the full content/search/scores JSONs; local repo only has a subset
PCLOUD_RESULTS_DIR = Path.home() / "pCloud Drive" / "MIT" / "MIT Work" / "Renewable Energy UROP" / "dispute-characterization" / "data" / "processed" / "results"
RESULTS_DIR = PCLOUD_RESULTS_DIR if PCLOUD_RESULTS_DIR.exists() else PROJECT_ROOT / "data" / "processed" / "results"
GABRIEL_DIR = Path(__file__).resolve().parent / "outputs"
COMPARISON_DIR = GABRIEL_DIR / "comparison_results"

OPPOSITION_VARS = [
    "mention_support", "mention_opp", "physical_opp", "policy_opp",
    "legal_opp", "opinion_opp", "environmental_opp", "participation_opp",
    "tribal_opp", "health_opp", "intergov_opp", "property_opp",
    "compensation", "delay", "co_land_use",
]


def load_original_article_relevance(plant_codes: list[str]) -> pd.DataFrame:
    """Load original article-level relevance scores from per-plant JSONs."""
    rows = []
    for pc in plant_codes:
        path = RESULTS_DIR / "article_relevance" / f"{pc}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if not data:
            continue
        scores = data.get("scores_and_justifications", [])
        for item in scores:
            rows.append({
                "plant_code": pc,
                "article_letter": item["article_letter"],
                "original_grade": item["grade"],
            })
    return pd.DataFrame(rows)


def load_original_content_relevance(plant_codes: list[str]) -> pd.DataFrame:
    """Load original content-level relevance scores from per-plant JSONs."""
    rows = []
    for i, pc in enumerate(plant_codes):
        if i % 500 == 0:
            print(f"  Loading content relevance: {i}/{len(plant_codes)}...", flush=True)
        path = RESULTS_DIR / "content_relevance" / f"{pc}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not data:
            continue
        scores = data.get("score_and_justification", [])
        if scores:
            rows.append({
                "plant_code": pc,
                "original_score": scores[0]["score"],
            })
    print(f"  Loaded {len(rows)} content relevance scores", flush=True)
    return pd.DataFrame(rows)


def load_original_opposition_scores(plant_codes: list[str]) -> pd.DataFrame:
    """Load original opposition binary variables from per-plant score JSONs."""
    rows = []
    for i, pc in enumerate(plant_codes):
        if i % 500 == 0:
            print(f"  Loading opposition scores: {i}/{len(plant_codes)}...", flush=True)
        path = RESULTS_DIR / "scores" / f"{pc}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        # Handle nested structure: data may have "all_scores_and_sources" or "scores"
        scores_list = data.get("all_scores_and_sources", data.get("scores", []))
        if not scores_list:
            continue

        entry = scores_list[0]
        row = {"plant_code": pc}
        for var in OPPOSITION_VARS:
            val = entry.get(var)
            # mention_support and mention_opp are lists of dicts with "score" key
            if isinstance(val, list) and val:
                row[var] = val[0].get("score", 0)
            elif isinstance(val, (int, float)):
                row[var] = int(val)
            else:
                row[var] = 0
        row["original_narrative"] = entry.get("narrative", "")
        rows.append(row)
    print(f"  Loaded {len(rows)} opposition scores", flush=True)
    return pd.DataFrame(rows)


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_article_relevance() -> str:
    """Compare article-level relevance scores."""
    gabriel_path = GABRIEL_DIR / "article_relevance" / "results.csv"
    if not gabriel_path.exists():
        return "Article relevance: GABRIEL results not found. Skipping.\n"

    gabriel_df = pd.read_csv(gabriel_path, dtype={"plant_code": str})
    plant_codes = gabriel_df["plant_code"].unique().tolist()
    original_df = load_original_article_relevance(plant_codes)

    if original_df.empty:
        return "Article relevance: No original results found for comparison.\n"

    # Merge on plant_code + article_letter
    merged = gabriel_df.merge(
        original_df,
        on=["plant_code", "article_letter"],
        how="inner",
    )

    if merged.empty:
        return "Article relevance: No overlapping articles found for comparison.\n"

    lines = ["=" * 60, "ARTICLE RELEVANCE COMPARISON", "=" * 60]
    lines.append(f"Matched articles: {len(merged)}")

    # Correlation between GABRIEL 0-100 and original 1-5
    r_raw, p_raw = stats.pearsonr(merged["relevance"], merged["original_grade"])
    lines.append(f"Pearson r (GABRIEL 0-100 vs Original 1-5): {r_raw:.3f} (p={p_raw:.4f})")

    # Correlation between derived 1-5 and original 1-5
    r_derived, p_derived = stats.pearsonr(merged["relevance_1to5"], merged["original_grade"])
    lines.append(f"Pearson r (GABRIEL 1-5 vs Original 1-5):   {r_derived:.3f} (p={p_derived:.4f})")

    # Exact match rate
    exact = (merged["relevance_1to5"] == merged["original_grade"]).mean()
    within1 = (abs(merged["relevance_1to5"] - merged["original_grade"]) <= 1).mean()
    lines.append(f"Exact match (1-5 scale): {exact:.1%}")
    lines.append(f"Within 1 point:          {within1:.1%}")

    # Distribution comparison
    lines.append("\nScore distribution:")
    lines.append("  Original 1-5: " + str(merged["original_grade"].value_counts().sort_index().to_dict()))
    lines.append("  GABRIEL  1-5: " + str(merged["relevance_1to5"].value_counts().sort_index().to_dict()))

    # Save merged data
    merged.to_csv(COMPARISON_DIR / "article_relevance_comparison.csv", index=False)
    lines.append(f"\nSaved to: {COMPARISON_DIR / 'article_relevance_comparison.csv'}")

    return "\n".join(lines) + "\n"


def compare_content_relevance() -> str:
    """Compare content-level relevance scores."""
    gabriel_path = GABRIEL_DIR / "content_relevance" / "results.csv"
    if not gabriel_path.exists():
        return "Content relevance: GABRIEL results not found. Skipping.\n"

    gabriel_df = pd.read_csv(gabriel_path, dtype={"plant_code": str})
    plant_codes = gabriel_df["plant_code"].unique().tolist()
    original_df = load_original_content_relevance(plant_codes)

    if original_df.empty:
        return "Content relevance: No original results found for comparison.\n"

    merged = gabriel_df.merge(original_df, on="plant_code", how="inner")

    if merged.empty:
        return "Content relevance: No overlapping projects found for comparison.\n"

    lines = ["\n" + "=" * 60, "CONTENT RELEVANCE COMPARISON", "=" * 60]
    lines.append(f"Matched projects: {len(merged)}")

    r_raw, p_raw = stats.pearsonr(merged["content_relevance"], merged["original_score"])
    lines.append(f"Pearson r (GABRIEL 0-100 vs Original 1-5): {r_raw:.3f} (p={p_raw:.4f})")

    r_derived, p_derived = stats.pearsonr(merged["content_relevance_1to5"], merged["original_score"])
    lines.append(f"Pearson r (GABRIEL 1-5 vs Original 1-5):   {r_derived:.3f} (p={p_derived:.4f})")

    exact = (merged["content_relevance_1to5"] == merged["original_score"]).mean()
    within1 = (abs(merged["content_relevance_1to5"] - merged["original_score"]) <= 1).mean()
    lines.append(f"Exact match (1-5 scale): {exact:.1%}")
    lines.append(f"Within 1 point:          {within1:.1%}")

    merged.to_csv(COMPARISON_DIR / "content_relevance_comparison.csv", index=False)
    lines.append(f"\nSaved to: {COMPARISON_DIR / 'content_relevance_comparison.csv'}")

    return "\n".join(lines) + "\n"


def compare_opposition() -> str:
    """Compare binary opposition variables using Cohen's kappa and accuracy."""
    gabriel_path = GABRIEL_DIR / "opposition_classify" / "results.csv"
    if not gabriel_path.exists():
        return "Opposition classify: GABRIEL results not found. Skipping.\n"

    gabriel_df = pd.read_csv(gabriel_path, dtype={"plant_code": str})
    plant_codes = gabriel_df["plant_code"].unique().tolist()
    original_df = load_original_opposition_scores(plant_codes)

    if original_df.empty:
        return "Opposition classify: No original results found for comparison.\n"

    merged = gabriel_df.merge(
        original_df, on="plant_code", how="inner", suffixes=("_gabriel", "_original")
    )

    if merged.empty:
        return "Opposition classify: No overlapping projects found for comparison.\n"

    lines = ["\n" + "=" * 60, "OPPOSITION CLASSIFICATION COMPARISON", "=" * 60]
    lines.append(f"Matched projects: {len(merged)}")

    var_stats = []
    for var in OPPOSITION_VARS:
        g_col = f"{var}_gabriel" if f"{var}_gabriel" in merged.columns else var
        o_col = f"{var}_original" if f"{var}_original" in merged.columns else var

        if g_col not in merged.columns or o_col not in merged.columns:
            continue

        g = merged[g_col].fillna(0).astype(int)
        o = merged[o_col].fillna(0).astype(int)

        accuracy = (g == o).mean()
        tp = ((g == 1) & (o == 1)).sum()
        fp = ((g == 1) & (o == 0)).sum()
        fn = ((g == 0) & (o == 1)).sum()
        tn = ((g == 0) & (o == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        # Cohen's kappa
        n = len(g)
        po = accuracy
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n > 0 else 0
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")

        var_stats.append({
            "variable": var,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "kappa": kappa,
            "original_prevalence": o.mean(),
            "gabriel_prevalence": g.mean(),
        })

    stats_df = pd.DataFrame(var_stats)
    lines.append("\n" + stats_df.to_string(index=False, float_format="%.3f"))

    # Summary
    lines.append(f"\nMean accuracy: {stats_df['accuracy'].mean():.3f}")
    lines.append(f"Mean kappa:    {stats_df['kappa'].mean():.3f}")
    lines.append(f"Mean precision: {stats_df['precision'].mean():.3f}")
    lines.append(f"Mean recall:   {stats_df['recall'].mean():.3f}")

    stats_df.to_csv(COMPARISON_DIR / "opposition_comparison_stats.csv", index=False)
    merged.to_csv(COMPARISON_DIR / "opposition_comparison_raw.csv", index=False)
    lines.append(f"\nSaved to: {COMPARISON_DIR / 'opposition_comparison_stats.csv'}")

    return "\n".join(lines) + "\n"


def compare_narratives() -> str:
    """Save side-by-side narrative comparisons for qualitative review."""
    gabriel_path = GABRIEL_DIR / "narrative_extract" / "results.csv"
    if not gabriel_path.exists():
        return "Narrative extract: GABRIEL results not found. Skipping.\n"

    gabriel_df = pd.read_csv(gabriel_path, dtype={"plant_code": str})
    plant_codes = gabriel_df["plant_code"].unique().tolist()
    original_df = load_original_opposition_scores(plant_codes)

    if original_df.empty:
        return "Narrative extract: No original results found for comparison.\n"

    merged = gabriel_df[["plant_code", "narrative"]].merge(
        original_df[["plant_code", "original_narrative"]],
        on="plant_code",
        how="inner",
    )

    if merged.empty:
        return "Narrative extract: No overlapping projects found for comparison.\n"

    # Save side-by-side for qualitative review
    lines_out = []
    for _, row in merged.iterrows():
        lines_out.append(f"=== Plant Code: {row['plant_code']} ===")
        lines_out.append(f"ORIGINAL (Claude Opus):\n{row['original_narrative']}\n")
        lines_out.append(f"GABRIEL:\n{row['narrative']}\n")
        lines_out.append("")

    narrative_path = COMPARISON_DIR / "narrative_comparison.txt"
    with open(narrative_path, "w") as f:
        f.write("\n".join(lines_out))

    merged.to_csv(COMPARISON_DIR / "narrative_comparison.csv", index=False)

    lines = ["\n" + "=" * 60, "NARRATIVE COMPARISON", "=" * 60]
    lines.append(f"Compared {len(merged)} project narratives")
    lines.append(f"Side-by-side saved to: {narrative_path}")
    lines.append(f"CSV saved to: {COMPARISON_DIR / 'narrative_comparison.csv'}")

    return "\n".join(lines) + "\n"


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print("GABRIEL vs Original Pipeline Comparison")
    print("=" * 60)

    report_parts = []
    report_parts.append(compare_article_relevance())
    report_parts.append(compare_content_relevance())
    report_parts.append(compare_opposition())
    report_parts.append(compare_narratives())

    full_report = "\n".join(report_parts)
    print(full_report)

    report_path = COMPARISON_DIR / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
