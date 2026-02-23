"""
GABRIEL comparison pipeline orchestrator.

Replicates the existing Claude-based analysis pipeline using OpenAI's GABRIEL
library for side-by-side comparison of approaches.

Usage:
    # Run all stages on 50-project sample (default)
    python scripts/gabriel_comparison/run_gabriel_pipeline.py

    # Run a specific stage
    python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage article_relevance

    # Run on more projects
    python scripts/gabriel_comparison/run_gabriel_pipeline.py --sample 100

    # Use a different model
    python scripts/gabriel_comparison/run_gabriel_pipeline.py --model gpt-5
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load API key before importing gabriel
load_dotenv(PROJECT_ROOT / ".env")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPEN_AI_API_KEY_2026", "")

from gabriel_stages import (
    run_article_relevance,
    run_content_relevance,
    run_narrative_extract,
    run_opposition_classify,
)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
# pCloud has the full content/search/scores JSONs; local repo only has a subset
PCLOUD_RESULTS_DIR = Path.home() / "pCloud Drive" / "MIT" / "MIT Work" / "Renewable Energy UROP" / "dispute-characterization" / "data" / "processed" / "results"
RESULTS_DIR = PCLOUD_RESULTS_DIR if PCLOUD_RESULTS_DIR.exists() else PROCESSED_DIR / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MAX_CONTENT_CHARS = 8000


def load_plants_df() -> pd.DataFrame:
    """Load the plants CSV with relevance scores."""
    csv_path = PROCESSED_DIR / "plants_with_relevance.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DIR / "plants_with_content.csv"
    if not csv_path.exists():
        csv_path = PROCESSED_DIR / "search_ready_plants.csv"
    print(f"Loading plants from: {csv_path}")
    return pd.read_csv(csv_path)


def load_content_json(plant_code: str) -> dict | None:
    """Load content JSON for a single plant."""
    path = RESULTS_DIR / "content" / f"{plant_code}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_relevant_content_json(plant_code: str) -> dict | None:
    """Load relevant_content JSON (filtered articles) for a single plant."""
    path = RESULTS_DIR / "relevant_content" / f"{plant_code}.json"
    if not path.exists():
        return load_content_json(plant_code)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return load_content_json(plant_code)


def truncate(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Truncate text to max_chars with indicator."""
    if len(text) > max_chars:
        return text[:max_chars] + f"... [Truncated from {len(text)} chars]"
    return text


# ============================================================================
# DataFrame Preparation
# ============================================================================

def prepare_article_relevance_df(
    plants_df: pd.DataFrame, sample_codes: list[str]
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per (plant_code, article) for article-level
    relevance rating. Embeds plant_info into the article text.
    """
    rows = []
    for _, row in plants_df[plants_df["plant_code"].isin(sample_codes)].iterrows():
        pc = row["plant_code"]
        plant_info = row["plant_info"]

        # Load search results for article metadata
        search_path = RESULTS_DIR / "search" / f"{pc}.json"
        if not search_path.exists():
            continue
        try:
            with open(search_path) as f:
                search_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        organic = search_data.get("organic", [])
        if not organic:
            continue

        for article in organic:
            article_text = (
                f"Project: {plant_info}\n"
                f"Title: {article.get('title', 'No title')}\n"
                f"Description: {article.get('description', 'No description')}\n"
                f"URL: {article.get('link', 'No link')}"
            )
            rows.append({
                "plant_code": str(pc),
                "article_letter": article.get("article_letter", chr(65 + organic.index(article))),
                "article_text": article_text,
                "plant_info": plant_info,
            })

    return pd.DataFrame(rows)


def preload_all_content(plants_df: pd.DataFrame, sample_codes: list) -> dict:
    """
    Load content and relevant_content JSONs for all sample plants once.
    Returns dict keyed by plant_code with {content, relevant_content, plant_info}.
    """
    cache = {}
    subset = plants_df[plants_df["plant_code"].isin(sample_codes)]
    for i, (_, row) in enumerate(subset.iterrows()):
        if i % 500 == 0:
            print(f"  Pre-loading data: {i}/{len(subset)}...", flush=True)
        pc = row["plant_code"]
        cache[pc] = {
            "plant_info": row["plant_info"],
            "content": load_content_json(pc),
            "relevant_content": load_relevant_content_json(pc),
        }
    print(f"  Pre-loaded {len(cache)} plants", flush=True)
    return cache


def prepare_content_relevance_df(
    plants_df: pd.DataFrame, sample_codes: list[str], cache: dict
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per plant_code for content-level relevance
    rating. Concatenates all article text and embeds plant_info.
    """
    rows = []
    for pc in sample_codes:
        entry = cache.get(pc)
        if entry is None or entry["content"] is None:
            continue
        plant_info = entry["plant_info"]
        full_text = entry["content"].get("full_text", "")
        content_text = truncate(f"Project: {plant_info}\n\n{full_text}")
        rows.append({
            "plant_code": str(pc),
            "content_text": content_text,
            "plant_info": plant_info,
        })
    return pd.DataFrame(rows)


def prepare_opposition_df(
    plants_df: pd.DataFrame, sample_codes: list[str], cache: dict
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per plant_code for opposition classification
    and narrative extraction. Uses relevant_content_text if available, else
    full_text.
    """
    rows = []
    for pc in sample_codes:
        entry = cache.get(pc)
        if entry is None:
            continue
        rc = entry["relevant_content"]
        if rc is None:
            continue
        plant_info = entry["plant_info"]
        text = rc.get("relevant_content_text", rc.get("full_text", ""))
        content_text = truncate(f"Project: {plant_info}\n\n{text}")
        rows.append({
            "plant_code": str(pc),
            "content_text": content_text,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Pipeline Runner
# ============================================================================

async def run_pipeline(stage: str, sample: int, model: str) -> None:
    """Run the specified GABRIEL pipeline stage(s)."""
    plants_df = load_plants_df()
    # Keep plant_code as native type (int) for DataFrame matching
    all_codes = plants_df["plant_code"].tolist()

    # Filter to codes that have content data (file names are str)
    available_codes = [
        c for c in all_codes
        if (RESULTS_DIR / "content" / f"{c}.json").exists()
    ]

    sample_codes = available_codes[:sample]
    print(f"Selected {len(sample_codes)} projects (from {len(available_codes)} available)")

    # Ensure output directories exist
    for subdir in ["article_relevance", "content_relevance", "opposition_classify", "narrative_extract"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    stages_to_run = (
        ["content_relevance", "opposition", "narrative"]
        if stage == "all"
        else [stage]
    )

    # Pre-load all content data once to avoid repeated pCloud reads
    print("\nPre-loading content data...")
    cache = preload_all_content(plants_df, sample_codes)

    for s in stages_to_run:
        print(f"\n{'='*60}")
        print(f"Running stage: {s}")
        print(f"{'='*60}")

        if s == "article_relevance":
            df = prepare_article_relevance_df(plants_df, sample_codes)
            if df.empty:
                print("No data for article relevance. Skipping.")
                continue
            print(f"Prepared {len(df)} article rows for {df['plant_code'].nunique()} projects")
            result = await run_article_relevance(
                df, str(OUTPUT_DIR / "article_relevance"), model
            )
            out_path = OUTPUT_DIR / "article_relevance" / "results.csv"
            result.to_csv(out_path, index=False)
            print(f"Saved article relevance results to {out_path}")

        elif s == "content_relevance":
            df = prepare_content_relevance_df(plants_df, sample_codes, cache)
            if df.empty:
                print("No data for content relevance. Skipping.")
                continue
            print(f"Prepared {len(df)} project rows for content relevance")
            result = await run_content_relevance(
                df, str(OUTPUT_DIR / "content_relevance"), model
            )
            out_path = OUTPUT_DIR / "content_relevance" / "results.csv"
            result.to_csv(out_path, index=False)
            print(f"Saved content relevance results to {out_path}")

        elif s == "opposition":
            df = prepare_opposition_df(plants_df, sample_codes, cache)
            if df.empty:
                print("No data for opposition classification. Skipping.")
                continue
            print(f"Prepared {len(df)} project rows for opposition classification")
            result = await run_opposition_classify(
                df, str(OUTPUT_DIR / "opposition_classify"), model
            )
            out_path = OUTPUT_DIR / "opposition_classify" / "results.csv"
            result.to_csv(out_path, index=False)
            print(f"Saved opposition classification results to {out_path}")

        elif s == "narrative":
            df = prepare_opposition_df(plants_df, sample_codes, cache)
            if df.empty:
                print("No data for narrative extraction. Skipping.")
                continue
            print(f"Prepared {len(df)} project rows for narrative extraction")
            result = await run_narrative_extract(
                df, str(OUTPUT_DIR / "narrative_extract"), model
            )
            out_path = OUTPUT_DIR / "narrative_extract" / "results.csv"
            result.to_csv(out_path, index=False)
            print(f"Saved narrative extraction results to {out_path}")

        else:
            print(f"Unknown stage: {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GABRIEL comparison pipeline for renewable energy opposition analysis"
    )
    parser.add_argument(
        "--stage",
        choices=["article_relevance", "content_relevance", "opposition", "narrative", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of projects to process (default: 50)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="OpenAI model to use (default: gpt-5-nano)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPEN_AI_API_KEY_2026 not found in .env file.")
        print("Add OPEN_AI_API_KEY_2026=<your-key> to your .env file.")
        sys.exit(1)

    print(f"GABRIEL Comparison Pipeline")
    print(f"  Stage: {args.stage}")
    print(f"  Sample size: {args.sample}")
    print(f"  Model: {args.model}")

    asyncio.run(run_pipeline(args.stage, args.sample, args.model))
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
