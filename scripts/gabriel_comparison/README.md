# GABRIEL Comparison Pipeline

Replicates the existing Claude-based analysis pipeline using OpenAI's [GABRIEL library](https://github.com/openai/GABRIEL) for side-by-side comparison.

## Setup

```bash
pip install openai-gabriel
```

Add `OPEN_AI_API_KEY_2026` to your `.env` file.

## Pipeline Stage Mapping

| Original Stage | Original Model | GABRIEL Function | Notes |
|---|---|---|---|
| Article Relevance (1-5) | Claude Haiku | `gabriel.rate` (0-100) | Mapped back to 1-5 for comparison |
| Content Relevance (1-5) | Claude Sonnet | `gabriel.rate` (0-100) | One row per project |
| 15 Binary Opposition Vars | Claude Opus | `gabriel.classify` | Multi-label binary classification |
| Narrative Summary | Claude Opus | `gabriel.extract` | Free-text extraction |

## Usage

```bash
# Run all stages on 50-project sample (default)
python scripts/gabriel_comparison/run_gabriel_pipeline.py

# Run a specific stage
python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage article_relevance
python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage content_relevance
python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage opposition
python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage narrative

# Adjust sample size
python scripts/gabriel_comparison/run_gabriel_pipeline.py --sample 100

# Use a different model
python scripts/gabriel_comparison/run_gabriel_pipeline.py --model gpt-5

# Compare results against original pipeline
python scripts/gabriel_comparison/compare_results.py
```

## Output Structure

```
scripts/gabriel_comparison/outputs/
├── article_relevance/   # GABRIEL rate results + checkpoints
│   └── results.csv
├── content_relevance/   # GABRIEL rate results + checkpoints
│   └── results.csv
├── opposition_classify/ # GABRIEL classify results + checkpoints
│   └── results.csv
├── narrative_extract/   # GABRIEL extract results + checkpoints
│   └── results.csv
└── comparison_results/  # Comparison analysis
    ├── comparison_report.txt
    ├── article_relevance_comparison.csv
    ├── content_relevance_comparison.csv
    ├── opposition_comparison_stats.csv
    ├── opposition_comparison_raw.csv
    ├── narrative_comparison.txt
    └── narrative_comparison.csv
```

## Key Design Decisions

- **Embedded context**: Since GABRIEL applies the same attribute/label definitions to all rows, project-specific context (plant_info) is embedded directly into each text field.
- **Content truncation**: Content is truncated to 8,000 characters per project to stay within context limits.
- **Checkpointing**: GABRIEL's `save_dir` with `reset_files=False` enables resuming interrupted runs.
- **Scale mapping**: GABRIEL's 0-100 ratings are mapped to 1-5 using equal-width bins (0-20, 21-40, 41-60, 61-80, 81-100) for comparison with original scores.

## Files

| File | Description |
|---|---|
| `run_gabriel_pipeline.py` | Main orchestrator: loads data, prepares DataFrames, runs stages |
| `gabriel_stages.py` | Async wrappers around `gabriel.rate`, `gabriel.classify`, `gabriel.extract` |
| `compare_results.py` | Loads original + GABRIEL results and computes agreement metrics |
