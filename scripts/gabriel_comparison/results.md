# GABRIEL Comparison Results

Cross-platform validation of the Claude-based opposition analysis pipeline using OpenAI's [GABRIEL](https://github.com/openai/GABRIEL) library with GPT-5-nano. This comparison evaluates whether a different LLM provider and prompting framework produces consistent classifications when applied to the same 5,011 renewable energy projects.

## Setup

| Parameter | Value |
|-----------|-------|
| Projects analyzed | 5,011 |
| GABRIEL model | `gpt-5-nano` |
| Original model | Claude Opus (via `instructor`) |
| Stages completed | Content relevance, Opposition classification |
| Stages partial | Narrative extraction (2,689/5,011 matched; API quota exhausted) |
| Total API cost | ~$5.60 |
| Total runtime | ~25 min (excluding data loading) |

The GABRIEL pipeline maps each original pipeline stage to a GABRIEL primitive:

| Stage | Original | GABRIEL Primitive |
|-------|----------|-------------------|
| Content relevance (1-5) | Claude Sonnet | `gabriel.rate` (0-100, mapped to 1-5) |
| 15 binary opposition variables | Claude Opus | `gabriel.classify` (binary labels) |
| Narrative summary | Claude Opus | `gabriel.extract` (free text) |

Article-level relevance was excluded from the comparison run to reduce cost.

---

## Opposition Classification (15 Binary Variables)

The primary comparison: both pipelines classify the same project content into 15 binary opposition/support categories. GABRIEL uses `gabriel.classify` with the same label definitions used in the original Claude prompts.

### Table 1: Per-Variable Agreement Metrics

| Variable | Accuracy | Precision | Recall | Kappa | Orig. Prev. | GABRIEL Prev. |
|----------|----------|-----------|--------|-------|-------------|---------------|
| mention_support | 0.720 | 0.612 | 0.535 | 0.364 | 0.349 | 0.305 |
| mention_opp | 0.712 | 0.782 | 0.596 | 0.425 | 0.505 | 0.385 |
| physical_opp | 0.979 | 0.310 | 0.456 | 0.359 | 0.014 | 0.020 |
| policy_opp | 0.813 | 0.485 | 0.588 | 0.416 | 0.180 | 0.219 |
| legal_opp | 0.791 | 0.653 | 0.640 | 0.498 | 0.298 | 0.292 |
| opinion_opp | 0.928 | 0.241 | 0.242 | 0.203 | 0.047 | 0.047 |
| environmental_opp | 0.852 | 0.597 | 0.528 | 0.472 | 0.179 | 0.158 |
| participation_opp | 0.905 | 0.149 | 0.473 | 0.191 | 0.030 | 0.094 |
| tribal_opp | 0.979 | 0.649 | 0.372 | 0.463 | 0.026 | 0.015 |
| health_opp | 0.896 | 0.529 | 0.598 | 0.502 | 0.111 | 0.126 |
| intergov_opp | 0.847 | 0.147 | 0.552 | 0.178 | 0.042 | 0.157 |
| property_opp | 0.845 | 0.717 | 0.372 | 0.409 | 0.200 | 0.104 |
| compensation | 0.883 | 0.168 | 0.390 | 0.182 | 0.046 | 0.107 |
| delay | 0.903 | 0.302 | 0.428 | 0.303 | 0.062 | 0.088 |
| co_land_use | 0.789 | 0.270 | 0.567 | 0.258 | 0.107 | 0.226 |
| **Mean** | **0.856** | **0.441** | **0.489** | **0.348** | | |

### Figure 1: Kappa and Accuracy by Variable

![Kappa and Accuracy](figures/fig1_kappa_accuracy.png)

Variables are sorted by Cohen's kappa (inter-rater agreement corrected for chance). Dashed lines indicate standard kappa interpretation thresholds. Key observations:

- **Best agreement** (kappa > 0.4, "moderate"): `health_opp`, `legal_opp`, `environmental_opp`, `tribal_opp`, `mention_opp`, `policy_opp`, `property_opp`. These variables have relatively clear-cut textual signals (e.g., lawsuits, environmental impact statements, tribal consultations) that both models can identify.
- **Weakest agreement** (kappa < 0.2, "slight"): `intergov_opp`, `compensation`, `participation_opp`. These require more subjective judgment about what constitutes "intergovernmental disagreement" or "lack of community participation," leading to divergent interpretations across models.
- **Accuracy vs. kappa gap**: Many variables show high accuracy (>0.85) but low kappa. This is a prevalence effect — rare labels (e.g., `physical_opp` at 1.4% prevalence) yield high accuracy by chance alone. Kappa corrects for this.

### Figure 2: Label Prevalence Comparison

![Prevalence Scatter](figures/fig2_prevalence_scatter.png)

Each point represents one of the 15 variables. Points on the diagonal indicate matching prevalence rates between the two pipelines. Observations:

- **Close agreement on common labels**: `mention_opp` (50.5% vs 38.5%), `legal_opp` (29.8% vs 29.2%), and `mention_support` (34.9% vs 30.5%) are the most prevalent variables and track reasonably well, though GABRIEL consistently predicts slightly lower rates for these.
- **GABRIEL over-predicts rare categories**: `intergov_opp` (4.2% original vs 15.7% GABRIEL), `co_land_use` (10.7% vs 22.6%), `compensation` (4.6% vs 10.7%), and `participation_opp` (3.0% vs 9.4%) all fall well above the diagonal. GPT-5-nano appears to have a lower threshold for flagging these categories.
- **GABRIEL under-predicts `property_opp`**: 20.0% original vs 10.4% GABRIEL. This is the only common variable where GABRIEL predicts substantially less than Claude.

### Figure 3: Precision-Recall by Variable

![Precision-Recall](figures/fig3_precision_recall.png)

Color encodes Cohen's kappa. Variables in the upper-right (high precision and recall) have the strongest cross-model agreement. Observations:

- **High precision, moderate recall**: `mention_opp` (precision 0.78) and `property_opp` (precision 0.72) — when GABRIEL predicts these, it usually agrees with Claude, but it misses some cases Claude catches.
- **Low precision, moderate recall**: `intergov_opp`, `compensation`, `participation_opp` — GABRIEL flags these frequently but many are false positives relative to Claude's classifications.
- The precision-recall tradeoff is consistent with GABRIEL's over-prediction of rare labels: it catches more true positives but at the cost of additional false positives.

### Figure 4: Classification Agreement Breakdown

![Agreement Breakdown](figures/fig4_agreement_breakdown.png)

Stacked bars show the proportion of projects in each confusion matrix cell (TP, FP, FN, TN). This visualizes both agreement and the nature of disagreements:

- For rare variables (physical, tribal, opinion), the bar is almost entirely true negatives — both models agree these are absent in most projects.
- For common variables (mention_opp, mention_support, legal_opp), the TP segment is substantial, showing both models agree on the presence of these labels in many projects.
- The FP (orange) and FN (yellow) segments indicate the direction of disagreement. `co_land_use` and `intergov_opp` have notably large FP segments (GABRIEL labels positive, Claude does not).

---

## Content Relevance

GABRIEL scored content relevance on a 0-100 scale using `gabriel.rate`, then mapped to 1-5 via equal-width bins. No per-plant original content relevance scores were available for direct comparison (the original pipeline did not persist these as individual JSONs for all projects), so this section reports the GABRIEL distribution only.

### Figure 5: Content Relevance Score Distribution

![Content Relevance Distribution](figures/fig5_content_relevance_dist.png)

The distribution is bimodal: a large cluster near 0 (projects with irrelevant search results) and a broad peak around 45-60 (projects with at least some relevant content). When mapped to the 1-5 scale:

| Score | Count | Percentage | Interpretation |
|-------|-------|------------|----------------|
| 1 (0-20) | 1,393 | 27.8% | No relevant articles found |
| 2 (21-40) | 741 | 14.8% | Some articles near location, none about specific project |
| 3 (41-60) | 1,881 | 37.5% | At least one article mentions the specific project |
| 4 (61-80) | 503 | 10.0% | Most articles mention the specific project |
| 5 (81-100) | 493 | 9.8% | Most articles mention the project with opposition/support |

The median score of 50 (mapping to 3) suggests that for most projects, search results contain at least some mention of the specific project but not necessarily detailed opposition/support information.

---

## Narrative Extraction (Partial: 2,689 / 5,011 Projects)

Narrative extraction completed for 2,905 of 5,011 projects before the OpenAI API quota was exhausted. Of these, 2,689 were successfully matched back to plant codes (93% match rate; unmatched cases are due to the model renaming projects in its output key). GABRIEL's checkpointing system saved progress, so re-running `--stage narrative` will resume from where it stopped.

### Narrative Coverage

| Category | Count | Percentage |
|----------|-------|------------|
| Both pipelines produced a narrative | 1,259 | 46.8% |
| Only Claude produced a narrative | 559 | 20.8% |
| Only GABRIEL produced a narrative | 270 | 10.0% |
| Neither produced a narrative | 601 | 22.4% |

Claude produced substantive narratives for 67.6% of projects compared to GABRIEL's 56.9%. This gap partly reflects GABRIEL's stricter "No relevant info found" threshold, and partly that ~42% of projects haven't been processed by GABRIEL yet (the remaining 2,322 are awaiting API quota replenishment).

GABRIEL narratives are longer on average (421 chars vs 347 chars for Claude), suggesting more detailed summaries when the model does find relevant information.

### Example: Side-by-Side Narrative Comparison

**Project 56355 — Whispering Willow Wind Farm, Franklin County, Iowa**

> **GABRIEL (GPT-5-nano):** The Whispering Willow Wind Farm in Franklin County has been the subject of public discussion about wind development and its impact on rural Iowa. Some articles describe controversy around the project, but one source notes that the controversy has been limited in Franklin County (McVicker). In associated transmission developments for Whispering Willow North, objections from landowners have occurred, notably from the Koenen family, though ITC Midwest reports progress in obtaining easements and emphasizes the interconnection benefits and public-interest rationale. Overall, sentiment appears mixed: a small but present opposition from certain landowners exists, while supporters and regulatory filings stress benefits to ratepayers and the public interest, with broader county-level controversy described as limited.

> **Claude (Opus):** The Whispering Willow Wind Farm - East in Franklin County, Iowa, developed by Alliant Energy, began operations in 2009 with a 200 MW capacity. There is some evidence of legal opposition from a single family related to eminent domain and property rights. However, the project was still completed, and Alliant Energy reports the controversy has been limited in the county.

Both models identify the same key facts (limited controversy, landowner opposition around property rights) but GABRIEL provides more granular detail (names the Koenen family, mentions ITC Midwest transmission developments, cites specific sources). Claude is more concise and emphasizes the outcome (project completed despite opposition).

The full side-by-side comparison for all 2,689 matched projects is saved in `outputs/comparison_results/narrative_comparison.csv`.

---

## Discussion

### What the agreement levels mean

The mean Cohen's kappa of 0.348 falls in the "fair" range (0.21-0.40) on the Landis & Koch scale. This is expected for a cross-model, cross-provider comparison where the two systems differ in:
- Base model architecture (Claude Opus vs GPT-5-nano)
- Prompting framework (structured outputs via `instructor` vs GABRIEL's rate/classify primitives)
- Model capability tier (Opus is a flagship model; GPT-5-nano is the smallest in its family)
- Prompt engineering (original pipeline used carefully tuned, multi-step prompts; GABRIEL received the same label definitions but through a standardized template)

For context, inter-annotator agreement on subjective text classification tasks in the social sciences typically ranges from kappa 0.4-0.7, and human-LLM agreement studies commonly report kappa 0.3-0.5.

### Systematic differences

GABRIEL (GPT-5-nano) shows a consistent pattern of **over-predicting rare categories**. This is most pronounced for:
- `intergov_opp`: 3.7x over-prediction (4.2% vs 15.7%)
- `participation_opp`: 3.1x over-prediction (3.0% vs 9.4%)
- `compensation`: 2.3x over-prediction (4.6% vs 10.7%)
- `co_land_use`: 2.1x over-prediction (10.7% vs 22.6%)

This suggests GPT-5-nano applies a lower confidence threshold for assigning labels — it finds weak signals that Claude Opus (with its "EXTREMELY CONFIDENT" instruction) does not flag. Whether this represents over-sensitivity in GPT-5-nano or under-sensitivity in Claude Opus is an open question that would require human validation labels to resolve.

### Implications for the paper

1. **The high-prevalence variables are robust**: `mention_opp`, `mention_support`, and `legal_opp` show reasonable agreement (kappa 0.36-0.50), suggesting these top-level findings are not artifacts of a single model's biases.
2. **Rare variables should be interpreted with more caution**: Categories like `intergov_opp`, `compensation`, and `participation_opp` show low cross-model agreement, meaning conclusions about these specific opposition types depend more heavily on which model is used.
3. **The overall opposition rate is model-sensitive**: Claude reports 50.5% of projects have any mention of opposition; GABRIEL reports 38.5%. This ~12pp gap is meaningful for aggregate statistics and suggests a confidence interval around the headline opposition rate.

---

## Reproducing These Results

```bash
# Generate figures
python scripts/gabriel_comparison/generate_figures.py

# Re-run comparison (requires pCloud data)
python scripts/gabriel_comparison/compare_results.py

# Re-run full pipeline (requires OpenAI API key)
python scripts/gabriel_comparison/run_gabriel_pipeline.py --sample 5011

# Complete narrative extraction (resumes from checkpoint)
python scripts/gabriel_comparison/run_gabriel_pipeline.py --stage narrative --sample 5011
```

Raw data files:
- `outputs/content_relevance/results.csv` — GABRIEL content relevance scores (5,011 rows)
- `outputs/opposition_classify/results.csv` — GABRIEL opposition classifications (5,011 rows)
- `outputs/narrative_extract/results.csv` — GABRIEL narrative extractions (2,689 rows, partial)
- `outputs/comparison_results/opposition_comparison_stats.csv` — Per-variable agreement metrics
- `outputs/comparison_results/opposition_comparison_raw.csv` — Row-level merged comparison data
- `outputs/comparison_results/narrative_comparison.csv` — Side-by-side narrative comparison (2,689 rows)
- `outputs/comparison_results/comparison_report.txt` — Text summary
