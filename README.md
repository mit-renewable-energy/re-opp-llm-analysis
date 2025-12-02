# Renewable Energy Opposition LLM Analysis

Analyzes public perceptions of U.S. renewable energy projects (solar and wind) using EIA 2022 data. The pipeline collects online content, scores relevance using LLMs, and extracts structured data about opposition and support patterns.

## Project Structure

```
re-opp-llm-analysis/
├── data/
│   ├── raw/                    # Original source data (EIA 2022)
│   ├── processed/              # Intermediate processing results
│   │   └── results/            # Per-plant JSON outputs
│   └── final/                  # Analysis-ready datasets
├── src/
│   ├── pipeline/               # Main processing pipeline (stages 2-4)
│   ├── scraping/               # Search result generation (stage 1)
│   ├── validation/             # Human validation interfaces
│   └── analysis/               # Visualization generation
├── scripts/                    # Paper-specific plot/table generation
├── notebooks/                  # Interactive analysis notebooks
├── config/                     # Configuration and styling
├── utils/                      # S3 data management utilities
├── viz/                        # Generated outputs
└── archive/                    # Deprecated files
```

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
apt-get install libmagic-dev libgl1-mesa-glx libglib2.0-0 python3-opencv poppler-utils tesseract-ocr

# macOS
brew install libmagic poppler tesseract
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with required API keys:

```bash
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
BRIGHTDATA_SERP_KEY=<your-key>
AWS_DEFAULT_REGION=<region>
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-key>
```

## Data Pipeline

**Flow**: Raw EIA Data → Query Generation → Search → Content Scraping → Relevance Scoring → Opposition Analysis → Final Dataset

1. **Raw Data**: EIA 2022 plant and generation data
2. **Query Generation**: Create search queries from plant info
3. **Search Execution**: Generate Google search results via BrightData
4. **Content Processing**: Scrape and parse article content via Modal
5. **Relevance Scoring**: Score article relevance (1-5 scale)
6. **Opposition Analysis**: Extract 15 binary opposition/support variables
7. **Final Dataset**: Combine results into analysis-ready format

### Running the Pipeline

The pipeline is executed in stages via `src/pipeline/process_projects.py`. Uncomment the appropriate section for each stage:

```bash
# Stage 1: Generate search results
python src/scraping/execute_searches.py

# Stages 2-4: Run remaining pipeline stages
python src/pipeline/process_projects.py
```

### S3 Data Access

Large data files (25k+ per-plant JSONs) are stored in S3. Use the utilities in `utils/s3_data.py`:

```python
from utils.s3_data import ensure_data_available, sync_from_s3

# Download a specific file
path = ensure_data_available("data/final/analysis_with_relevance.pkl")

# Sync all data from S3
sync_from_s3()
```

## Opposition Variables

The pipeline extracts 15 binary variables:

| Variable | Description |
|----------|-------------|
| `mention_support` | Any mention of project support |
| `mention_opp` | Any mention of project opposition |
| `physical_opp` | Physical opposition (protests, demonstrations) |
| `policy_opp` | Legislative/policy opposition |
| `legal_opp` | Legal challenges and court actions |
| `opinion_opp` | Opinion editorials opposing project |
| `environmental_opp` | Environmental concerns |
| `participation_opp` | Lack of community participation concerns |
| `tribal_opp` | Tribal/Indigenous opposition |
| `health_opp` | Health and safety concerns |
| `intergov_opp` | Intergovernmental disagreements |
| `property_opp` | Property value impact concerns |
| `compensation` | Compensation/community benefits issues |
| `delay` | Evidence of project delays |
| `co_land_use` | Evidence of co-existing land uses |

## Validation

Human validation interface for assessing model accuracy:

```bash
# Run validation app
streamlit run src/validation/validate_results_app.py

# Simple labeling interface
streamlit run src/validation/simple_labeling_app.py

# Analyze validation results
python src/validation/analyze_validation_results.py
```

## Visualization

Generate publication-ready plots:

```bash
python src/analysis/generate_visualizations.py
```

For interactive analysis, see `notebooks/plots.ipynb` and `notebooks/analysis.ipynb`.

## Configuration

Centralized path management in `config/config.py`. Adjust paths and settings as needed for your environment.

## License

See LICENSE file.
