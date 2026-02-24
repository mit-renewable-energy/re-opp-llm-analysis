"""
GABRIEL function wrappers for each pipeline stage.

Each function takes a prepared DataFrame and runs the corresponding
GABRIEL primitive (rate, classify, or extract).
"""

import gabriel
import pandas as pd
from pathlib import Path


# ============================================================================
# Stage 1: Article Relevance (gabriel.rate)
# ============================================================================

async def run_article_relevance(
    df: pd.DataFrame,
    save_dir: str,
    model: str = "gpt-5-nano",
) -> pd.DataFrame:
    """
    Rate article-level relevance on a 0-100 scale.

    Args:
        df: DataFrame with columns [plant_code, article_letter, article_text, plant_info].
            article_text is pre-formatted as:
            "Project: {plant_info}\nTitle: {title}\nDescription: {desc}\nURL: {link}"
        save_dir: Directory for GABRIEL checkpoints/outputs.
        model: OpenAI model to use.

    Returns:
        DataFrame with 0-100 relevance scores and a derived 1-5 scale column.
    """
    additional_instructions = (
        "You are scoring how relevant an article is to understanding public "
        "opposition or support for a specific renewable energy project. "
        "The project name and location are embedded at the start of each text.\n\n"
        "Scoring guidance (mapped to a 1-5 rubric):\n"
        "- 0-20 (maps to 1): Article does not mention renewable energy or the "
        "specific project, but may reference a different project or ordinance.\n"
        "- 21-40 (maps to 2): Article might relate to renewable energy near the "
        "location but does not mention the specific project.\n"
        "- 41-60 (maps to 3): Article mentions the specific project and location "
        "but only provides basic information, no opposition or support details.\n"
        "- 61-80 (maps to 4): You are EXTREMELY CONFIDENT the article mentions "
        "the exact project and location.\n"
        "- 81-100 (maps to 5): You are EXTREMELY CONFIDENT the article describes "
        "the narrative of the specific project development, including mentions of "
        "opposition and support.\n\n"
        "Score high ONLY if the article clearly discusses the specific project named."
    )

    result = await gabriel.rate(
        df,
        column_name="article_text",
        attributes={
            "relevance": (
                "How relevant this article is to understanding public opposition "
                "or support for the specified renewable energy project. Score high "
                "only if the article clearly discusses the specific project named."
            ),
        },
        save_dir=save_dir,
        model=model,
        additional_instructions=additional_instructions,
        reset_files=False,
    )

    # Derive 1-5 scale from 0-100
    result["relevance_1to5"] = pd.cut(
        result["relevance"],
        bins=[-1, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    # Carry over identifiers
    result["plant_code"] = df["plant_code"].values
    result["article_letter"] = df["article_letter"].values

    return result


# ============================================================================
# Stage 2: Content Relevance (gabriel.rate)
# ============================================================================

async def run_content_relevance(
    df: pd.DataFrame,
    save_dir: str,
    model: str = "gpt-5-nano",
) -> pd.DataFrame:
    """
    Rate overall content relevance per project on a 0-100 scale.

    Args:
        df: DataFrame with columns [plant_code, content_text, plant_info].
            content_text is pre-formatted as:
            "Project: {plant_info}\n\n{full_text}" (truncated to 8000 chars)
        save_dir: Directory for GABRIEL checkpoints/outputs.
        model: OpenAI model to use.

    Returns:
        DataFrame with 0-100 content relevance scores and derived 1-5 column.
    """
    additional_instructions = (
        "You are scoring how relevant a collection of search results is to "
        "understanding public perceptions of a specific renewable energy project. "
        "The project name and location are embedded at the start of each text.\n\n"
        "Scoring guidance (mapped to a 1-5 rubric):\n"
        "- 0-20 (maps to 1): NONE of the articles mention the specific project "
        "or renewable energy near the location.\n"
        "- 21-40 (maps to 2): SOME articles might relate to renewable energy "
        "near the location but do not mention the specific project.\n"
        "- 41-60 (maps to 3): AT LEAST ONE article mentions the specific project.\n"
        "- 61-80 (maps to 4): MOST articles mention the specific project.\n"
        "- 81-100 (maps to 5): MOST articles mention the specific project AND "
        "there are also mentions of opposition or support."
    )

    result = await gabriel.rate(
        df,
        column_name="content_text",
        attributes={
            "content_relevance": (
                "How relevant the overall collection of search results is to "
                "understanding public perceptions of the specified renewable "
                "energy project."
            ),
        },
        save_dir=save_dir,
        model=model,
        additional_instructions=additional_instructions,
        reset_files=False,
    )

    # Derive 1-5 scale from 0-100
    result["content_relevance_1to5"] = pd.cut(
        result["content_relevance"],
        bins=[-1, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    result["plant_code"] = df["plant_code"].values

    return result


# ============================================================================
# Stage 3: Opposition Classification (gabriel.classify)
# ============================================================================

OPPOSITION_LABELS = {
    "mention_support": "Any mention of support for this renewable energy project",
    "mention_opp": "Any mention of opposition to this renewable energy project",
    "physical_opp": "Evidence of physical opposition (protests, marches, demonstrations)",
    "policy_opp": "Evidence of legislative/permitting opposition (ordinances, moratoria, zoning)",
    "legal_opp": "Evidence of legal challenges and court actions against the project",
    "opinion_opp": "Opinion editorials opposing the project",
    "environmental_opp": "Environmental concerns (wildlife, water, soil impacts)",
    "participation_opp": "Concerns about lack of community participation or fairness",
    "tribal_opp": "Tribal or Indigenous opposition to the project",
    "health_opp": "Health and safety concerns related to the project",
    "intergov_opp": "Intergovernmental disagreements about the project",
    "property_opp": "Property value impact concerns",
    "compensation": "Issues with compensation or community benefits",
    "delay": "Evidence of substantial project delays due to opposition",
    "co_land_use": "Evidence of co-existing land uses (agriculture, grazing, etc.)",
}


async def run_opposition_classify(
    df: pd.DataFrame,
    save_dir: str,
    model: str = "gpt-5-nano",
) -> pd.DataFrame:
    """
    Classify project content into 15 binary opposition/support labels.

    Args:
        df: DataFrame with columns [plant_code, content_text].
            content_text is pre-formatted as:
            "Project: {plant_info}\n\n{relevant_content_text}"
        save_dir: Directory for GABRIEL checkpoints/outputs.
        model: OpenAI model to use.

    Returns:
        DataFrame with binary columns for each of the 15 labels.
    """
    additional_instructions = (
        "You are analyzing online media content about a specific renewable energy "
        "project to determine whether evidence exists for various types of "
        "opposition and support. The project name and location are embedded "
        "at the start of each text.\n\n"
        "IMPORTANT: Only assign a label if you are EXTREMELY CONFIDENT that "
        "there is clear evidence in the text for the specific project and "
        "location mentioned. If the content is not relevant to the named "
        "project, do not assign any labels.\n\n"
        "Label definitions:\n"
        + "\n".join(
            f"- {name}: {desc}" for name, desc in OPPOSITION_LABELS.items()
        )
    )

    result = await gabriel.classify(
        df,
        column_name="content_text",
        labels=OPPOSITION_LABELS,
        save_dir=save_dir,
        model=model,
        additional_instructions=additional_instructions,
        reset_files=False,
    )

    result["plant_code"] = df["plant_code"].values

    return result


# ============================================================================
# Stage 4: Narrative Extraction (gabriel.extract)
# ============================================================================

async def run_narrative_extract(
    df: pd.DataFrame,
    save_dir: str,
    model: str = "gpt-5-nano",
) -> pd.DataFrame:
    """
    Extract a narrative summary of public perceptions for each project.

    Args:
        df: DataFrame with columns [plant_code, content_text].
            Same input as opposition_classify.
        save_dir: Directory for GABRIEL checkpoints/outputs.
        model: OpenAI model to use.

    Returns:
        DataFrame with a narrative text column per project.
    """
    result = await gabriel.extract(
        df,
        column_name="content_text",
        attributes={
            "narrative": (
                "A 3-4 sentence summary of public perceptions of this renewable "
                "energy project based on the provided content. Cover any evidence "
                "of opposition, support, or community sentiment. If no relevant "
                "information is found, return 'No relevant info found.'"
            ),
        },
        save_dir=save_dir,
        model=model,
        modality="text",
        additional_instructions=(
            "The project name and location are embedded at the start of each text. "
            "Focus your summary on the specific project named. If the content does "
            "not contain information about the named project, return "
            "'No relevant info found.'"
        ),
        reset_files=False,
    )

    # GABRIEL extract may return more rows than input (multiple entities per text).
    # The input plant_code column is preserved by GABRIEL, so just deduplicate.
    if len(result) == len(df):
        result["plant_code"] = df["plant_code"].values
    elif "plant_code" in result.columns:
        # Deduplicate: keep first narrative per plant
        result = result.drop_duplicates(subset="plant_code", keep="first").reset_index(drop=True)
    else:
        result["plant_code"] = df["plant_code"].values[:len(result)]

    return result
