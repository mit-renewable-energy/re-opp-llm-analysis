"""
Project Processing Pipeline for Renewable Energy Opposition LLM Analysis

This module implements the core data processing pipeline:
1. Content scraping from search results (via Modal)
2. Article-level relevance scoring
3. Content-level relevance scoring
4. Opposition/support analysis and scoring

Usage:
    The pipeline is executed in stages by uncommenting the appropriate sections
    in the __main__ block. Each stage reads from and writes to JSON files in
    data/processed/results/, enabling incremental processing and debugging.

Stages:
    1. Search Result Generation (run via src/scraping/execute_searches.py)
    2. Content Scraping: Uncomment Modal scraping code
    3. Article Relevance: Uncomment get_relevance_scores section
    4. Content Relevance: Uncomment get_content_relevance section
    5. Project Summary: Uncomment get_project_summary section

Required Environment Variables:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, BRIGHTDATA_SERP_KEY
"""

from tqdm import tqdm
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.parse
import os
import json
from modal import Image
import pandas as pd
import modal
import requests
from unstructured.partition.auto import partition
from unstructured.cleaners.core import group_broken_paragraphs
from pydantic import BaseModel, Field, ValidationError, conint, SkipValidation
from typing import List
from anthropic import Anthropic
from openai import OpenAI
import instructor

from dotenv import load_dotenv

# Model identifiers
OPUS = "claude-3-opus-20240229"  # 200k context window
SONNET = "claude-3-sonnet-20240229"  # 200k context window
HAIKU = "claude-3-haiku-20240307"  # 200k context window
GPT4_TURBO = "gpt-4-turbo-2024-04-09"  # 128k context window
GPT35_TURBO = "gpt-3.5-turbo-0125"  # 16k context window

import sys
sys.path.append('.')
from config.config import (
    get_raw_data_path, get_processed_data_path, get_final_data_path,
    get_data_path, get_viz_path
)


# =============================================================================
# Modal Configuration for Remote Content Scraping
# =============================================================================

def pull_unstructured():
    """Download required NLTK data for text processing."""
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


bright_data_search_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libmagic-dev")
    .pip_install('unstructured[all-docs]')
    .pip_install("pandas", "numpy", "urllib3", "requests", "tqdm", "python-dotenv")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "python3-opencv")
    .run_commands("apt-get install -y poppler-utils tesseract-ocr")
    .pip_install("nltk")
)

stub = modal.Stub("bright_data_search", image=bright_data_search_image)


@stub.function(concurrency_limit=10)
def partition_content(search_results):
    """
    Extract and process content from search result URLs.

    Args:
        search_results: Dict containing 'organic' list of search results

    Returns:
        Dict with 'full_text' (concatenated content) and 'individual_results' (list)
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    def truncate_content(content, max_chars=10000):
        """Truncate content to first N characters with indicator."""
        if len(content) > max_chars:
            return content[:max_chars] + f"... [Truncated from {len(content)} chars]"
        return content

    current_results = search_results
    organic_results = current_results.get('organic', [])
    if organic_results == []:
        return {
            "full_text": "No organic results found.",
            "individual_results": []
        }

    end_result = []

    for index, search_result in enumerate(organic_results):
        current_result = {}
        try:
            current_result['link'] = search_result['link']
            r = requests.get(search_result['link'], headers, timeout=30)
            content_type = r.headers.get('content-type')
            if 'text/html' in content_type:
                text = requests.get(f"https://r.jina.ai/" + search_result['link'], timeout=30).text
            else:
                elements = partition(url=search_result['link'], headers=headers, timeout=30)
                text = "\n".join(element.text for element in elements)
            current_result['content'] = group_broken_paragraphs(truncate_content(text))
        except requests.exceptions.Timeout:
            current_result['content'] = 'Timed out'
        except:
            current_result['content'] = 'Could not access content'
        current_result['article_letter'] = chr(65 + index)
        current_result['link'] = search_result.get("link", "")
        current_result['title'] = search_result.get("title", "")
        current_result['description'] = search_result.get("description", "")
        end_result.append(current_result)

    return {
        "full_text": "\n".join([
            f"<doc>\nArticle Letter: {r['article_letter']}\n{r['title']}\n{r['description']}\n{r['content']}\n</doc>"
            for r in end_result
        ]),
        "individual_results": end_result
    }


# =============================================================================
# Search Functions
# =============================================================================

@backoff.on_exception(backoff.expo, Exception, max_time=120)
def get_search_results(search_query: str):
    """
    Execute Google search via BrightData proxy.

    Args:
        search_query: Search query string

    Returns:
        Dict containing search results with 'organic' key
    """
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(
            {'http': os.environ['BRIGHTDATA_SERP_KEY'],
             'https': os.environ['BRIGHTDATA_SERP_KEY']}))
    search_query = urllib.parse.quote_plus(search_query)
    results = json.loads(opener.open(f'http://www.google.com/search?q={search_query}&brd_json=1').read())
    return results


# =============================================================================
# Pydantic Models for Structured LLM Output
# =============================================================================

class ArticleScoreandJustification(BaseModel):
    """Single article relevance score with justification."""
    article_letter: str = Field(..., description="The letter of the article (A, B, C, etc.)")
    grade: int = Field(..., description="The score of the article (1-5). Score above 3 only if confident content is relevant to specific project.")
    justification: str = Field(..., description="Brief justification for the grade. Max 8 words.")


class ArticleRelevanceScores(BaseModel):
    """Collection of article-level relevance scores."""
    scores_and_justifications: List[ArticleScoreandJustification]


class RelevanceScoreandJustification(BaseModel):
    """Overall content relevance score for a project."""
    score: int = Field(..., description="Score 1-5 for whether ALL content is relevant. Default to 1 if unsure.")
    justification: str = Field(..., description="Brief justification with specific evidence. Max 8 words.")


class ContentRelevance(BaseModel):
    """Wrapper for content-level relevance scoring."""
    score_and_justification: List[RelevanceScoreandJustification]


class PerceptionsScoreandSources(BaseModel):
    """Binary score with source article references."""
    score: int = Field(..., description="Binary score: 1 if evidence found, 0 if not")
    sources: str = Field(..., description="Article letters with evidence (e.g., A, B, D) or brief justification. Max 8 words.")


class ProjectPerceptionVariables(BaseModel):
    """
    Full set of opposition/support variables extracted from project content.

    Variables:
        mention_support: Any mention of project support
        mention_opp: Any mention of project opposition
        physical_opp: Physical opposition (protests, demonstrations)
        policy_opp: Legislative/policy opposition (ordinances, moratoria)
        legal_opp: Legal challenges and court actions
        opinion_opp: Opinion editorials opposing project
        environmental_opp: Environmental concerns (wildlife, water, soil)
        participation_opp: Opposition due to lack of community participation
        tribal_opp: Tribal/Indigenous opposition
        health_opp: Health and safety concerns
        intergov_opp: Intergovernmental disagreements
        property_opp: Property value impact concerns
        compensation: Compensation/community benefits issues
        delay: Evidence of substantial project delays
        co_land_use: Evidence of co-existing land uses
        narrative: Summary of public perceptions
    """
    mention_support: List[SkipValidation[PerceptionsScoreandSources]] = Field(
        ..., description="Binary score for any mention of support with sources.")
    mention_opp: List[SkipValidation[PerceptionsScoreandSources]] = Field(
        ..., description="Binary score for any mention of opposition with sources.")
    physical_opp: int = Field(
        ..., description="1 if evidence of physical opposition (protests, marches), 0 if not.")
    policy_opp: int = Field(
        ..., description="1 if evidence of legislative/permitting opposition, 0 if not")
    legal_opp: int = Field(
        ..., description="1 if evidence of legal challenges, 0 if not")
    opinion_opp: int = Field(
        ..., description="1 if opinion-editorials opposing project exist, 0 if not")
    environmental_opp: int = Field(
        ..., description="1 if evidence of environmental concerns, 0 if not")
    participation_opp: int = Field(
        ..., description="1 if evidence of participation/fairness concerns, 0 if not")
    tribal_opp: int = Field(
        ..., description="1 if evidence of tribal opposition, 0 if not")
    health_opp: int = Field(
        ..., description="1 if evidence of health/safety concerns, 0 if not")
    intergov_opp: int = Field(
        ..., description="1 if evidence of intergovernmental disagreement, 0 if not")
    property_opp: int = Field(
        ..., description="1 if evidence of property value concerns, 0 if not")
    compensation: int = Field(
        ..., description="1 if evidence of compensation/benefits issues, 0 if not")
    delay: int = Field(
        ..., description="1 if evidence of substantial delay due to opposition, 0 if not")
    co_land_use: int = Field(
        ..., description="1 if evidence of co-existing land uses, 0 if not")
    narrative: str = Field(
        ..., description="3-4 sentence summary of public perceptions. 'No relevant info found.' if none.")


class ProjectSummary(BaseModel):
    """Wrapper for project perception analysis results."""
    all_scores_and_sources: List[ProjectPerceptionVariables]


# =============================================================================
# LLM Scoring Functions
# =============================================================================

def get_relevance_scores(search_query, search_results, plant_info):
    """
    Score article-level relevance using Claude Haiku.

    Args:
        search_query: Original search query
        search_results: Formatted string of search result metadata
        plant_info: Project identification string

    Returns:
        ArticleRelevanceScores with per-article grades
    """
    try:
        client = instructor.from_anthropic(Anthropic())
        relevance_scores = client.messages.create(
            model=HAIKU,
            response_model=ArticleRelevanceScores,
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "system", "content": f"You are an expert on public perceptions on large renewable energy projects. "
                 f"Your aim is to take a set of search results from Google corresponding "
                 f"to the following search query: {search_query} and determine whether or not the search results are "
                 f"relevant to our research question. Here are the search results: {search_results}"},
                {"role": "user", "content": f"Based on the title, display link, and description of each URL, we would "
                 f"like to identify which search results are most relevant to this research question: 'What is the narrative "
                 f"surrounding the development of this renewable energy project in this location, and what evidence of opposition "
                 f"or support for the project can be identified?' Score each search result based on the article letter with a number between 1-5, "
                 f"with 1 meaning that the article is least relevant and 5 being the most relevant to the research question. "
                 f"Here are examples of what might receive the following scores:"
                 f"\n1 - an article that does not mention renewable energy or the project in question ({plant_info}), but may have info about a different project or ordinance"
                 f"\n2 - an article that might be related to renewable energy near the location in question but does not mention the specific project ({plant_info})"
                 f"\n3 - an article that mentions the specific project and location in question ({plant_info}), but only provides basic information about the project and no information on opposition or support"
                 f"\n4 - an article that you are EXTREMELY CONFIDENT mentions the exact project and location in question ({plant_info})"
                 f"\n5 - an article that you are EXTREMELY CONFIDENT describes the narrative of the specific project development ({plant_info}), including mentions of opposition and support."
                 },
            ],
        )
        assert isinstance(relevance_scores, ArticleRelevanceScores)
        return relevance_scores
    except Exception as e:
        print("Error occurred: ", str(e))
        breakpoint()


@backoff.on_exception(backoff.expo, Exception, max_time=30)
def get_content_relevance(search_query, search_results, plant_info):
    """
    Score overall content relevance using Claude Sonnet.

    Args:
        search_query: Original search query
        search_results: Formatted string of search result metadata
        plant_info: Project identification string

    Returns:
        ContentRelevance with overall relevance score
    """
    client = instructor.from_anthropic((Anthropic))
    relevance_scores = client.messages.create(
        model=SONNET,
        response_model=ContentRelevance,
        max_tokens=4096,
        temperature=0.1,
        messages=[
            {"role": "system", "content": f"You are an expert on public perceptions on large renewable energy projects. "
             f"Your aim is to take a set of search results from Google corresponding "
             f"to the following search query: {search_query} and determine whether or not the search results are "
             f"relevant to our research question. Here are the search results: {search_results}"},
            {"role": "user", "content": f"Based on the description of each URL and other metadata, we would "
             f"like to identify which search results are most relevant to this research question: 'What is the narrative "
             f"surrounding the development of this renewable energy project in this location, and what evidence of opposition "
             f"or support for the project can be identified?' Score all of the search results as a whole with a number between 1-5, "
             f"with 1 meaning that the content is least relevant and 5 being the most relevant to the research question. "
             f"Here are examples of what might receive the following scores:"
             f"\n1 - NONE of the articles mention the specific project {plant_info} or renewable energy near the location, but they might refer to a different project or ordinance"
             f"\n2 - SOME of the articles might be related to renewable energy near the location in question but does not mention the specific project {plant_info}"
             f"\n3 - AT LEAST ONE article mentions the specific project: {plant_info} "
             f"\n4 - MOST of the articles mention the specific project: {plant_info}"
             f"\n5 - MOST of the articles mention the specific project {plant_info}, AND there are also mentions of opposition or support"},
        ],
    )
    assert isinstance(relevance_scores, ContentRelevance)
    return relevance_scores


def get_project_summary(plant_info, content):
    """
    Extract opposition/support variables from project content using Claude Opus.

    Args:
        plant_info: Project identification string
        content: Full text content from relevant articles

    Returns:
        ProjectSummary with all binary opposition variables and narrative
    """
    try:
        client = instructor.from_anthropic(Anthropic())
        project_perceptions = client.messages.create(
            model=OPUS,
            response_model=ProjectSummary,
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "system", "content": f'You are an expert on public perceptions on large renewable energy projects. Here is the name and location of the project in question ({plant_info}) from which the following search result content is generated: {content}.'},
                {"role": "user", "content": f'Our aim is to understand the public opinion and perceptions of a particular renewable energy project ({plant_info}) based solely on online media evidence from a search engine query on the project. Based on the full text content of all relevant search results, we would like to answer several binary questions about whether or not there is evidence of opposition or support for the project. Use only the text content provided to answer these questions with a "1" if evidence is found and "0" if not, and finally to create a one-paragraph summary of public perceptions of the project. Note that none of the info in the content may be relevant to the project in question, and if so, all integers should be 0 and narrative should be "No relevant info found." Remember: ONLY SCORE 1 if you are EXTREMELY CONFIDENT that there is evidence to support the score for the specific project and location ({plant_info}).'},
            ],
        )
        return project_perceptions
    except Exception as e:
        print("Error occurred: ", str(e))
        breakpoint()


# =============================================================================
# Modal Entry Point for Content Scraping
# =============================================================================

@stub.local_entrypoint()
def main():
    """Modal entry point for running content scraping remotely."""
    print("Running content scraping via Modal...")
    plant_codes = pd.read_csv("data/processed/search_ready_plants.csv")['plant_code']

    plant_codes = [
        pc for pc in plant_codes
        if not os.path.exists(f'data/processed/results/content/{pc}.json')
    ]

    search_results = []
    for plant_code in plant_codes[:1]:
        with open(f'data/processed/results/search/{plant_code}.json', 'r') as f:
            search_result = json.load(f)
            search_results.append(search_result)

    partitioned_results = partition_content.map(search_results)

    for plant_code, partitioned_result in tqdm(zip(plant_codes, partitioned_results), desc="Processing plant codes"):
        with open(f'data/processed/results/content/{plant_code}.json', 'w') as f:
            json.dump(partitioned_result, f)


# =============================================================================
# Main Execution - Uncomment sections as needed
# =============================================================================

if __name__ == "__main__":
    load_dotenv()

    # =========================================================================
    # STAGE 1: Search Result Generation
    # Run via: python src/scraping/execute_searches.py
    # Or uncomment below for direct execution
    # =========================================================================
    # df = pd.read_csv("data/processed/search_ready_plants.csv")
    # queries = list(df['search_query'])
    # plant_codes = list(df['plant_code'])
    #
    # with ThreadPoolExecutor(max_workers=100) as executor:
    #     future_to_plant_code = {
    #         executor.submit(get_search_results, query): plant_code
    #         for plant_code, query in zip(plant_codes, queries)
    #         if not os.path.exists(f'data/processed/results/search/{plant_code}.json')
    #     }
    #     for future in tqdm(as_completed(future_to_plant_code), total=len(future_to_plant_code)):
    #         plant_code = future_to_plant_code[future]
    #         result = future.result()
    #         with open(f'data/processed/results/search/{plant_code}.json', 'w') as f:
    #             json.dump(result, f)

    # =========================================================================
    # STAGE 2: Article Relevance Scoring
    # Scores individual articles on 1-5 scale
    # =========================================================================
    # df = pd.read_csv("data/processed/plants_with_content.csv")
    # plant_codes = list(df['plant_code'])
    # plant_codes = [
    #     pc for pc in plant_codes
    #     if not os.path.exists(f'data/processed/results/article_relevance/{pc}.json')
    # ]
    # search_queries = [query for code, query in zip(df['plant_code'], df['search_query']) if code in plant_codes]
    # plant_infos = [info for code, info in zip(df['plant_code'], df['plant_info']) if code in plant_codes]
    #
    # def process_plant_code(plant_code, search_query, plant_info):
    #     search_file_path = f'data/processed/results/search/{plant_code}.json'
    #     with open(search_file_path, 'r') as f:
    #         search_data = json.load(f)
    #         organic_results = search_data.get('organic', [])
    #         if organic_results == []:
    #             return []
    #         formatted_results = []
    #         for index, article in enumerate(organic_results):
    #             formatted_article = f"Article Letter: {article.get('article_letter', 'No article letter')}, Title: {article.get('title', 'No article title')}, Display URL: {article.get('display_link', 'No article display link.')}, Description: {article.get('description', 'No article description.')}"
    #             formatted_results.append(formatted_article)
    #         search_result_string = "<article>" + "</article>\n<article>".join(formatted_results) + "</article>"
    #         return get_relevance_scores(search_query, search_result_string, plant_info)
    #
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     future_to_plant_code = {
    #         executor.submit(process_plant_code, plant_code, search_query, plant_info): plant_code
    #         for plant_code, search_query, plant_info in zip(plant_codes, search_queries, plant_infos)
    #     }
    #     for future in tqdm(as_completed(future_to_plant_code), total=len(future_to_plant_code)):
    #         plant_code = future_to_plant_code[future]
    #         relevance_score = future.result()
    #         if relevance_score == []:
    #             print("Plant code: ", plant_code, " has no organic results.")
    #         with open(f'data/processed/results/article_relevance/{plant_code}.json', 'w') as f:
    #             json.dump(relevance_score.model_dump() if relevance_score else relevance_score, f)

    # =========================================================================
    # STAGE 3: Content Relevance Scoring
    # Scores overall content relevance for project
    # =========================================================================
    # df = pd.read_csv("data/processed/plants_with_content.csv")
    # plant_codes = list(df['plant_code'])
    # plant_codes = [
    #     pc for pc in plant_codes
    #     if not os.path.exists(f'data/processed/results/content_relevance/{pc}.json')
    # ]
    # search_queries = [query for code, query in zip(df['plant_code'], df['search_query']) if code in plant_codes]
    # plant_infos = [info for code, info in zip(df['plant_code'], df['plant_info']) if code in plant_codes]
    #
    # def process_content_relevance(plant_code, search_query, plant_info):
    #     search_file_path = f'data/processed/results/search/{plant_code}.json'
    #     with open(search_file_path, 'r') as f:
    #         search_data = json.load(f)
    #         organic_results = search_data.get('organic', [])
    #         if organic_results == []:
    #             return []
    #         formatted_results = []
    #         for index, article in enumerate(organic_results):
    #             formatted_article = f"Article Letter: {article['article_letter']}, Title: {article['title']}, Display URL: {article['display_link']}, Description: {article['description']}"
    #             formatted_results.append(formatted_article)
    #         search_result_string = "<article>" + "</article>\n<article>".join(formatted_results) + "</article>"
    #         return get_content_relevance(search_query, search_result_string, plant_info)
    #
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     future_to_plant_code = {
    #         executor.submit(process_content_relevance, plant_code, search_query, plant_info): plant_code
    #         for plant_code, search_query, plant_info in zip(plant_codes, search_queries, plant_infos)
    #     }
    #     for future in tqdm(as_completed(future_to_plant_code), total=len(future_to_plant_code)):
    #         plant_code = future_to_plant_code[future]
    #         content_relevance = future.result()
    #         with open(f'data/processed/results/content_relevance/{plant_code}.json', 'w') as f:
    #             json.dump(content_relevance.model_dump() if content_relevance else content_relevance, f)

    # =========================================================================
    # STAGE 4: Project Summary / Opposition Analysis
    # Extracts 15 binary variables and narrative
    # =========================================================================
    df = pd.read_csv("data/processed/plants_with_relevance.csv")
    queries = list(df['search_query'])
    plant_codes = list(df['plant_code'])

    plant_codes = [
        pc for pc in plant_codes
        if not os.path.exists(f'data/processed/results/scores/{pc}.json')
    ]

    plant_infos = [info for code, info in zip(df['plant_code'], df['plant_info']) if code in plant_codes]
    all_content = []
    for plant_code in plant_codes:
        with open(f'data/processed/results/content/{plant_code}.json', 'r') as f:
            content = json.load(f)
            all_content.append(content['relevant_content_text'])
    print(f"Processing {len(plant_codes)} remaining projects...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_plant_code = {
            executor.submit(get_project_summary, plant_info, content): plant_code
            for plant_code, plant_info, content in zip(plant_codes, plant_infos, all_content)
            if not os.path.exists(f'data/processed/results/scores/{plant_code}.json')
        }
        for future in tqdm(as_completed(future_to_plant_code), total=len(future_to_plant_code)):
            plant_code = future_to_plant_code[future]
            summary = future.result()
            with open(f'data/processed/results/scores/{plant_code}.json', 'w') as f:
                json.dump(summary.model_dump(), f)
