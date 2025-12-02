"""
Configuration module for Renewable Energy Project Dispute Characterization
Provides centralized settings for paths, visualization, and processing parameters

S3 Integration:
    Set USE_S3=true in environment to enable automatic S3 data fetching.
    When enabled, get_data_path() and related functions will automatically
    download missing files from S3.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 configuration
USE_S3 = os.getenv("USE_S3", "true").lower() == "true"

# Get project root directory (config is now in config/ subdirectory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def load_project_settings():
    """Load settings from .claude/settings.json or use defaults"""
    settings_path = PROJECT_ROOT / ".claude" / "settings.json"
    try:
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load settings from {settings_path}: {e}")
    
    # Fallback to default settings
    return get_default_settings()

def get_default_settings():
    """Default settings if .claude/settings.json doesn't exist"""
    return {
        "paths": {
            "data_dir": "./data",
            "raw_data_dir": "./data/raw",
            "processed_data_dir": "./data/processed", 
            "final_data_dir": "./data/final",
            "viz_dir": "./viz",
            "plots_dir": "./viz/plots",
            "validation_outputs_dir": "./viz/validation_outputs",
            "test_outputs_dir": "./viz/test_outputs",
            "src_dir": "./src",
            "config_dir": "./config",
            "exploratory_dir": "./exploratory",
            "archive_dir": "./archive"
        },
        "visualization": {
            "figure_format": "png",
            "figure_dpi": 300,
            "colorblind_palette": {
                "CB91_Blue": "#2CBDFE",
                "CB91_Green": "#47DBCD", 
                "CB91_Pink": "#F3A0F2",
                "CB91_Purple": "#9D2EC5",
                "CB91_Violet": "#661D98",
                "CB91_Amber": "#F5B14C"
            },
            "font_settings": {
                "family": "Arial",
                "sizes": {
                    "normal": 16,
                    "title": 20,
                    "label": 18
                }
            }
        }
    }

# Load settings
SETTINGS = load_project_settings()

# Path configurations
class Paths:
    """Centralized path management"""
    ROOT = PROJECT_ROOT
    DATA = PROJECT_ROOT / SETTINGS["paths"]["data_dir"]
    RAW_DATA = PROJECT_ROOT / SETTINGS["paths"]["raw_data_dir"]
    PROCESSED_DATA = PROJECT_ROOT / SETTINGS["paths"]["processed_data_dir"]
    FINAL_DATA = PROJECT_ROOT / SETTINGS["paths"]["final_data_dir"]
    VIZ = PROJECT_ROOT / SETTINGS["paths"]["viz_dir"]
    PLOTS = PROJECT_ROOT / SETTINGS["paths"]["plots_dir"]
    VALIDATION_OUTPUTS = PROJECT_ROOT / SETTINGS["paths"]["validation_outputs_dir"]
    TEST_OUTPUTS = PROJECT_ROOT / SETTINGS["paths"]["test_outputs_dir"]
    SRC = PROJECT_ROOT / SETTINGS["paths"]["src_dir"]
    CONFIG = PROJECT_ROOT / SETTINGS["paths"]["config_dir"]
    EXPLORATORY = PROJECT_ROOT / SETTINGS["paths"]["exploratory_dir"]
    ARCHIVE = PROJECT_ROOT / SETTINGS["paths"]["archive_dir"]

# Data file configurations  
class DataFiles:
    """Centralized data file management"""
    # Raw data files (in data/raw/)
    EIA_PLANTS_2022 = "eia_plants_2022.csv"
    EIA_GENERATION_2022 = "eia_generation_2022.csv"
    US_STATE_ABBREVIATIONS = "us_state_abbreviations.json"
    US_ABBREVIATIONS_TO_STATE = "us_abbreviations_to_state.json"
    
    # Processed data files (in data/processed/)
    SEARCH_READY_PLANTS = "search_ready_plants.csv"
    PLANTS_WITH_CONTENT = "plants_with_content.csv"
    PLANTS_WITH_RELEVANCE = "plants_with_relevance.csv"
    
    # Final analysis files (in data/final/)
    COMPLETE_ANALYSIS_DATASET = "complete_analysis_dataset.pkl"
    ANALYSIS_WITH_RELEVANCE = "analysis_with_relevance.pkl"
    VALIDATION_HUMAN_LABELS = "validation_human_labels.json"
    VALIDATION_SAMPLE_DATA = "validation_sample_data.pkl"
    
    # Legacy names for backward compatibility
    READY_TO_SEARCH = "search_ready_plants.csv"  # for backward compatibility
    POST_CONTENT_PLANTS = "plants_with_content.csv" 
    POST_RELEVANCE_PLANTS = "plants_with_relevance.csv"
    PLANT_2022 = "eia_plants_2022.csv"
    OPGEN_2022 = "eia_generation_2022.csv"
    VALIDATION_RESULTS = "validation_human_labels.json"
    MAIN_ANALYSIS_DATASET = "complete_analysis_dataset.pkl"
    DATASET_WITH_RELEVANCE = "analysis_with_relevance.pkl"

# Visualization configurations
class VizConfig:
    """Centralized visualization configuration"""
    
    # Color palette (colorblind-friendly)
    COLORS = SETTINGS["visualization"]["colorblind_palette"]
    CB91_Blue = COLORS["CB91_Blue"]
    CB91_Green = COLORS["CB91_Green"] 
    CB91_Pink = COLORS["CB91_Pink"]
    CB91_Purple = COLORS["CB91_Purple"]
    CB91_Violet = COLORS["CB91_Violet"]
    CB91_Amber = COLORS["CB91_Amber"]
    
    # Color list for cycling
    COLOR_LIST = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
    
    # Gradient colors
    CB91_Grad_BP = ['#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8',
                    '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
                    '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
                    '#568ae6', '#5986e4', '#5c81e2', '#607de0',
                    '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
                    '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
                    '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
                    '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
                    '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
                    '#a924b7', '#ac20b5', '#b01bb3', '#b317b1']
    
    # Font settings
    FONT_FAMILY = SETTINGS["visualization"]["font_settings"]["family"]
    FONT_SIZES = SETTINGS["visualization"]["font_settings"]["sizes"]
    
    # Output settings
    FIGURE_FORMAT = SETTINGS["visualization"]["figure_format"]
    FIGURE_DPI = SETTINGS["visualization"]["figure_dpi"]
    
    # Standard figure sizes
    FIG_SIZE_STANDARD = (10, 6)
    FIG_SIZE_LARGE = (14, 6) 
    FIG_SIZE_SQUARE = (8, 8)
    FIG_SIZE_MAP = (12, 8)

def setup_plot_style():
    """Set up matplotlib and seaborn styling for consistent plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=VizConfig.COLOR_LIST)
    
    # Set seaborn style
    sns.set(font=VizConfig.FONT_FAMILY,
            rc={
                'axes.axisbelow': False,
                'axes.edgecolor': 'lightgrey',
                'axes.facecolor': 'None',
                'axes.labelcolor': 'dimgrey',
                'axes.spines.right': False,
                'axes.spines.top': False,
                'figure.facecolor': 'white',
                'lines.solid_capstyle': 'round',
                'patch.edgecolor': 'w',
                'patch.force_edgecolor': True,
                'text.color': 'dimgrey',
                'xtick.bottom': False,
                'xtick.color': 'dimgrey',
                'xtick.direction': 'out',
                'xtick.top': False,
                'ytick.color': 'dimgrey',
                'ytick.direction': 'out',
                'ytick.left': False,
                'ytick.right': False
            })
    
    sns.set_context("notebook", rc={
        "font.size": VizConfig.FONT_SIZES["normal"],
        "axes.titlesize": VizConfig.FONT_SIZES["title"],
        "axes.labelsize": VizConfig.FONT_SIZES["label"]
    })
    
    sns.set_style("whitegrid", {'axes.grid': False})

def save_figure(filename, fig=None, **kwargs):
    """Save figure with consistent settings"""
    import matplotlib.pyplot as plt
    
    # Default save parameters
    save_params = {
        'dpi': VizConfig.FIGURE_DPI,
        'format': VizConfig.FIGURE_FORMAT,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    # Construct full path
    full_path = Paths.PLOTS / filename
    
    # Ensure directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    if fig is None:
        plt.savefig(full_path, **save_params)
    else:
        fig.savefig(full_path, **save_params)
    
    print(f"Saved: {full_path}")

# Processing configurations (for reference)
class ProcessingConfig:
    """Processing and API configurations"""
    MAX_WORKERS_SEARCH = 100
    MAX_WORKERS_RELEVANCE = 5 
    MAX_WORKERS_SUMMARY = 10
    
    # API Models
    CLAUDE_OPUS = "claude-3-opus-20240229"
    CLAUDE_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"
    GPT4_TURBO = "gpt-4-turbo-2024-04-09"
    GPT35_TURBO = "gpt-3.5-turbo-0125"
    
    # Content limits
    MAX_CHARS = 10000
    TIMEOUT_SECS = 30

# Opposition variables for analysis
OPPOSITION_VARIABLES = [
    "mention_support", "mention_opp", "physical_opp", "policy_opp", 
    "legal_opp", "opinion_opp", "environmental_opp", "participation_opp",
    "tribal_opp", "health_opp", "intergov_opp", "property_opp", 
    "compensation", "delay", "co_land_use"
]

def get_raw_data_path(filename):
    """Get full path for a raw data file"""
    return Paths.RAW_DATA / filename

def get_processed_data_path(filename):
    """Get full path for a processed data file"""
    return Paths.PROCESSED_DATA / filename

def get_final_data_path(filename):
    """Get full path for a final analysis file"""
    return Paths.FINAL_DATA / filename

def _try_s3_download(rel_path: str) -> bool:
    """Try to download a file from S3 if USE_S3 is enabled. Returns True if successful."""
    if not USE_S3:
        return False
    try:
        from utils.s3_data import ensure_data_available
        result = ensure_data_available(rel_path, PROJECT_ROOT)
        return result is not None
    except ImportError:
        # utils.s3_data not available
        return False
    except Exception as e:
        print(f"Warning: S3 download failed for {rel_path}: {e}")
        return False


def get_data_path(filename):
    """
    Get full path for a data file - tries different locations for backward compatibility.

    If USE_S3=true and file doesn't exist locally, attempts to download from S3.
    """
    # Try final data first
    if (Paths.FINAL_DATA / filename).exists():
        return Paths.FINAL_DATA / filename
    # Try processed data
    elif (Paths.PROCESSED_DATA / filename).exists():
        return Paths.PROCESSED_DATA / filename
    # Try raw data
    elif (Paths.RAW_DATA / filename).exists():
        return Paths.RAW_DATA / filename
    # Try S3 download if enabled
    elif USE_S3:
        # Try to download from S3 - check each location
        for subdir in ["data/final", "data/processed", "data/raw"]:
            rel_path = f"{subdir}/{filename}"
            if _try_s3_download(rel_path):
                return PROJECT_ROOT / rel_path
    # Default to processed for new files
    return Paths.PROCESSED_DATA / filename

def get_results_path(subfolder, filename):
    """Get full path for a results file (now in processed data)"""
    return Paths.PROCESSED_DATA / subfolder / filename

def get_viz_path(filename):
    """Get full path for a visualization file"""
    return Paths.PLOTS / filename

# Helper function to check if paths exist
def verify_paths():
    """Verify that all configured paths exist"""
    paths_to_check = [
        ("Data directory", Paths.DATA),
        ("Raw data directory", Paths.RAW_DATA),
        ("Processed data directory", Paths.PROCESSED_DATA),
        ("Final data directory", Paths.FINAL_DATA),
        ("Visualizations directory", Paths.PLOTS),
        ("Source directory", Paths.SRC),
        ("Config directory", Paths.CONFIG),
        ("Exploratory directory", Paths.EXPLORATORY),
    ]
    
    for name, path in paths_to_check:
        if path.exists():
            print(f" {name}: {path}")
        else:
            print(f" {name}: {path} (does not exist)")

if __name__ == "__main__":
    print("Renewable Energy Project Configuration")
    print("=" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    verify_paths()