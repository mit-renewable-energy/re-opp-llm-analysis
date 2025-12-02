#!/usr/bin/env python3
"""
Corrected plotting functions that match the original plots.ipynb exactly
These two functions replicate the exact plots from the notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Import configuration
import sys
sys.path.append('.')
from config.config import Paths, setup_plot_style, get_data_path

# Use test directory for output
TEST_VIZ_DIR = Paths.ROOT / "visualizations_test_corrected"

def save_figure_corrected(filename, **kwargs):
    """Save figure to corrected test directory"""
    save_params = {
        'dpi': 300,
        'format': 'png',
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    full_path = TEST_VIZ_DIR / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(full_path, **save_params)
    print(f" Saved corrected plot: {full_path}")

def load_and_filter_data():
    """Load and filter data exactly as in the notebook"""
    print(" Loading and filtering data...")
    
    # Try different possible dataset files
    for dataset_name in ["data/final/analysis_intermediate.pkl", "dataset_analysis_allscores.pkl", "data/final/complete_analysis_dataset.pkl"]:
        dataset_path = get_data_path(dataset_name)
        if dataset_path.exists():
            print(f"Loading {dataset_name}...")
            with open(dataset_path, "rb") as f:
                joined_data = pickle.load(f)
            break
    else:
        print(" No suitable dataset found")
        return None
    
    print(f"Loaded dataset with {len(joined_data)} projects")
    print(f"Available columns: {list(joined_data.columns)}")
    
    # Filter to renewable energy types - check for various column name possibilities
    technology_col = None
    for col_name in ['TECHNOLOGY', 'technology', 'tech_type', 'Technology']:
        if col_name in joined_data.columns:
            technology_col = col_name
            break
    
    if technology_col:
        print(f"Using technology column: {technology_col}")
        unique_techs = joined_data[technology_col].value_counts()
        print(f"Available technologies: {unique_techs}")
        
        # Filter for renewable energy (Solar and Wind) - handle different naming conventions
        if technology_col == 'tech_type':
            # Handle PV (Solar) and WT (Wind) codes
            renewable_mask = joined_data[technology_col].isin(['PV', 'WT'])
        else:
            # Handle full technology names
            renewable_mask = joined_data[technology_col].str.contains('Solar|Wind|PV|WT', case=False, na=False)
        
        filtered_data = joined_data[renewable_mask]
        print(f"After filtering for renewable energy: {len(filtered_data)} projects")
    else:
        print("  No technology column found, using all data")
        filtered_data = joined_data
    
    return filtered_data

def plot_mention_scores_by_year_corrected(filtered_data):
    """
    Create the exact mention_scores_by_year.png plot from the notebook
    """
    print(" Creating corrected mention_scores_by_year plot...")
    
    # Check for required columns
    required_cols = ['op_year']
    opp_score_col = None
    support_score_col = None
    
    # Find opposition and support score columns
    for col in filtered_data.columns:
        if 'mention_opp' in col.lower():
            opp_score_col = col
        if 'mention_support' in col.lower():
            support_score_col = col
    
    if not opp_score_col:
        print(" No opposition score column found")
        return
    if not support_score_col:
        print(" No support score column found") 
        return
    if 'op_year' not in filtered_data.columns:
        print(" No op_year column found")
        return
    
    print(f"Using opposition column: {opp_score_col}")
    print(f"Using support column: {support_score_col}")
    
    # Filter to reasonable years
    year_data = filtered_data[(filtered_data['op_year'] >= 2000) & (filtered_data['op_year'] <= 2024)]
    
    if len(year_data) == 0:
        print(" No data in reasonable year range")
        return
    
    # Count the number of plants with opposition scores by year
    opp_score_counts = year_data.groupby('op_year')[opp_score_col].value_counts().unstack(fill_value=0)
    
    # Ensure we have both 0 and 1 columns
    if 1 not in opp_score_counts.columns:
        opp_score_counts[1] = 0
    if 0 not in opp_score_counts.columns:
        opp_score_counts[0] = 0
        
    opp_score_1 = opp_score_counts[1]
    opp_score_0 = opp_score_counts[0]
    
    # Count the number of plants with support scores by year
    support_score_counts = year_data.groupby('op_year')[support_score_col].value_counts().unstack(fill_value=0)
    
    # Ensure we have both 0 and 1 columns
    if 1 not in support_score_counts.columns:
        support_score_counts[1] = 0
    if 0 not in support_score_counts.columns:
        support_score_counts[0] = 0
        
    support_score_1 = support_score_counts[1]
    support_score_0 = support_score_counts[0]
    
    # Create a figure with two subplots (EXACT replica of notebook)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Create an index for each tick position
    ind = np.arange(len(opp_score_1))
    
    # Plotting mention_opp_score as stacked bars (EXACT colors from notebook)
    ax1.bar(ind, opp_score_1, label='Opposition', color='red')
    ax1.bar(ind, opp_score_0, bottom=opp_score_1, label='No Opposition', color='gray')
    
    # Plotting mention_support_score as stacked bars (EXACT colors from notebook)
    ax2.bar(ind, support_score_1, label='Support', color='blue')
    ax2.bar(ind, support_score_0, bottom=support_score_1, label='No Support', color='gray')
    
    # Set the labels and titles (EXACT text from notebook)
    ax1.set_xlabel('Operational Year')
    ax1.set_ylabel('Number of Plants')
    ax2.set_xlabel('Operational Year')
    
    # Set a common title for both subplots (EXACT title from notebook)
    fig.suptitle('Opposition and Support for Projects Over Time')
    
    # Set x-ticks to only every 5 years to avoid crowding (EXACT from notebook)
    ax1.set_xticks(ind[::5])
    ax1.set_xticklabels(opp_score_1.index[::5])
    ax2.set_xticks(ind[::5])
    ax2.set_xticklabels(support_score_1.index[::5])
    
    ax1.legend()
    ax2.legend()
    
    # Grid and layout adjustment (EXACT from notebook)
    ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to fit the common title
    
    # Save the figure
    save_figure_corrected("mention_scores_by_year.png")
    plt.show()
    plt.close()

def plot_capacity_distribution_opp_status_corrected(filtered_data):
    """
    Create the exact capacity_distribution_opp_status.png plot from the notebook
    """
    print(" Creating corrected capacity_distribution_opp_status plot...")
    
    # Find required columns
    capacity_col = None
    opp_score_col = None
    
    for col in filtered_data.columns:
        if col.lower() == 'capacity':  # Use exact capacity column, not capacity_bin
            capacity_col = col
        if 'mention_opp' in col.lower():
            opp_score_col = col
    
    if not capacity_col:
        print(" No capacity column found")
        return
    if not opp_score_col:
        print(" No opposition score column found")
        return
    
    print(f"Using capacity column: {capacity_col}")
    print(f"Using opposition column: {opp_score_col}")
    
    # Create opposition status categories (EXACT from notebook)
    opposition_data = filtered_data[filtered_data[opp_score_col] == 1]
    no_opposition_data = filtered_data[filtered_data[opp_score_col] == 0]
    
    print(f"Opposition projects: {len(opposition_data)}")
    print(f"No opposition projects: {len(no_opposition_data)}")
    
    if len(opposition_data) == 0 or len(no_opposition_data) == 0:
        print(" Insufficient data for comparison")
        return
    
    # Create histogram for capacity distribution by opposition status (EXACT from notebook)
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each group (EXACT parameters from notebook)
    plt.hist([no_opposition_data[capacity_col].dropna(), opposition_data[capacity_col].dropna()], 
             bins=50, density=True, alpha=0.7, 
             label=['No Opposition', 'Opposition'], 
             color=['gray', 'red'])
    
    # EXACT labels and formatting from notebook
    plt.title('Distribution of Capacity for Projects by Opposition Status')
    plt.xlabel('Capacity (MW) in Log Scale')
    plt.ylabel('Relative Frequency of Projects')
    plt.xscale('log')
    plt.xticks([1, 50, 100, 500, 1000], ['1', '50', '100', '500', '1000'])
    plt.legend()
    plt.grid(False)
    
    plt.tight_layout()
    
    # Save the figure
    save_figure_corrected("capacity_distribution_opp_status.png")
    plt.show()
    plt.close()

def main():
    """Main function to generate corrected plots"""
    print("🔧 GENERATING CORRECTED PLOTS FROM ORIGINAL NOTEBOOK CODE")
    print("=" * 60)
    
    # Load and filter data
    filtered_data = load_and_filter_data()
    if filtered_data is None:
        print(" Cannot proceed without data")
        return
    
    print("\n Generating corrected visualizations...")
    
    # Generate the two specific corrected plots
    plot_mention_scores_by_year_corrected(filtered_data)
    plot_capacity_distribution_opp_status_corrected(filtered_data)
    
    print("\n CORRECTED PLOTS GENERATION COMPLETE!")
    print(f" Corrected plots saved to: {TEST_VIZ_DIR}")
    
    # List generated files
    if TEST_VIZ_DIR.exists():
        corrected_files = list(TEST_VIZ_DIR.glob("*"))
        print(f" Generated {len(corrected_files)} corrected files:")
        for file in sorted(corrected_files):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()