#!/usr/bin/env python3
"""
Publication-Ready Plot Generator for Renewable Energy Dispute Analysis
Generates specific plots: mention_scores_by_year.png and capacity_distribution_opp_status.png

This script creates publication-ready versions of key visualizations using the specified
color palette and formatting requirements for A4 portrait page layout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from pathlib import Path
import sys

# Import our custom styling
from config.paper_style import (
    PaperColors, PaperFormat, setup_paper_style, save_paper_figure
)

# Import path configurations
from config.config import Paths, get_data_path

def load_analysis_dataset():
    """Load the main analysis dataset"""
    print(" Loading analysis dataset...")
    
    try:
        # Try loading dataset with relevance scores first
        relevance_path = get_data_path("analysis_with_relevance.pkl")
        if relevance_path.exists():
            with open(relevance_path, "rb") as f:
                joined_data = pickle.load(f)
            print(f" Loaded dataset with relevance: {len(joined_data)} projects")
        else:
            # Fall back to main dataset
            main_path = get_data_path("complete_analysis_dataset.pkl")
            if not main_path.exists():
                # Try alternative naming
                main_path = get_data_path("dataset_analysis_allscores.pkl")
            
            with open(main_path, "rb") as f:
                joined_data = pickle.load(f)
            print(f" Loaded main dataset: {len(joined_data)} projects")
            
        return joined_data
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None

def plot_mention_scores_by_year_paper(joined_data, output_folder):
    """Generate publication-ready mention scores by year plot - side by side stacked bars"""
    print(" Creating publication-ready time series plot...")
    
    if 'op_year' not in joined_data.columns:
        print(" Operation year data not available")
        return
    
    # Filter to reasonable years (2000-2024)
    filtered_data = joined_data[joined_data['op_year'] >= 2000]
    
    if len(filtered_data) == 0:
        print(" No data in reasonable year range")
        return
    
    # Check column names - could be mention_opp_score or mention_opp
    opp_col = 'mention_opp_score' if 'mention_opp_score' in filtered_data.columns else 'mention_opp'
    support_col = 'mention_support_score' if 'mention_support_score' in filtered_data.columns else 'mention_support'
    
    if opp_col not in filtered_data.columns or support_col not in filtered_data.columns:
        print(f" Required columns not found. Available: {list(filtered_data.columns)}")
        return
    
    # Count opposition by year
    opp_score_counts = filtered_data.groupby('op_year')[opp_col].value_counts().unstack(fill_value=0)
    opp_score_1 = opp_score_counts.get(1, pd.Series(index=opp_score_counts.index, data=0))
    opp_score_0 = opp_score_counts.get(0, pd.Series(index=opp_score_counts.index, data=0))
    
    # Count support by year  
    support_score_counts = filtered_data.groupby('op_year')[support_col].value_counts().unstack(fill_value=0)
    support_score_1 = support_score_counts.get(1, pd.Series(index=support_score_counts.index, data=0))
    support_score_0 = support_score_counts.get(0, pd.Series(index=support_score_counts.index, data=0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PaperFormat.FIG_WIDTH_DOUBLE, PaperFormat.FIG_HEIGHT_STANDARD), sharey=True)
    
    # Create index for tick positions
    ind = np.arange(len(opp_score_1))
    
    # Plot opposition stacked bars
    ax1.bar(ind, opp_score_1, label='Opposition', color=PaperColors.OPPOSITION_COLOR)
    ax1.bar(ind, opp_score_0, bottom=opp_score_1, label='No Opposition', color=PaperColors.NEUTRAL_COLOR)
    
    # Plot support stacked bars
    ax2.bar(ind, support_score_1, label='Support', color=PaperColors.SUPPORT_COLOR)
    ax2.bar(ind, support_score_0, bottom=support_score_1, label='No Support', color=PaperColors.NEUTRAL_COLOR)
    
    # Labels and formatting
    ax1.set_xlabel('Operational Year', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
    ax1.set_ylabel('Number of Plants', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
    ax2.set_xlabel('Operational Year', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
    
    # Common title
    fig.suptitle('Opposition and Support for Projects Over Time', fontsize=PaperFormat.FONT_SIZE_LARGE)
    
    # Set x-ticks every 5 years to avoid crowding
    ax1.set_xticks(ind[::5])
    ax1.set_xticklabels(opp_score_1.index[::5])
    ax2.set_xticks(ind[::5])  
    ax2.set_xticklabels(support_score_1.index[::5])
    
    # Legends and remove grids
    ax1.legend()
    ax2.legend()
    ax1.grid(False)
    ax2.grid(False)
    
    # Layout adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    save_paper_figure("mention_scores_by_year.png", output_folder, fig=fig)
    plt.show()
    
    print(f" Generated side-by-side stacked bar plot with {len(filtered_data)} projects")

def plot_capacity_distribution_opp_status_paper(joined_data, output_folder):
    """Generate publication-ready capacity distribution by opposition status plot"""
    print(" Creating publication-ready capacity distribution plot...")
    
    # Check column names - could be mention_opp_score or mention_opp
    opp_col = 'mention_opp_score' if 'mention_opp_score' in joined_data.columns else 'mention_opp'
    
    if 'capacity' not in joined_data.columns or opp_col not in joined_data.columns:
        print(f" Required data not available. Have: {list(joined_data.columns)}")
        return
    
    # Prepare data
    plot_data = joined_data[['capacity', opp_col]].dropna()
    plot_data = plot_data[plot_data['capacity'] > 0]  # Remove zero/negative capacities
    plot_data['Opposition Status'] = plot_data[opp_col].map({
        0: 'No Opposition Mentioned', 
        1: 'Opposition Mentioned'
    })
    
    if len(plot_data) == 0:
        print(" No valid data for capacity distribution plot")
        return
    
    # Set up the plot
    plt.figure(figsize=(PaperFormat.FIG_WIDTH_SINGLE, PaperFormat.FIG_HEIGHT_TALL))
    
    # Create box plot with custom colors
    box_colors = [PaperColors.SUPPORT_COLOR, PaperColors.OPPOSITION_COLOR]
    
    # Use seaborn for cleaner box plots
    ax = sns.boxplot(data=plot_data, x='Opposition Status', y='capacity', 
                     palette=box_colors)
    
    # Set log scale for capacity
    plt.yscale('log')
    
    # Formatting
    plt.xlabel('Opposition Status', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
    plt.ylabel('Capacity (MW)', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
    plt.title('Capacity Distribution by\nOpposition Status', 
              fontsize=PaperFormat.FONT_SIZE_LARGE)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha='center')
    
    # Remove grid for clean paper look
    plt.grid(False)
    ax.grid(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    save_paper_figure("capacity_distribution_opp_status.png", output_folder)
    plt.show()
    
    # Print some summary statistics
    no_opp = plot_data[plot_data[opp_col] == 0]['capacity']
    with_opp = plot_data[plot_data[opp_col] == 1]['capacity']
    
    print(f" Generated capacity distribution plot:")
    print(f"   • Projects without opposition: {len(no_opp)} (median: {no_opp.median():.1f} MW)")
    print(f"   • Projects with opposition: {len(with_opp)} (median: {with_opp.median():.1f} MW)")

def create_supplementary_plots(joined_data, output_folder):
    """Create additional publication-ready plots as bonus"""
    print(" Creating supplementary publication plots...")
    
    # Technology comparison if data available
    if 'technology' in joined_data.columns:
        plot_tech_comparison(joined_data, output_folder)
    
    # Opposition variables summary
    plot_opposition_summary(joined_data, output_folder)

def plot_tech_comparison(joined_data, output_folder):
    """Plot comparison between solar and wind projects"""
    # Filter for solar and wind projects
    solar_data = joined_data[joined_data['technology'].str.contains('Solar', case=False, na=False)]
    wind_data = joined_data[joined_data['technology'].str.contains('Wind', case=False, na=False)]
    
    if len(solar_data) > 0 and len(wind_data) > 0:
        plt.figure(figsize=(PaperFormat.FIG_WIDTH_DOUBLE, PaperFormat.FIG_HEIGHT_STANDARD))
        
        # Calculate opposition rates
        solar_opp_rate = solar_data.get('mention_opp', pd.Series()).mean()
        wind_opp_rate = wind_data.get('mention_opp', pd.Series()).mean()
        
        # Bar plot
        technologies = ['Solar', 'Wind']
        opp_rates = [solar_opp_rate, wind_opp_rate]
        colors = [PaperColors.SOLAR_COLOR, PaperColors.WIND_COLOR]
        
        bars = plt.bar(technologies, opp_rates, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, opp_rates):
            if not pd.isna(val):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', 
                        fontsize=PaperFormat.FONT_SIZE_SMALL)
        
        plt.ylabel('Opposition Rate', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
        plt.title('Opposition Rates by Technology Type', 
                  fontsize=PaperFormat.FIG_SIZE_LARGE)
        plt.ylim(0, max(opp_rates) * 1.2 if not any(pd.isna(opp_rates)) else 1)
        
        save_paper_figure("technology_comparison.png", output_folder)
        plt.show()

def plot_opposition_summary(joined_data, output_folder):
    """Plot summary of different opposition types"""
    # Opposition variables to analyze
    opp_vars = ['mention_opp', 'physical_opp', 'legal_opp', 'environmental_opp', 'policy_opp']
    available_vars = [var for var in opp_vars if var in joined_data.columns]
    
    if len(available_vars) > 0:
        plt.figure(figsize=(PaperFormat.FIG_WIDTH_DOUBLE, PaperFormat.FIG_HEIGHT_STANDARD))
        
        # Calculate frequencies
        var_counts = joined_data[available_vars].sum().sort_values(ascending=False)
        
        # Create bar plot
        bars = plt.bar(range(len(var_counts)), var_counts.values, 
                      color=PaperColors.PRIMARY_PALETTE[:len(var_counts)],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Formatting
        plt.xlabel('Opposition Type', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
        plt.ylabel('Number of Projects', fontsize=PaperFormat.FONT_SIZE_MEDIUM)
        plt.title('Frequency of Opposition Types', fontsize=PaperFormat.FONT_SIZE_LARGE)
        
        # Clean up variable names for labels
        clean_labels = [var.replace('_', ' ').replace('opp', '').strip().title() 
                       for var in var_counts.index]
        plt.xticks(range(len(var_counts)), clean_labels, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, var_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(val)), ha='center', va='bottom',
                    fontsize=PaperFormat.FONT_SIZE_SMALL)
        
        plt.tight_layout()
        save_paper_figure("opposition_types_summary.png", output_folder)
        plt.show()

def main():
    """Main function to generate publication-ready plots"""
    print("🚀 Starting Publication Plot Generator")
    print("=" * 60)
    
    # Set up publication styling
    setup_paper_style()
    print(" Applied publication styling")
    
    # Create output directory
    output_folder = Path("viz/paper-ready")
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f" Output folder: {output_folder}")
    
    # Load data
    joined_data = load_analysis_dataset()
    if joined_data is None:
        print(" Cannot proceed without dataset")
        return
    
    print(f" Loaded dataset with {len(joined_data)} projects")
    
    # Generate the two main plots
    print("\n Generating main publication plots...")
    plot_mention_scores_by_year_paper(joined_data, output_folder)
    plot_capacity_distribution_opp_status_paper(joined_data, output_folder)
    
    # Generate supplementary plots
    print("\n Generating supplementary plots...")
    create_supplementary_plots(joined_data, output_folder)
    
    print("\n PUBLICATION PLOT GENERATION COMPLETE!")
    print("=" * 60)
    print(f" All plots saved to: {output_folder.absolute()}")
    print(" Format: PNG at 300 DPI with Arial 8pt font")
    print("📏 Sized for A4 portrait page layout")

if __name__ == "__main__":
    main()