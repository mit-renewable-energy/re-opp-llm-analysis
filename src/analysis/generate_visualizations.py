#!/usr/bin/env python3
"""
Renewable Energy Project Dispute Characterization - Visualization Generator
Converted from plots.ipynb to a replicable Python script

This script generates all publication-ready visualizations for the renewable energy
dispute characterization project. It maintains all functionality from the original
notebook while using configurable paths and settings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import folium
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from wordcloud import WordCloud
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import configuration
"from config.config import" (
    Paths, DataFiles, VizConfig, setup_plot_style, save_figure,
    get_data_path, get_viz_path, OPPOSITION_VARIABLES
)

def initialize_plotting():
    """Initialize matplotlib and seaborn settings"""
    print(" Initializing plotting environment...")
    setup_plot_style()
    
    # Create visualizations directory if it doesn't exist
    Paths.VISUALIZATIONS.mkdir(parents=True, exist_ok=True)
    
    print(f" Plots will be saved to: {Paths.VISUALIZATIONS}")

def load_main_dataset():
    """Load the main analysis dataset"""
    print(" Loading main analysis dataset...")
    
    try:
        # Try loading dataset with relevance scores first
        relevance_path = get_data_path("data/final/analysis_with_relevance.pkl")
        if relevance_path.exists():
            with open(relevance_path, "rb") as f:
                joined_data = pickle.load(f)
            print(f" Loaded dataset with relevance: {len(joined_data)} projects")
        else:
            # Fall back to main dataset and compute relevance
            main_path = get_data_path("dataset_analysis_allscores.pkl")
            if not main_path.exists():
                main_path = get_data_path("data/final/complete_analysis_dataset.pkl")
            
            with open(main_path, "rb") as f:
                joined_data = pickle.load(f)
            print(f" Loaded main dataset: {len(joined_data)} projects")
            
            # Compute average relevance scores
            joined_data = compute_avg_relevance(joined_data)
            
        return joined_data
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None

def compute_avg_relevance(joined_data):
    """Compute average relevance scores for projects"""
    print(" Computing average relevance scores...")
    
    def calculate_avg_score(plant_code):
        path = Paths.RESULTS / "article_relevance" / f"{plant_code}.json"
        
        if not path.exists():
            return None
            
        with open(path, "r") as f:
            relevance_data = json.load(f)
            
        try:
            if relevance_data == []:
                return None
            scores = [item['grade'] for item in relevance_data.get('scores_and_justifications', [])]
        except:
            print(f" Error processing plant_code: {plant_code}")
            return None
            
        if scores:
            return sum(scores) / len(scores)
        else:
            return None
    
    joined_data['avg_relevance_score'] = joined_data['plant_code'].apply(calculate_avg_score)
    joined_data['avg_relevance_score'] = pd.to_numeric(joined_data['avg_relevance_score'], errors='coerce')
    
    # Save dataset with relevance for future use
    output_path = get_data_path("data/final/analysis_with_relevance.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(joined_data, f)
    
    print(f" Computed relevance scores for {joined_data['avg_relevance_score'].notna().sum()} projects")
    return joined_data

def load_coordinate_data():
    """Load project coordinate data for mapping"""
    print("  Loading coordinate data...")
    
    try:
        coord_path = get_data_path("output.csv")
        if coord_path.exists():
            df_with_coords = pd.read_csv(coord_path)
            print(f" Loaded coordinates for {len(df_with_coords)} projects")
            return df_with_coords
        else:
            print(" output.csv not found - maps will not be generated")
            return None
    except Exception as e:
        print(f" Error loading coordinate data: {e}")
        return None

def load_demographic_data():
    """Load demographic and geographic data"""
    print("🏛️  Loading demographic data...")
    
    try:
        # Load state boundaries
        state_path = Paths.DEMOGRAPHIC_DATA / "cb_2023_us_state_20m.zip"
        if state_path.exists():
            states_gdf = gpd.read_file(state_path)
            print(f" Loaded state boundaries: {len(states_gdf)} states")
        else:
            print("  State boundary file not found")
            states_gdf = None
            
        # Load demographic shapefiles
        demo_path = Paths.DEMOGRAPHIC_DATA / "1.0-shapefile-codebook" / "usa" / "usa.pkl"
        if demo_path.exists():
            with open(demo_path, "rb") as f:
                demographic_data = pickle.load(f)
            print(" Loaded demographic data")
        else:
            print("  Demographic data not found")
            demographic_data = None
            
        return states_gdf, demographic_data
    except Exception as e:
        print(f" Error loading demographic data: {e}")
        return None, None

def plot_relevance_distribution(joined_data):
    """Plot distribution of average relevance scores"""
    print(" Creating relevance score distribution plot...")
    
    plt.figure(figsize=VizConfig.FIG_SIZE_STANDARD)
    
    relevance_data = joined_data['avg_relevance_score'].dropna()
    if len(relevance_data) == 0:
        print(" No relevance data available")
        return
    
    # Create histogram with KDE
    ax = sns.histplot(relevance_data, bins=20, kde=True, alpha=0.7)
    ax.set_title('Distribution of Average Relevance Scores', 
                 fontsize=VizConfig.FONT_SIZES["title"])
    ax.set_xlabel('Average Relevance Score', 
                  fontsize=VizConfig.FONT_SIZES["label"])
    ax.set_ylabel('Frequency', 
                  fontsize=VizConfig.FONT_SIZES["label"])
    
    save_figure("avg_relevance_score_distribution.png")
    plt.show()

def plot_joint_distributions(joined_data):
    """Create joint distribution plots"""
    print(" Creating joint distribution plots...")
    
    # Capacity vs Relevance
    if 'capacity' in joined_data.columns and 'avg_relevance_score' in joined_data.columns:
        # Prepare data
        plot_data = joined_data[['capacity', 'avg_relevance_score']].dropna()
        
        if len(plot_data) > 0:
            # Calculate correlation
            corr_coef, p_value = pearsonr(plot_data['capacity'], plot_data['avg_relevance_score'])
            
            # Create joint plot
            g = sns.jointplot(data=plot_data, x='avg_relevance_score', y='capacity', 
                             kind='scatter', height=8, ratio=5, space=0.2, alpha=0.5)
            g.ax_joint.set_yscale('log')
            g.ax_joint.set_title(f'Capacity vs Average Relevance Score\n(r = {corr_coef:.3f}, p = {p_value:.3f})',
                               fontsize=VizConfig.FONT_SIZES["title"])
            
            save_figure("capacity_relevance_joint_dist.png", fig=g.fig)
            plt.show()
    
    # Operation Year vs Relevance
    if 'op_year' in joined_data.columns and 'avg_relevance_score' in joined_data.columns:
        plot_data = joined_data[['op_year', 'avg_relevance_score']].dropna()
        
        if len(plot_data) > 0:
            g = sns.jointplot(data=plot_data, x='op_year', y='avg_relevance_score',
                             kind='scatter', height=8, ratio=5, space=0.2, alpha=0.5)
            g.ax_joint.set_title('Operation Year vs Average Relevance Score',
                               fontsize=VizConfig.FONT_SIZES["title"])
            
            save_figure("op_year_relevance_joint_dist.png", fig=g.fig)
            plt.show()

def plot_capacity_distributions(joined_data):
    """Plot capacity-related distributions"""
    print(" Creating capacity distribution plots...")
    
    if 'capacity' not in joined_data.columns:
        print(" Capacity data not available")
        return
    
    # Capacity distribution by opposition status
    if 'mention_opp' in joined_data.columns:
        plt.figure(figsize=VizConfig.FIG_SIZE_STANDARD)
        
        # Create box plot
        plot_data = joined_data[['capacity', 'mention_opp']].dropna()
        plot_data['Opposition Status'] = plot_data['mention_opp'].map({0: 'No Opposition', 1: 'Opposition Mentioned'})
        
        sns.boxplot(data=plot_data, x='Opposition Status', y='capacity')
        plt.yscale('log')
        plt.title('Capacity Distribution by Opposition Status',
                  fontsize=VizConfig.FONT_SIZES["title"])
        plt.ylabel('Capacity (MW)', fontsize=VizConfig.FONT_SIZES["label"])
        
        save_figure("capacity_distribution_opp_status.png")
        plt.show()

def plot_mention_scores_by_capacity(joined_data):
    """Plot mention scores by capacity bins"""
    print(" Creating mention scores by capacity plots...")
    
    if 'capacity' not in joined_data.columns:
        print(" Capacity data not available")
        return
    
    # Create capacity bins
    joined_data_copy = joined_data.copy()
    joined_data_copy = joined_data_copy[joined_data_copy['capacity'] > 0]
    
    # Define capacity bins
    bins = [0, 1, 5, 25, 100, 500, float('inf')]
    labels = ['<1 MW', '1-5 MW', '5-25 MW', '25-100 MW', '100-500 MW', '>500 MW']
    joined_data_copy['capacity_bin'] = pd.cut(joined_data_copy['capacity'], bins=bins, labels=labels)
    
    # Plot mention scores by capacity
    opposition_vars = ['mention_support', 'mention_opp', 'physical_opp']
    
    plt.figure(figsize=VizConfig.FIG_SIZE_LARGE)
    
    bin_stats = joined_data_copy.groupby('capacity_bin')[opposition_vars].agg(['mean', 'count']).round(3)
    
    x_pos = range(len(labels))
    width = 0.25
    
    for i, var in enumerate(opposition_vars):
        plt.bar([x + width * i for x in x_pos], 
                bin_stats[(var, 'mean')], 
                width, 
                label=var.replace('_', ' ').title(),
                alpha=0.8)
    
    plt.xlabel('Capacity Range', fontsize=VizConfig.FONT_SIZES["label"])
    plt.ylabel('Proportion of Projects', fontsize=VizConfig.FONT_SIZES["label"])
    plt.title('Opposition and Support Mentions by Capacity Range',
              fontsize=VizConfig.FONT_SIZES["title"])
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    save_figure("mention_scores_by_capacity.png")
    plt.show()
    
    # Special analysis for >500MW projects
    large_projects = joined_data_copy[joined_data_copy['capacity'] > 500]
    if len(large_projects) > 0:
        plt.figure(figsize=VizConfig.FIG_SIZE_STANDARD)
        
        large_stats = large_projects[opposition_vars].mean()
        colors = [VizConfig.CB91_Blue, VizConfig.CB91_Amber, VizConfig.CB91_Pink]
        
        bars = plt.bar(range(len(opposition_vars)), large_stats, color=colors, alpha=0.8)
        plt.xlabel('Opposition/Support Type', fontsize=VizConfig.FONT_SIZES["label"])
        plt.ylabel('Proportion of Projects', fontsize=VizConfig.FONT_SIZES["label"])
        plt.title(f'Opposition and Support in Large Projects (>500 MW)\nn = {len(large_projects)}',
                  fontsize=VizConfig.FONT_SIZES["title"])
        plt.xticks(range(len(opposition_vars)), 
                   [var.replace('_', ' ').title() for var in opposition_vars])
        
        # Add value labels on bars
        for bar, val in zip(bars, large_stats):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom')
        
        save_figure("mention_scores_by_capacity_500MWplus.png")
        plt.show()

def plot_time_series(joined_data):
    """Plot mention scores over time"""
    print(" Creating time series plots...")
    
    if 'op_year' not in joined_data.columns:
        print(" Operation year data not available")
        return
    
    # Filter to reasonable years
    time_data = joined_data[(joined_data['op_year'] >= 2000) & (joined_data['op_year'] <= 2024)]
    
    if len(time_data) == 0:
        print(" No data in reasonable year range")
        return
    
    # Plot mention scores by year
    plt.figure(figsize=VizConfig.FIG_SIZE_LARGE)
    
    yearly_stats = time_data.groupby('op_year')[['mention_opp', 'mention_support', 'physical_opp']].mean()
    
    for var in ['mention_opp', 'mention_support', 'physical_opp']:
        if var in yearly_stats.columns:
            plt.plot(yearly_stats.index, yearly_stats[var], 
                    marker='o', linewidth=2, markersize=4,
                    label=var.replace('_', ' ').title())
    
    plt.xlabel('Operation Year', fontsize=VizConfig.FONT_SIZES["label"])
    plt.ylabel('Proportion of Projects', fontsize=VizConfig.FONT_SIZES["label"])
    plt.title('Opposition and Support Mentions Over Time',
              fontsize=VizConfig.FONT_SIZES["title"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_figure("mention_scores_by_year.png")
    plt.show()
    
    # Percentage chart over years
    plt.figure(figsize=VizConfig.FIG_SIZE_LARGE)
    
    yearly_counts = time_data.groupby('op_year').size()
    yearly_opp_counts = time_data[time_data['mention_opp'] == 1].groupby('op_year').size()
    yearly_support_counts = time_data[time_data['mention_support'] == 1].groupby('op_year').size()
    
    opp_pct = (yearly_opp_counts / yearly_counts * 100).fillna(0)
    support_pct = (yearly_support_counts / yearly_counts * 100).fillna(0)
    
    plt.bar(opp_pct.index, opp_pct.values, alpha=0.7, 
           label='Opposition Mentioned', color=VizConfig.CB91_Pink)
    plt.bar(support_pct.index, support_pct.values, alpha=0.7,
           label='Support Mentioned', color=VizConfig.CB91_Green)
    
    plt.xlabel('Operation Year', fontsize=VizConfig.FONT_SIZES["label"])
    plt.ylabel('Percentage of Projects', fontsize=VizConfig.FONT_SIZES["label"])
    plt.title('Percentage of Projects with Opposition/Support Mentions by Year',
              fontsize=VizConfig.FONT_SIZES["title"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_figure("mention_opp_support_pct_years.png")
    plt.show()

def create_interactive_maps(joined_data, coord_data, states_gdf=None):
    """Create interactive maps using Folium"""
    print("  Creating interactive maps...")
    
    if coord_data is None:
        print(" No coordinate data available for mapping")
        return
    
    # Merge coordinate data with analysis data
    map_data = coord_data.merge(joined_data[['plant_code', 'mention_opp', 'mention_support']], 
                               on='plant_code', how='inner')
    
    if len(map_data) == 0:
        print(" No matching data for mapping")
        return
    
    # Create maps for opposition and support
    for variable, title in [('mention_opp', 'Opposition'), ('mention_support', 'Support')]:
        print(f"Creating {title} map...")
        
        # Create base map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Add state boundaries if available
        if states_gdf is not None:
            folium.GeoJson(
                states_gdf,
                style_function=lambda feature: {
                    'fillColor': 'lightgray',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
        
        # Add project markers
        for idx, row in map_data.iterrows():
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                color = VizConfig.CB91_Pink if row[variable] == 1 else VizConfig.CB91_Blue
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Plant: {row['plant_code']}<br>{title}: {'Yes' if row[variable] == 1 else 'No'}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Save map
        map_path = get_viz_path(f"map_{variable}_usa.html")
        m.save(str(map_path))
        print(f" Saved {title} map: {map_path}")

def generate_demographic_analysis(joined_data, demographic_data):
    """Generate demographic analysis by technology type"""
    print("👥 Generating demographic analysis...")
    
    if demographic_data is None:
        print(" Demographic data not available")
        return
    
    # Analyze by technology type
    tech_types = ['Solar', 'Wind'] if 'technology' in joined_data.columns else []
    
    for tech in tech_types:
        if 'technology' in joined_data.columns:
            tech_data = joined_data[joined_data['technology'].str.contains(tech, case=False, na=False)]
            
            if len(tech_data) > 0:
                # Calculate demographic statistics
                stats_summary = {
                    'total_projects': len(tech_data),
                    'mean_capacity': tech_data.get('capacity', pd.Series()).mean(),
                    'opposition_rate': tech_data.get('mention_opp', pd.Series()).mean(),
                    'support_rate': tech_data.get('mention_support', pd.Series()).mean()
                }
                
                # Save technology-specific statistics
                stats_df = pd.DataFrame([stats_summary])
                stats_path = get_viz_path(f"{tech.lower()}_dem_stats.csv")
                stats_df.to_csv(stats_path, index=False)
                print(f" Saved {tech} demographic stats: {stats_path}")

def create_wordcloud(joined_data):
    """Create word cloud from narrative text"""
    print("☁️  Creating narrative word cloud...")
    
    # Check if we have narrative data from results
    narratives = []
    results_dir = Paths.RESULTS / "scores"
    
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'all_scores_and_sources' in data:
                        for score_data in data['all_scores_and_sources']:
                            if 'narrative' in score_data:
                                narrative = score_data['narrative']
                                if narrative and narrative != "No relevant info found.":
                                    narratives.append(narrative)
            except:
                continue
    
    if narratives:
        # Combine all narratives
        all_text = ' '.join(narratives)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_text)
        
        plt.figure(figsize=VizConfig.FIG_SIZE_LARGE)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Project Narratives',
                  fontsize=VizConfig.FONT_SIZES["title"])
        
        save_figure("wordcloud_narrative.png")
        plt.show()
        
        print(f" Created word cloud from {len(narratives)} narratives")
    else:
        print(" No narrative data found for word cloud")

def generate_summary_statistics(joined_data):
    """Generate comprehensive summary statistics"""
    print(" Generating summary statistics...")
    
    # Overall statistics
    overall_stats = {
        'total_projects': len(joined_data),
        'projects_with_coordinates': len(joined_data.dropna(subset=['latitude', 'longitude'])) if 'latitude' in joined_data.columns else 0,
        'projects_with_relevance_scores': joined_data['avg_relevance_score'].notna().sum() if 'avg_relevance_score' in joined_data.columns else 0,
        'mean_capacity': joined_data.get('capacity', pd.Series()).mean(),
        'median_capacity': joined_data.get('capacity', pd.Series()).median(),
        'projects_with_opposition': joined_data.get('mention_opp', pd.Series()).sum(),
        'projects_with_support': joined_data.get('mention_support', pd.Series()).sum(),
        'opposition_rate': joined_data.get('mention_opp', pd.Series()).mean(),
        'support_rate': joined_data.get('mention_support', pd.Series()).mean()
    }
    
    # Save overall statistics
    stats_df = pd.DataFrame([overall_stats])
    stats_path = get_viz_path("all_projects_summary.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f" Saved overall statistics: {stats_path}")
    
    # Opposition variable frequencies
    if all(var in joined_data.columns for var in OPPOSITION_VARIABLES):
        var_stats = joined_data[OPPOSITION_VARIABLES].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(var_stats)), var_stats.values, 
                      color=VizConfig.COLOR_LIST[:len(var_stats)])
        
        plt.xlabel('Opposition/Support Variables', fontsize=VizConfig.FONT_SIZES["label"])
        plt.ylabel('Number of Projects', fontsize=VizConfig.FONT_SIZES["label"])
        plt.title('Frequency of Opposition and Support Variables',
                  fontsize=VizConfig.FONT_SIZES["title"])
        plt.xticks(range(len(var_stats)), 
                   [var.replace('_', ' ').title() for var in var_stats.index], 
                   rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, var_stats.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(val)), ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure("public_perception_vars.png")
        plt.show()
    
    return overall_stats

def run_statistical_analysis(joined_data):
    """Run statistical analysis and save results"""
    print("🔬 Running statistical analysis...")
    
    if not all(var in joined_data.columns for var in ['capacity', 'mention_opp']):
        print(" Required variables not available for statistical analysis")
        return
    
    # Prepare data for modeling
    model_data = joined_data[['capacity', 'mention_opp'] + 
                            [var for var in OPPOSITION_VARIABLES if var in joined_data.columns]].dropna()
    
    if len(model_data) < 50:
        print(" Insufficient data for statistical modeling")
        return
    
    # Basic correlation analysis
    correlations = model_data.corr()['mention_opp'].sort_values(key=abs, ascending=False)
    
    # Save correlation results
    corr_path = get_viz_path("correlation_analysis.csv")
    correlations.to_csv(corr_path)
    print(f" Saved correlation analysis: {corr_path}")
    
    # Create a simple summary (placeholder for more complex modeling)
    summary_text = f"""
Statistical Analysis Summary
===========================
Sample Size: {len(model_data)}
Opposition Rate: {model_data['mention_opp'].mean():.3f}
Top Correlates with Opposition:
{correlations.head().to_string()}
"""
    
    summary_path = get_viz_path("statistical_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f" Saved statistical summary: {summary_path}")

def main():
    """Main function to generate all visualizations"""
    print("🚀 Starting Renewable Energy Dispute Visualization Generator")
    print("=" * 70)
    
    # Initialize
    initialize_plotting()
    
    # Load data
    joined_data = load_main_dataset()
    if joined_data is None:
        print(" Cannot proceed without main dataset")
        return
    
    coord_data = load_coordinate_data()
    states_gdf, demographic_data = load_demographic_data()
    
    # Generate visualizations
    print("\n Generating Distribution Plots...")
    plot_relevance_distribution(joined_data)
    plot_joint_distributions(joined_data)
    
    print("\n Generating Capacity Analysis...")
    plot_capacity_distributions(joined_data)
    plot_mention_scores_by_capacity(joined_data)
    
    print("\n Generating Time Series Analysis...")
    plot_time_series(joined_data)
    
    print("\n  Generating Maps...")
    create_interactive_maps(joined_data, coord_data, states_gdf)
    
    print("\n👥 Generating Demographic Analysis...")
    generate_demographic_analysis(joined_data, demographic_data)
    
    print("\n☁️  Generating Text Analysis...")
    create_wordcloud(joined_data)
    
    print("\n Generating Summary Statistics...")
    stats = generate_summary_statistics(joined_data)
    
    print("\n🔬 Running Statistical Analysis...")
    run_statistical_analysis(joined_data)
    
    print("\n VISUALIZATION GENERATION COMPLETE!")
    print("=" * 70)
    print(f" All outputs saved to: {Paths.VISUALIZATIONS}")
    print(f" Generated plots for {len(joined_data)} projects")
    print(f" Using colorblind-friendly palette with {VizConfig.FIGURE_FORMAT.upper()} format")

if __name__ == "__main__":
    main()