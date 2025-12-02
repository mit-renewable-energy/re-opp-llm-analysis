#!/usr/bin/env python3
"""
Focused Publication Plot Generator - Just the two main plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid timeout
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Publication colors
class PaperColors:
    CORNFLOWER = "#8ECBE6"
    EASTERN_BLUE = "#229EBC"
    GREEN_VOGUE = "#023047" 
    BLUMINE = "#1E5571"
    SELECTIVE_YELLOW = "#FFB706"
    FLUSH_ORANGE = "#FC8500"
    CORNFLOWER_BLUE = "#5E8CF1"
    WHITE = "#FFFFFF"
    SILVER = "#C0C0C0"
    GALLERY = "#EBEBEB"
    THUNDERBIRD = "#C1131F"
    RED_BERRY = "#980100"
    SERENADE = "#FFF4E5"
    
    # Special colors
    SOLAR_COLOR = FLUSH_ORANGE
    WIND_COLOR = BLUMINE
    OPPOSITION_COLOR = THUNDERBIRD
    SUPPORT_COLOR = EASTERN_BLUE
    NEUTRAL_COLOR = SILVER

# Set up publication style
def setup_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.linewidth': 0.5,
        'axes.edgecolor': 'black',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.0
    })

def load_data():
    """Load the analysis dataset"""
    print(" Loading dataset...")
    
    try:
        with open("data/final/complete_analysis_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        print(f" Loaded {len(data)} projects")
        return data
    except Exception as e:
        print(f" Error loading data: {e}")
        return None

def plot_mention_scores_by_year(data, output_folder):
    """Generate side-by-side stacked bar plot"""
    print(" Creating mention scores by year plot...")
    
    if data is None:
        return
    
    # Filter data from 2000 onwards
    filtered_data = data[data['op_year'] >= 2000]
    print(f"  Filtered to {len(filtered_data)} projects from 2000+")
    
    # Check which columns we have
    opp_col = None
    support_col = None
    
    for col in ['mention_opp_score', 'mention_opp']:
        if col in filtered_data.columns:
            opp_col = col
            break
            
    for col in ['mention_support_score', 'mention_support']:
        if col in filtered_data.columns:
            support_col = col
            break
    
    if opp_col is None or support_col is None:
        print(f"   Required columns not found. Available: {list(filtered_data.columns)}")
        return
    
    print(f"  Using opposition column: {opp_col}")
    print(f"  Using support column: {support_col}")
    
    # Count by year
    opp_counts = filtered_data.groupby('op_year')[opp_col].value_counts().unstack(fill_value=0)
    support_counts = filtered_data.groupby('op_year')[support_col].value_counts().unstack(fill_value=0)
    
    # Get positive and negative counts
    opp_yes = opp_counts.get(1, pd.Series(index=opp_counts.index, data=0))
    opp_no = opp_counts.get(0, pd.Series(index=opp_counts.index, data=0))
    support_yes = support_counts.get(1, pd.Series(index=support_counts.index, data=0))
    support_no = support_counts.get(0, pd.Series(index=support_counts.index, data=0))
    
    # Create plot with half-page height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.0), sharey=True)
    
    # Bar positions
    years = opp_yes.index
    x_pos = np.arange(len(years))
    
    # Opposition plot
    ax1.bar(x_pos, opp_yes, label='Opposition', color=PaperColors.OPPOSITION_COLOR)
    ax1.bar(x_pos, opp_no, bottom=opp_yes, label='No Opposition', color=PaperColors.NEUTRAL_COLOR)
    
    # Support plot  
    ax2.bar(x_pos, support_yes, label='Support', color=PaperColors.SUPPORT_COLOR)
    ax2.bar(x_pos, support_no, bottom=support_yes, label='No Support', color=PaperColors.NEUTRAL_COLOR)
    
    # Formatting
    ax1.set_xlabel('Operational Year')
    ax1.set_ylabel('Number of Plants')
    ax2.set_xlabel('Operational Year')
    
    fig.suptitle('Opposition and Support for Projects Over Time')
    
    # X-tick labels (every 5 years)
    tick_positions = x_pos[::5]
    tick_labels = years[::5]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    
    # Legends and clean up
    ax1.legend()
    ax2.legend()
    ax1.grid(False)
    ax2.grid(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    output_path = Path(output_folder) / "mention_scores_by_year.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved to {output_path}")

def plot_capacity_distribution(data, output_folder):
    """Generate capacity distribution KDE plot (like Aug20_2025 version)"""
    print(" Creating capacity distribution plot...")
    
    if data is None:
        return
    
    # Check columns
    opp_col = None
    for col in ['mention_opp_score', 'mention_opp']:
        if col in data.columns:
            opp_col = col
            break
    
    if opp_col is None or 'capacity' not in data.columns:
        print(f"   Required columns not found")
        return
    
    # Filter data like in the original script
    opp_data = data[data[opp_col] == 1]
    no_opp_data = data[data[opp_col] == 0]
    
    # Calculate averages
    average_capacity_opp = opp_data['capacity'].mean()
    average_capacity_no_opp = no_opp_data['capacity'].mean()
    
    print(f"  Projects with opposition: {len(opp_data)} (avg: {average_capacity_opp:.1f} MW)")
    print(f"  Projects without opposition: {len(no_opp_data)} (avg: {average_capacity_no_opp:.1f} MW)")
    
    # Create plot with half-page height
    plt.figure(figsize=(7, 4.0))
    
    # KDE plots with updated colors (red for opposition, blue for no opposition)
    sns.kdeplot(opp_data['capacity'], color=PaperColors.OPPOSITION_COLOR, 
                label='With Opposition', linewidth=2)
    sns.kdeplot(no_opp_data['capacity'], color=PaperColors.SUPPORT_COLOR, 
                label='No Opposition', linewidth=2)
    
    # Average lines
    plt.axvline(x=average_capacity_opp, color=PaperColors.OPPOSITION_COLOR, 
                linestyle='--', linewidth=2, 
                label=f'Average Capacity with Opposition: {average_capacity_opp:.1f} MW')
    plt.axvline(x=average_capacity_no_opp, color=PaperColors.SUPPORT_COLOR, 
                linestyle='--', linewidth=2, 
                label=f'Average Capacity without Opposition: {average_capacity_no_opp:.1f} MW')
    
    # Formatting
    plt.title('Distribution of Capacity for Projects by Opposition Status')
    plt.xlabel('Capacity (MW) in Log Scale')
    plt.ylabel('Relative Frequency of Projects')
    plt.xscale('log')
    plt.xticks([1, 50, 100, 500, 1000], ['1', '50', '100', '500', '1000'])
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / "capacity_distribution_opp_status.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved to {output_path}")

def plot_solar_wind_variables(data, output_folder):
    """Generate solar vs wind variables comparison plot"""
    print(" Creating solar vs wind variables comparison...")
    
    if data is None:
        return
    
    # Variable labels dictionary (from plot_test_me.py)
    variable_labels = {
        "mention_support_score": "Support",
        "mention_opp_score": "Opposition", 
        "physical_opp": "Physical",
        "policy_opp": "Policy",
        "legal_opp": "Legal",
        "opinion_opp": "Opinion",
        "environmental_opp": "Environmental",
        "participation_opp": "Participation",
        "tribal_opp": "Tribal",
        "health_opp": "Health",
        "intergov_opp": "Intergovt",
        "property_opp": "Property",
        "compensation": "Compensation",
        "delay": "Delay",
        "co_land_use": "Co-Use"
    }
    
    # Filter data for PV and WT
    if 'tech_type' not in data.columns:
        print("   tech_type column not found")
        return
        
    pv_data = data[data['tech_type'] == 'PV']
    wt_data = data[data['tech_type'] == 'WT']
    
    print(f"  Solar projects: {len(pv_data)}")
    print(f"  Wind projects: {len(wt_data)}")
    
    if len(pv_data) == 0 or len(wt_data) == 0:
        print("   Insufficient data for comparison")
        return
    
    # Calculate percentages for each variable
    pv_percentages = {}
    wt_percentages = {}
    
    for var, label in variable_labels.items():
        if var in data.columns:
            pv_pct = (pv_data[var] == 1).sum() / len(pv_data) * 100 if len(pv_data) > 0 else 0
            wt_pct = (wt_data[var] == 1).sum() / len(wt_data) * 100 if len(wt_data) > 0 else 0
            pv_percentages[label] = pv_pct
            wt_percentages[label] = wt_pct
    
    # Create DataFrames
    pv_df = pd.DataFrame(list(pv_percentages.items()), columns=['Variable', 'Solar'])
    wt_df = pd.DataFrame(list(wt_percentages.items()), columns=['Variable', 'Wind'])
    merged_df = pd.merge(pv_df, wt_df, on='Variable')
    
    # Create plot with half-page height
    fig, ax = plt.subplots(figsize=(7, 4.0))
    
    # Bar plot with updated colors (orange for solar, blue for wind)
    merged_df.plot(kind='bar', ax=ax, 
                   color={'Solar': PaperColors.SOLAR_COLOR, 'Wind': PaperColors.WIND_COLOR})
    
    # Formatting
    ax.set_xticklabels(merged_df['Variable'], rotation=45, ha='right')
    ax.set_ylabel('Percentage of Projects')
    ax.set_title('Solar vs. Wind Projects by Mention of Variables')
    plt.grid(False)
    ax.grid(False)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / "solar_wind_variables_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved to {output_path}")

def create_styled_map(data, output_folder):
    """Create combined solar/wind map with Alaska and Hawaii insets (non-bold fonts)"""
    print("  Creating styled solar/wind comparison map...")
    
    try:
        import geopandas as gpd
        
        if data is None:
            print("   No data available for map")
            return
            
        variable = 'mention_opp_score'
        
        # Filter for solar and wind projects
        solar_data = data[data['tech_type'] == 'PV']
        wind_data = data[data['tech_type'] == 'WT']
        
        if solar_data.empty or wind_data.empty:
            print("   Missing solar or wind projects")
            return
            
        # Load state boundaries
        state_data_path = Path("data/raw/demographic_data/cb_2023_us_state_20m.zip")
        if not state_data_path.exists():
            print("   State boundary file not found")
            return
            
        state_data = gpd.read_file(f"zip://{state_data_path}")
        
        # Calculate state-level percentages for both technologies
        solar_opp_score = solar_data[solar_data[variable] == 1].groupby('state').size() / solar_data.groupby('state').size() * 100
        solar_opp_score = solar_opp_score.reset_index(name=f'solar_percentage_{variable}_1')
        
        wind_opp_score = wind_data[wind_data[variable] == 1].groupby('state').size() / wind_data.groupby('state').size() * 100
        wind_opp_score = wind_opp_score.reset_index(name=f'wind_percentage_{variable}_1')
        
        # Merge with state geometries
        state_data_solar = state_data.merge(solar_opp_score, left_on='STUSPS', right_on='state', how='left')
        state_data_wind = state_data.merge(wind_opp_score, left_on='STUSPS', right_on='state', how='left')
        
        # Create figure with subplots
        fig, (ax_solar, ax_wind) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Common color range for both maps
        all_values = pd.concat([
            state_data_solar[f'solar_percentage_{variable}_1'].dropna(),
            state_data_wind[f'wind_percentage_{variable}_1'].dropna()
        ])
        vmin = all_values.min()
        vmax = all_values.max()
        
        # ===== SOLAR MAP (TOP) =====
        # Separate regions for solar
        solar_continental = state_data_solar[~state_data_solar['STUSPS'].isin(['AK', 'HI'])]
        solar_alaska = state_data_solar[state_data_solar['STUSPS'] == 'AK']
        solar_hawaii = state_data_solar[state_data_solar['STUSPS'] == 'HI']
        
        # Plot solar continental US
        solar_continental.plot(column=f'solar_percentage_{variable}_1',
                              ax=ax_solar,
                              cmap='OrRd',
                              vmin=vmin,
                              vmax=vmax,
                              edgecolor='white',
                              linewidth=0.5,
                              missing_kwds={'color': '#A0A0A0'})
        
        ax_solar.set_xlim(-125, -66)
        ax_solar.set_ylim(20.5, 49.5)
        ax_solar.set_title('Solar Projects Opposition by State', fontsize=14, fontweight='normal', pad=10)  # Changed from bold
        ax_solar.axis('off')
        
        # Add solar project points
        continental_solar = solar_data[~solar_data['state'].isin(['AK', 'HI'])].dropna(subset=['lat', 'long'])
        opp_solar = continental_solar[continental_solar[variable] == 1]
        no_opp_solar = continental_solar[continental_solar[variable] == 0]
        
        ax_solar.scatter(no_opp_solar['long'], no_opp_solar['lat'], 
                        c='#808080', s=4, alpha=0.6, edgecolors='none')
        ax_solar.scatter(opp_solar['long'], opp_solar['lat'], 
                        c='black', s=6, alpha=0.6, edgecolors='none')
        
        # ===== WIND MAP (BOTTOM) =====
        # Separate regions for wind
        wind_continental = state_data_wind[~state_data_wind['STUSPS'].isin(['AK', 'HI'])]
        wind_alaska = state_data_wind[state_data_wind['STUSPS'] == 'AK']
        wind_hawaii = state_data_wind[state_data_wind['STUSPS'] == 'HI']
        
        # Plot wind continental US
        wind_continental.plot(column=f'wind_percentage_{variable}_1',
                             ax=ax_wind,
                             cmap='OrRd',
                             vmin=vmin,
                             vmax=vmax,
                             edgecolor='white',
                             linewidth=0.5,
                             missing_kwds={'color': '#A0A0A0'})
        
        ax_wind.set_xlim(-125, -66)
        ax_wind.set_ylim(20.5, 49.5)
        ax_wind.set_title('Wind Projects Opposition by State', fontsize=14, fontweight='normal', pad=10)  # Changed from bold
        ax_wind.axis('off')
        
        # Add wind project points
        continental_wind = wind_data[~wind_data['state'].isin(['AK', 'HI'])].dropna(subset=['lat', 'long'])
        opp_wind = continental_wind[continental_wind[variable] == 1]
        no_opp_wind = continental_wind[continental_wind[variable] == 0]
        
        ax_wind.scatter(no_opp_wind['long'], no_opp_wind['lat'], 
                       c='#808080', s=4, alpha=0.6, edgecolors='none')
        ax_wind.scatter(opp_wind['long'], opp_wind['lat'], 
                       c='black', s=6, alpha=0.6, edgecolors='none')
        
        # ===== ADD ALASKA AND HAWAII INSETS =====
        # Inset dimensions
        alaska_width = 0.12
        alaska_height = 0.15
        hawaii_width = 0.08
        hawaii_height = 0.12
        
        # Solar Alaska inset (top subplot)
        alaska_left = -0.05
        alaska_top = 0.75
        ax_solar_alaska = fig.add_axes([alaska_left, alaska_top, alaska_width, alaska_height])
        
        if not solar_alaska.empty:
            solar_alaska.plot(column=f'solar_percentage_{variable}_1',
                             ax=ax_solar_alaska,
                             cmap='OrRd',
                             vmin=vmin,
                             vmax=vmax,
                             edgecolor='gray',
                             linewidth=0.8,
                             missing_kwds={'color': '#A0A0A0', 'edgecolor': 'gray'})
            ax_solar_alaska.set_xlim(-180, -130)
            ax_solar_alaska.set_ylim(54, 72)
            
            # Add Alaska solar project points
            alaska_solar_projects = solar_data[solar_data['state'] == 'AK'].dropna(subset=['lat', 'long'])
            ak_opp_solar = alaska_solar_projects[alaska_solar_projects[variable] == 1]
            ak_no_opp_solar = alaska_solar_projects[alaska_solar_projects[variable] == 0]
            
            ax_solar_alaska.scatter(ak_no_opp_solar['long'], ak_no_opp_solar['lat'], 
                                   c='#808080', s=3, alpha=0.6, edgecolors='none')
            ax_solar_alaska.scatter(ak_opp_solar['long'], ak_opp_solar['lat'], 
                                   c='black', s=4, alpha=0.6, edgecolors='none')
        
        ax_solar_alaska.set_title('Alaska', fontsize=9, pad=2)
        ax_solar_alaska.axis('off')
        
        # Solar Hawaii inset (top subplot)
        hawaii_left = 0.15
        hawaii_top = 0.58
        ax_solar_hawaii = fig.add_axes([hawaii_left, hawaii_top, hawaii_width, hawaii_height])
        
        if not solar_hawaii.empty:
            solar_hawaii.plot(column=f'solar_percentage_{variable}_1',
                             ax=ax_solar_hawaii,
                             cmap='OrRd',
                             vmin=vmin,
                             vmax=vmax,
                             edgecolor='gray',
                             linewidth=0.8,
                             missing_kwds={'color': '#A0A0A0', 'edgecolor': 'gray'})
            
            # Add Hawaii solar project points
            hawaii_solar_projects = solar_data[solar_data['state'] == 'HI'].dropna(subset=['lat', 'long'])
            hi_opp_solar = hawaii_solar_projects[hawaii_solar_projects[variable] == 1]
            hi_no_opp_solar = hawaii_solar_projects[hawaii_solar_projects[variable] == 0]
            
            ax_solar_hawaii.scatter(hi_no_opp_solar['long'], hi_no_opp_solar['lat'], 
                                   c='#808080', s=3, alpha=0.6, edgecolors='none')
            ax_solar_hawaii.scatter(hi_opp_solar['long'], hi_opp_solar['lat'], 
                                   c='black', s=4, alpha=0.6, edgecolors='none')
        
        ax_solar_hawaii.set_title('Hawaii', fontsize=9, pad=2)
        ax_solar_hawaii.axis('off')
        
        # Wind Alaska inset (bottom subplot)
        wind_alaska_left = -0.05
        wind_alaska_top = 0.32
        ax_wind_alaska = fig.add_axes([wind_alaska_left, wind_alaska_top, alaska_width, alaska_height])
        
        if not wind_alaska.empty:
            wind_alaska.plot(column=f'wind_percentage_{variable}_1',
                            ax=ax_wind_alaska,
                            cmap='OrRd',
                            vmin=vmin,
                            vmax=vmax,
                            edgecolor='gray',
                            linewidth=0.8,
                            missing_kwds={'color': '#A0A0A0', 'edgecolor': 'gray'})
            ax_wind_alaska.set_xlim(-180, -130)
            ax_wind_alaska.set_ylim(54, 72)
            
            # Add Alaska wind project points
            alaska_wind_projects = wind_data[wind_data['state'] == 'AK'].dropna(subset=['lat', 'long'])
            ak_opp_wind = alaska_wind_projects[alaska_wind_projects[variable] == 1]
            ak_no_opp_wind = alaska_wind_projects[alaska_wind_projects[variable] == 0]
            
            ax_wind_alaska.scatter(ak_no_opp_wind['long'], ak_no_opp_wind['lat'], 
                                  c='#808080', s=3, alpha=0.6, edgecolors='none')
            ax_wind_alaska.scatter(ak_opp_wind['long'], ak_opp_wind['lat'], 
                                  c='black', s=4, alpha=0.6, edgecolors='none')
        
        ax_wind_alaska.set_title('Alaska', fontsize=9, pad=2)
        ax_wind_alaska.axis('off')
        
        # Wind Hawaii inset (bottom subplot)
        wind_hawaii_left = 0.15
        wind_hawaii_top = 0.15
        ax_wind_hawaii = fig.add_axes([wind_hawaii_left, wind_hawaii_top, hawaii_width, hawaii_height])
        
        if not wind_hawaii.empty:
            wind_hawaii.plot(column=f'wind_percentage_{variable}_1',
                            ax=ax_wind_hawaii,
                            cmap='OrRd',
                            vmin=vmin,
                            vmax=vmax,
                            edgecolor='gray',
                            linewidth=0.8,
                            missing_kwds={'color': '#A0A0A0', 'edgecolor': 'gray'})
            
            # Add Hawaii wind project points
            hawaii_wind_projects = wind_data[wind_data['state'] == 'HI'].dropna(subset=['lat', 'long'])
            hi_opp_wind = hawaii_wind_projects[hawaii_wind_projects[variable] == 1]
            hi_no_opp_wind = hawaii_wind_projects[hawaii_wind_projects[variable] == 0]
            
            ax_wind_hawaii.scatter(hi_no_opp_wind['long'], hi_no_opp_wind['lat'], 
                                  c='#808080', s=3, alpha=0.6, edgecolors='none')
            ax_wind_hawaii.scatter(hi_opp_wind['long'], hi_opp_wind['lat'], 
                                  c='black', s=4, alpha=0.6, edgecolors='none')
        
        ax_wind_hawaii.set_title('Hawaii', fontsize=9, pad=2)
        ax_wind_hawaii.axis('off')
        
        # Add shared colorbar
        cbar_ax = fig.add_axes([0.75, 0.3, 0.02, 0.4])
        sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Projects with Opposition Mention (%)', fontsize=12, labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='#808080', s=20, alpha=0.6, label='No Opposition Mention'),
            plt.scatter([], [], c='black', s=20, alpha=0.6, label='Opposition Mention'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#A0A0A0', label='No Projects')
        ]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.12), 
                  ncol=3, fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        # Layout adjustment
        plt.tight_layout()
        plt.subplots_adjust(right=0.73, bottom=0.15)
        
        # Save
        output_path = Path(output_folder) / "map_solar_wind_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.close()
        
        print(f"   Saved complete map with Alaska/Hawaii insets to {output_path}")
        
    except Exception as e:
        print(f"   Error creating styled map: {e}")
        # Fallback to copying existing map
        import shutil
        source_path = Path("viz/test_outputs/map_solar_wind_comparison.png")
        if source_path.exists():
            dest_path = Path(output_folder) / "map_solar_wind_comparison.png"
            shutil.copy2(source_path, dest_path)
            print(f"   Fallback: Copied existing map to {dest_path}")

def main():
    """Main function"""
    print("🚀 Publication Plot Generator (Focused)")
    print("=" * 50)
    
    # Setup
    setup_style()
    
    # Create output directory
    output_folder = Path("viz/paper-ready")
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f" Output: {output_folder}")
    
    # Load data
    data = load_data()
    
    # Generate plots
    print("\n Main plots...")
    plot_mention_scores_by_year(data, output_folder)
    plot_capacity_distribution(data, output_folder)
    
    print("\n Additional plots...")
    plot_solar_wind_variables(data, output_folder)
    create_styled_map(data, output_folder)
    
    print("\n Generation complete!")

if __name__ == "__main__":
    main()