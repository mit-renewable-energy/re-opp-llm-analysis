#!/usr/bin/env python3
"""
Publication Table Generator for Renewable Energy Dispute Analysis
Generates formatted tables for academic publication from analysis data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

def load_demographic_stats():
    """Load the demographic comparison data"""
    print(" Loading demographic comparison data...")
    
    try:
        # Load all projects demographics (from Aug20_2025_all_demographics_stats.csv if it exists)
        all_demo_path = Path("viz/plots/all_demographics_stats.csv")
        if all_demo_path.exists():
            all_demo = pd.read_csv(all_demo_path)
            print(f" Loaded all projects demographics: {len(all_demo)} variables")
        else:
            print(" All projects demographics file not found")
            all_demo = None
            
        # Load solar demographics
        solar_demo_path = Path("viz/plots/solar_dem_stats.csv")
        if solar_demo_path.exists():
            solar_demo = pd.read_csv(solar_demo_path)
            print(f" Loaded solar demographics: {len(solar_demo)} variables")
        else:
            print(" Solar demographics file not found")
            return None
            
        # Load wind demographics
        wind_demo_path = Path("viz/plots/wind_dem_stats.csv")
        if wind_demo_path.exists():
            wind_demo = pd.read_csv(wind_demo_path)
            print(f" Loaded wind demographics: {len(wind_demo)} variables")
        else:
            print(" Wind demographics file not found")
            return None
            
        return all_demo, solar_demo, wind_demo
        
    except Exception as e:
        print(f" Error loading demographic data: {e}")
        return None

def format_demographic_table(all_demo, solar_demo, wind_demo):
    """Create the formatted demographic comparison table"""
    print(" Formatting demographic comparison table...")
    
    # Create the combined table structure
    combined_table = []
    
    # Use solar data as the base (all should have same variables)
    for idx, row in solar_demo.iterrows():
        variable = row['Variables']
        
        # Clean up variable names for publication
        clean_var = variable.replace('(per sq mile)', '').strip()
        if clean_var == 'Population Density':
            clean_var = 'Population\nDensity'
        elif 'Percent 25 or Older with Less Than HS Degree' in clean_var:
            clean_var = '% >25 without\nHS Degree'
        elif 'Percent Ages 10-64' in clean_var:
            clean_var = '% Ages 10-64'
        elif 'Percent of Tract in Tribal Areas' in clean_var:
            clean_var = '% of Tract in\nTribal Areas'
        elif 'LMI Percentile Based on AMI' in clean_var:
            clean_var = 'LMI Percentile\nFrom AMI'
        elif 'PM2.5 in the Air Percentile' in clean_var:
            clean_var = 'PM2.5 in the Air\nPercentile'
        elif 'Energy Burden Percentile' in clean_var:
            clean_var = 'Energy Burden\nPercentile'
        elif 'Unemployment Percentile' in clean_var:
            clean_var = 'Unemployment\nPercentile'
        
        # Get corresponding wind data
        wind_row = wind_demo[wind_demo['Variables'] == variable]
        
        # Get all projects data if available
        if all_demo is not None:
            all_row = all_demo[all_demo['Variables'] == variable]
        else:
            all_row = None
        
        # Build the combined row with proper column headers matching the screenshot
        combined_row = {
            'Variables': clean_var,
            # All Projects columns
            'Average With Opposition (All)': all_row['Average With Opposition'].iloc[0] if all_row is not None and len(all_row) > 0 else '',
            'Average Without Opposition (All)': all_row['Average Without Opposition'].iloc[0] if all_row is not None and len(all_row) > 0 else '',
            't-statistic (All)': all_row['t-statistic'].iloc[0] if all_row is not None and len(all_row) > 0 else '',
            # Solar columns
            'Average With Opposition (Solar)': row['Average With Opposition'],
            'Average Without Opposition (Solar)': row['Average Without Opposition'],
            't-statistic (Solar)': row['t-statistic'],
            # Wind columns
            'Average With Opposition (Wind)': wind_row['Average With Opposition'].iloc[0] if len(wind_row) > 0 else '',
            'Average Without Opposition (Wind)': wind_row['Average Without Opposition'].iloc[0] if len(wind_row) > 0 else '',
            't-statistic (Wind)': wind_row['t-statistic'].iloc[0] if len(wind_row) > 0 else ''
        }
        
        combined_table.append(combined_row)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_table)
    
    # Clean up the formatting of numerical values
    numeric_cols = [col for col in combined_df.columns if col != 'Variables']
    for col in numeric_cols:
        if 't_statistic' in col:
            # Keep t-statistics as is (they include significance stars)
            continue
        else:
            # Format numeric values consistently
            combined_df[col] = combined_df[col].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x))
    
    return combined_df

def load_validation_data():
    """Load validation accuracy data"""
    print("🔍 Loading validation accuracy data...")
    
    try:
        # Load the validation results
        val_path = Path("data/final/validation_human_labels.json")
        if not val_path.exists():
            print(" Validation data file not found")
            return None
            
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        print(f" Loaded validation data: {len(validation_data)} projects")
        return validation_data
        
    except Exception as e:
        print(f" Error loading validation data: {e}")
        return None

def calculate_accuracy_metrics(validation_data):
    """Calculate accuracy metrics from validation data"""
    print(" Calculating accuracy metrics...")
    
    # Define article and perception variables with nice labels
    article_vars = [f'article_{letter}' for letter in 'ABCDEFGHIJ']
    article_labels = [f'Article {letter}' for letter in 'ABCDEFGHIJ']
    
    perception_vars = [
        'mention_support', 'mention_opp', 'physical_opp', 'policy_opp', 
        'legal_opp', 'opinion_opp', 'environmental_opp', 'participation_opp',
        'tribal_opp', 'health_opp', 'intergov_opp', 'property_opp', 
        'compensation', 'delay', 'co_land_use', 'narrative'
    ]
    
    perception_labels = [
        'Support', 'Opposition', 'Physical', 'Policy',
        'Legal', 'Opinion', 'Environmental', 'Participation',
        'Tribal', 'Health', 'Intergovt', 'Property',
        'Compensation', 'Delay', 'Co-Use', 'Narrative'
    ]
    
    # Calculate metrics for each variable
    def calculate_metrics_for_var(var_name):
        accuracy_key = f'accuracy_{var_name}'
        
        # Count TP, TN, FP, FN
        tp = fn = fp = tn = 0
        total = 0
        
        for project in validation_data:
            if accuracy_key in project.get('accuracy_summary', {}):
                result = project['accuracy_summary'][accuracy_key]
                total += 1
                
                if result == 'TP':
                    tp += 1
                elif result == 'TN':
                    tn += 1
                elif result == 'FP':
                    fp += 1
                elif result == 'FN':
                    fn += 1
        
        if total == 0:
            return None
            
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        prevalence = (tp + fn) / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'prevalence': prevalence,
            'n': total,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    # Calculate for article variables
    article_results = []
    for var, label in zip(article_vars, article_labels):
        metrics = calculate_metrics_for_var(var)
        if metrics:
            article_results.append({
                'Variable': label,
                'Accuracy (%)': f"{metrics['accuracy']*100:.1f}",
                'FPR': f"{metrics['fpr']:.3f}",
                'FNR': f"{metrics['fnr']:.3f}",
                'Prevalence': f"{metrics['prevalence']:.3f}",
                'n': metrics['n']
            })
    
    # Calculate for perception variables
    perception_results = []
    for var, label in zip(perception_vars, perception_labels):
        metrics = calculate_metrics_for_var(var)
        if metrics:
            perception_results.append({
                'Variable': label,
                'Accuracy (%)': f"{metrics['accuracy']*100:.1f}",
                'FPR': f"{metrics['fpr']:.3f}",
                'FNR': f"{metrics['fnr']:.3f}",
                'Prevalence': f"{metrics['prevalence']:.3f}",
                'n': metrics['n']
            })
    
    return article_results, perception_results

def save_tables(demographic_table, article_accuracy, perception_accuracy, output_folder):
    """Save all tables to CSV files"""
    print("💾 Saving tables...")
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save demographic comparison table
    demo_path = output_path / "demographic_comparison_table.csv"
    demographic_table.to_csv(demo_path, index=False)
    print(f" Saved demographic comparison table: {demo_path}")
    
    # Save article accuracy table
    article_df = pd.DataFrame(article_accuracy)
    article_path = output_path / "validation_accuracy_articles.csv"
    article_df.to_csv(article_path, index=False)
    print(f" Saved article accuracy table: {article_path}")
    
    # Save perception accuracy table
    perception_df = pd.DataFrame(perception_accuracy)
    perception_path = output_path / "validation_accuracy_perception.csv"
    perception_df.to_csv(perception_path, index=False)
    print(f" Saved perception accuracy table: {perception_path}")
    
    return demo_path, article_path, perception_path

def main():
    """Main function to generate all tables"""
    print("🚀 Starting Publication Table Generator")
    print("=" * 60)
    
    # Create output directory
    output_folder = Path("viz/paper-ready")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate demographic comparison table
    print("\n Generating demographic comparison table...")
    demo_data = load_demographic_stats()
    if demo_data:
        all_demo, solar_demo, wind_demo = demo_data
        demographic_table = format_demographic_table(all_demo, solar_demo, wind_demo)
    else:
        print(" Cannot generate demographic table")
        return
    
    # Generate validation accuracy tables
    print("\n🔍 Generating validation accuracy tables...")
    validation_data = load_validation_data()
    if validation_data:
        article_accuracy, perception_accuracy = calculate_accuracy_metrics(validation_data)
    else:
        print(" Cannot generate validation tables")
        return
    
    # Save all tables
    print("\n💾 Saving tables...")
    demo_path, article_path, perception_path = save_tables(
        demographic_table, article_accuracy, perception_accuracy, output_folder
    )
    
    print("\n TABLE GENERATION COMPLETE!")
    print("=" * 60)
    print(f" All tables saved to: {output_folder}")
    print(f" Demographic comparison: {demo_path.name}")
    print(f" Article accuracy: {article_path.name}")  
    print(f" Perception accuracy: {perception_path.name}")

if __name__ == "__main__":
    main()