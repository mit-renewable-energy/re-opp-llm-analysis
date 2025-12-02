import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import sys
sys.path.append('.')
from config.config import get_raw_data_path, get_processed_data_path, get_final_data_path, get_data_path, get_viz_path

# Create output directory
os.makedirs('validation_viz', exist_ok=True)

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_validation_data():
    """Load the validation data from JSON"""
    print("Loading validation data...")
    with open("data/final/validation_human_labels.json", 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} projects")
    return data

def extract_validation_results(data):
    """Extract all validation results into a structured format"""
    print("Extracting validation results...")
    
    results = []
    
    for project in data:
        plant_code = project['plant_code']
        accuracy_summary = project['accuracy_summary']
        detailed_validation = project['detailed_validation']
        
        if detailed_validation is None:
            continue
            
        # Extract article validations
        article_validations = detailed_validation.get('article_validations', {})
        score_validations = detailed_validation.get('score_validations', {})
        narrative_validation = detailed_validation.get('narrative_validation', {})
        
        # Process each article
        for article_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            article_key = f'article_{article_letter}'
            if article_key in article_validations:
                validation = article_validations[article_key]
                results.append({
                    'plant_code': plant_code,
                    'variable': article_key,
                    'result': validation['result'],
                    'confidence': validation['confidence'],
                    'reasoning': validation['reasoning'],
                    'category': 'article'
                })
        
        # Process each score variable
        score_variables = [
            'mention_support', 'mention_opp', 'physical_opp', 'policy_opp', 
            'legal_opp', 'opinion_opp', 'environmental_opp', 'participation_opp',
            'tribal_opp', 'health_opp', 'intergov_opp', 'property_opp',
            'compensation', 'delay', 'co_land_use'
        ]
        
        for var in score_variables:
            if var in score_validations:
                validation = score_validations[var]
                results.append({
                    'plant_code': plant_code,
                    'variable': var,
                    'result': validation['result'],
                    'confidence': validation['confidence'],
                    'reasoning': validation['reasoning'],
                    'category': 'score'
                })
        
        # Process narrative
        if narrative_validation:
            results.append({
                'plant_code': plant_code,
                'variable': 'narrative',
                'result': narrative_validation['result'],
                'confidence': narrative_validation['confidence'],
                'reasoning': narrative_validation['reasoning'],
                'category': 'narrative'
            })
    
    return pd.DataFrame(results)

def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics"""
    print("Calculating summary statistics...")
    
    # Overall rates
    total_validations = len(df)
    result_counts = df['result'].value_counts()
    
    # Rates by category
    category_stats = df.groupby('category')['result'].value_counts().unstack(fill_value=0)
    
    # Rates by variable
    variable_stats = df.groupby('variable')['result'].value_counts().unstack(fill_value=0)
    
    # Confidence statistics
    confidence_stats = df.groupby('variable')['confidence'].agg(['mean', 'std', 'count']).round(3)
    
    # Accuracy rates (TP + TN) / Total
    accuracy_rates = {}
    for var in df['variable'].unique():
        var_data = df[df['variable'] == var]
        correct = len(var_data[var_data['result'].isin(['TP', 'TN'])])
        total = len(var_data)
        accuracy_rates[var] = correct / total if total > 0 else 0
    
    return {
        'total_validations': total_validations,
        'result_counts': result_counts,
        'category_stats': category_stats,
        'variable_stats': variable_stats,
        'confidence_stats': confidence_stats,
        'accuracy_rates': accuracy_rates
    }

def create_overall_rates_plot(stats):
    """Create plot for overall TP/TN/FP/FN rates"""
    print("Creating overall rates plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall rates
    result_counts = stats['result_counts']
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1']  # Green, Red, Teal, Blue
    
    ax1.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Overall Validation Results Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(result_counts.index, result_counts.values, color=colors)
    ax2.set_title('Count of Each Validation Result', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Result Type')
    
    # Add value labels on bars
    for i, v in enumerate(result_counts.values):
        ax2.text(i, v + max(result_counts.values) * 0.01, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('"viz/validation_outputs/"overall_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_accuracy_by_variable_plot(stats):
    """Create plot for accuracy rates by variable"""
    print("Creating accuracy by variable plot...")
    
    accuracy_rates = stats['accuracy_rates']
    
    # Separate articles and scores
    articles = {k: v for k, v in accuracy_rates.items() if k.startswith('article_')}
    scores = {k: v for k, v in accuracy_rates.items() if not k.startswith('article_')}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Articles
    article_df = pd.DataFrame(list(articles.items()), columns=['Variable', 'Accuracy'])
    article_df = article_df.sort_values('Accuracy', ascending=True)
    
    bars1 = ax1.barh(article_df['Variable'], article_df['Accuracy'], color='skyblue')
    ax1.set_title('Accuracy Rates by Article (A-J)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Accuracy Rate')
    ax1.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars1, article_df['Accuracy'])):
        ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}', 
                va='center', fontweight='bold')
    
    # Scores
    score_df = pd.DataFrame(list(scores.items()), columns=['Variable', 'Accuracy'])
    score_df = score_df.sort_values('Accuracy', ascending=True)
    
    bars2 = ax2.barh(score_df['Variable'], score_df['Accuracy'], color='lightcoral')
    ax2.set_title('Accuracy Rates by Score Variable', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Accuracy Rate')
    ax2.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars2, score_df['Accuracy'])):
        ax2.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('"viz/validation_outputs/"accuracy_by_variable.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_analysis_plot(df, stats):
    """Create plots for confidence analysis"""
    print("Creating confidence analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall confidence distribution
    ax1.hist(df['confidence'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["confidence"].mean():.3f}')
    ax1.legend()
    
    # Confidence by result type
    confidence_by_result = df.groupby('result')['confidence'].mean().sort_values(ascending=False)
    bars = ax2.bar(confidence_by_result.index, confidence_by_result.values, 
                   color=['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Average Confidence by Result Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Confidence')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, confidence_by_result.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Top 10 variables by confidence
    confidence_stats = stats['confidence_stats']
    top_confidence = confidence_stats.nlargest(10, 'mean')
    
    bars = ax3.barh(range(len(top_confidence)), top_confidence['mean'], color='lightgreen')
    ax3.set_yticks(range(len(top_confidence)))
    ax3.set_yticklabels(top_confidence.index)
    ax3.set_title('Top 10 Variables by Average Confidence', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Average Confidence')
    ax3.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_confidence['mean'])):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontweight='bold')
    
    # Bottom 10 variables by confidence
    bottom_confidence = confidence_stats.nsmallest(10, 'mean')
    
    bars = ax4.barh(range(len(bottom_confidence)), bottom_confidence['mean'], color='lightcoral')
    ax4.set_yticks(range(len(bottom_confidence)))
    ax4.set_yticklabels(bottom_confidence.index)
    ax4.set_title('Bottom 10 Variables by Average Confidence', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Confidence')
    ax4.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, bottom_confidence['mean'])):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('"viz/validation_outputs/"confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fp_fn_analysis_plot(df):
    """Create plots for FP/FN analysis"""
    print("Creating FP/FN analysis plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # FP vs FN by variable
    fp_counts = df[df['result'] == 'FP'].groupby('variable').size()
    fn_counts = df[df['result'] == 'FN'].groupby('variable').size()
    
    # Combine and sort by total errors
    error_df = pd.DataFrame({
        'FP': fp_counts,
        'FN': fn_counts
    }).fillna(0)
    error_df['Total_Errors'] = error_df['FP'] + error_df['FN']
    error_df = error_df.sort_values('Total_Errors', ascending=False).head(15)
    
    x = np.arange(len(error_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, error_df['FP'], width, label='False Positives', color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, error_df['FN'], width, label='False Negatives', color='#4ECDC4')
    
    ax1.set_title('False Positives vs False Negatives by Variable', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_df.index, rotation=45, ha='right')
    ax1.legend()
    
    # FP/FN ratio
    fp_fn_ratio = (error_df['FP'] / (error_df['FN'] + 1)).sort_values(ascending=False).head(15)
    
    bars = ax2.bar(range(len(fp_fn_ratio)), fp_fn_ratio.values, color='orange')
    ax2.set_title('FP/FN Ratio by Variable (Higher = More FPs)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('FP/FN Ratio')
    ax2.set_xticks(range(len(fp_fn_ratio)))
    ax2.set_xticklabels(fp_fn_ratio.index, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, fp_fn_ratio.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('"viz/validation_outputs/"fp_fn_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_tables(stats):
    """Generate summary tables and save as CSV"""
    print("Generating summary tables...")
    
    # Overall statistics
    overall_stats = pd.DataFrame({
        'Metric': ['Total Validations', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
        'Count': [
            stats['total_validations'],
            stats['result_counts'].get('TP', 0),
            stats['result_counts'].get('TN', 0),
            stats['result_counts'].get('FP', 0),
            stats['result_counts'].get('FN', 0)
        ]
    })
    
    # Calculate percentages
    total = stats['total_validations']
    overall_stats['Percentage'] = (overall_stats['Count'] / total * 100).round(2)
    
    # Accuracy rates by variable
    accuracy_df = pd.DataFrame(list(stats['accuracy_rates'].items()), 
                              columns=['Variable', 'Accuracy_Rate'])
    accuracy_df = accuracy_df.sort_values('Accuracy_Rate', ascending=False)
    
    # Confidence statistics
    confidence_df = stats['confidence_stats'].round(3)
    
    # Save tables
    overall_stats.to_csv('"viz/validation_outputs/"overall_statistics.csv', index=False)
    accuracy_df.to_csv('"viz/validation_outputs/"accuracy_by_variable.csv', index=False)
    confidence_df.to_csv('"viz/validation_outputs/"confidence_statistics.csv')
    
    return overall_stats, accuracy_df, confidence_df

def generate_insights(df, stats):
    """Generate insights from the analysis"""
    print("Generating insights...")
    
    insights = []
    
    # Overall accuracy
    total_correct = len(df[df['result'].isin(['TP', 'TN'])])
    overall_accuracy = total_correct / len(df)
    insights.append(f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    
    # Most accurate variables
    accuracy_rates = stats['accuracy_rates']
    most_accurate = max(accuracy_rates.items(), key=lambda x: x[1])
    least_accurate = min(accuracy_rates.items(), key=lambda x: x[1])
    insights.append(f"Most accurate variable: {most_accurate[0]} ({most_accurate[1]:.3f})")
    insights.append(f"Least accurate variable: {least_accurate[0]} ({least_accurate[1]:.3f})")
    
    # FP vs FN analysis
    fp_count = len(df[df['result'] == 'FP'])
    fn_count = len(df[df['result'] == 'FN'])
    fp_ratio = fp_count / (fp_count + fn_count) if (fp_count + fn_count) > 0 else 0
    insights.append(f"False Positive rate: {fp_ratio:.3f} ({fp_ratio*100:.1f}% of errors)")
    insights.append(f"False Negative rate: {1-fp_ratio:.3f} ({(1-fp_ratio)*100:.1f}% of errors)")
    
    # Confidence analysis
    avg_confidence = df['confidence'].mean()
    insights.append(f"Average confidence: {avg_confidence:.3f}")
    
    # High confidence but wrong
    high_conf_wrong = df[(df['confidence'] > 0.8) & (df['result'].isin(['FP', 'FN']))]
    insights.append(f"High confidence (>0.8) but wrong: {len(high_conf_wrong)} cases")
    
    # Low confidence but right
    low_conf_right = df[(df['confidence'] < 0.5) & (df['result'].isin(['TP', 'TN']))]
    insights.append(f"Low confidence (<0.5) but right: {len(low_conf_right)} cases")
    
    # Save insights
    with open('"viz/validation_outputs/"insights.txt', 'w') as f:
        f.write("VALIDATION ANALYSIS INSIGHTS\n")
        f.write("=" * 50 + "\n\n")
        for insight in insights:
            f.write(f"• {insight}\n")
    
    return insights

def main():
    """Main analysis function"""
    print("Starting validation analysis...")
    
    # Load data
    data = load_validation_data()
    df = extract_validation_results(data)
    
    # Calculate statistics
    stats = calculate_summary_statistics(df)
    
    # Create visualizations
    create_overall_rates_plot(stats)
    create_accuracy_by_variable_plot(stats)
    create_confidence_analysis_plot(df, stats)
    create_fp_fn_analysis_plot(df)
    
    # Generate tables
    overall_stats, accuracy_df, confidence_df = generate_summary_tables(stats)
    
    # Generate insights
    insights = generate_insights(df, stats)
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total validations analyzed: {stats['total_validations']}")
    print(f"Overall accuracy: {len(df[df['result'].isin(['TP', 'TN'])]) / len(df):.3f}")
    print(f"Average confidence: {df['confidence'].mean():.3f}")
    print(f"Files saved in: "viz/validation_outputs/"")
    print("\nKey insights:")
    for insight in insights[:5]:  # Show first 5 insights
        print(f"• {insight}")
    print("\nSee "viz/validation_outputs/"insights.txt for complete insights")

if __name__ == "__main__":
    main() 