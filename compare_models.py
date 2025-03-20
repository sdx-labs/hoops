import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
import argparse

def compare_submissions(submission_paths, names=None, output_dir=None):
    """
    Compare multiple submission files to analyze differences between models
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load all submissions
    submissions = []
    
    for i, path in enumerate(submission_paths):
        try:
            df = pd.read_csv(path)
            name = names[i] if names and i < len(names) else f"Model {i+1}"
            df['Model'] = name
            submissions.append(df)
            print(f"Loaded {path} with {len(df)} predictions")
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
    
    if not submissions:
        print("No valid submission files loaded")
        return False
    
    # Combine all submissions
    all_preds = pd.concat(submissions, ignore_index=True)
    
    # Create a pivot table for comparison
    pivot_df = all_preds.pivot(index='ID', columns='Model', values='Pred')
    
    # Calculate correlation between models
    corr = pivot_df.corr()
    print("\nModel Prediction Correlation:")
    print(corr)
    
    # Calculate summary statistics for each model
    stats = []
    for model in pivot_df.columns:
        preds = pivot_df[model]
        stats.append({
            'Model': model,
            'Min': preds.min(),
            'Max': preds.max(),
            'Mean': preds.mean(),
            'Median': preds.median(),
            'Std Dev': preds.std(),
            '% < 0.3': (preds < 0.3).mean() * 100,
            '% 0.3-0.7': ((preds >= 0.3) & (preds <= 0.7)).mean() * 100,
            '% > 0.7': (preds > 0.7).mean() * 100
        })
    
    stats_df = pd.DataFrame(stats)
    print("\nModel Prediction Statistics:")
    print(stats_df)
    
    # Visualizations
    if output_dir:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0.8, vmax=1)
        plt.title('Model Prediction Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_correlation.png'), dpi=300)
        
        # Prediction distributions
        plt.figure(figsize=(12, 8))
        for model in pivot_df.columns:
            sns.kdeplot(pivot_df[model], label=model)
        plt.title('Prediction Distributions by Model')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'), dpi=300)
        
        # Box plot of predictions
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pivot_df)
        plt.title('Prediction Ranges by Model')
        plt.ylabel('Predicted Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_boxplot.png'), dpi=300)
        
        # Save statistics to CSV
        stats_df.to_csv(os.path.join(output_dir, 'model_statistics.csv'), index=False)
        
        # Save model correlations to CSV
        corr.to_csv(os.path.join(output_dir, 'model_correlations.csv'))
    
    # Calculate pairwise differences
    print("\nPairwise Differences (Mean Absolute Difference):")
    for i, model1 in enumerate(pivot_df.columns):
        for j, model2 in enumerate(pivot_df.columns):
            if i < j:
                mad = abs(pivot_df[model1] - pivot_df[model2]).mean()
                print(f"  {model1} vs {model2}: {mad:.4f}")
    
    return {
        'statistics': stats_df,
        'correlations': corr
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model predictions")
    parser.add_argument('submissions', nargs='+',
                      help='Paths to submission files to compare')
    parser.add_argument('--names', nargs='+',
                      help='Names for each model (optional)')
    parser.add_argument('--output', type=str, 
                      default='/Volumes/MINT/projects/model/evaluation/model_comparison',
                      help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    if args.names and len(args.names) != len(args.submissions):
        print("Warning: Number of names doesn't match number of submissions")
        args.names = None
    
    compare_submissions(args.submissions, args.names, args.output)
