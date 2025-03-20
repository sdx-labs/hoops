import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def validate_kaggle_submission(submission_path, sample_path=None, output_dir=None):
    """
    Thoroughly validate a submission file for the March Madness competition
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load submission
    print(f"Loading submission file: {submission_path}")
    submission = pd.read_csv(submission_path)
    print(f"Submission shape: {submission.shape}")
    
    # Basic checks
    print("\n=== Basic Format Checks ===")
    
    # Check columns
    if list(submission.columns) != ['ID', 'Pred']:
        print("ERROR: Submission should have exactly two columns: 'ID' and 'Pred'")
    else:
        print("PASS: Submission has correct columns")
    
    # Check for nulls
    if submission.isnull().any().any():
        print(f"ERROR: Submission has {submission.isnull().sum().sum()} null values")
    else:
        print("PASS: No null values found")
    
    # Check ID format
    print("\n=== ID Format Checks ===")
    id_parts = submission['ID'].str.split('_', expand=True)
    id_parts.columns = ['Season', 'Team1', 'Team2']
    
    # Convert to integers for comparisons
    id_parts['Season'] = id_parts['Season'].astype(int)
    id_parts['Team1'] = id_parts['Team1'].astype(int)
    id_parts['Team2'] = id_parts['Team2'].astype(int)
    
    # Check season
    seasons = id_parts['Season'].unique()
    print(f"Seasons in submission: {sorted(seasons)}")
    
    # Check team ordering (Team1 should be < Team2)
    team_order_issues = (id_parts['Team1'] >= id_parts['Team2']).sum()
    if team_order_issues > 0:
        print(f"ERROR: {team_order_issues} ID entries have Team1 >= Team2")
    else:
        print("PASS: All IDs have Team1 < Team2")
    
    # Check team ID ranges
    men_range = (1000, 1999)
    women_range = (3000, 3999)
    
    men_teams = id_parts[
        ((id_parts['Team1'] >= men_range[0]) & (id_parts['Team1'] <= men_range[1])) &
        ((id_parts['Team2'] >= men_range[0]) & (id_parts['Team2'] <= men_range[1]))
    ]
    
    women_teams = id_parts[
        ((id_parts['Team1'] >= women_range[0]) & (id_parts['Team1'] <= women_range[1])) &
        ((id_parts['Team2'] >= women_range[0]) & (id_parts['Team2'] <= women_range[1]))
    ]
    
    mixed_teams = len(id_parts) - len(men_teams) - len(women_teams)
    
    print(f"Men's matchups: {len(men_teams)}")
    print(f"Women's matchups: {len(women_teams)}")
    print(f"Mixed gender matchups (should be 0): {mixed_teams}")
    
    if mixed_teams > 0:
        print("ERROR: Submission contains mixed gender matchups")
    
    # Check prediction values
    print("\n=== Prediction Value Checks ===")
    pred_min = submission['Pred'].min()
    pred_max = submission['Pred'].max()
    pred_mean = submission['Pred'].mean()
    
    print(f"Prediction range: [{pred_min:.6f}, {pred_max:.6f}], mean: {pred_mean:.6f}")
    
    if pred_min < 0 or pred_max > 1:
        print("ERROR: Predictions must be probabilities between 0 and 1")
    else:
        print("PASS: All predictions are within valid range [0,1]")
    
    # Compare with sample submission if provided
    if sample_path and os.path.exists(sample_path):
        print("\n=== Comparison with Sample Submission ===")
        sample = pd.read_csv(sample_path)
        print(f"Sample submission shape: {sample.shape}")
        
        # Check if all required IDs are present
        sample_ids = set(sample['ID'])
        submission_ids = set(submission['ID'])
        
        missing_ids = sample_ids - submission_ids
        extra_ids = submission_ids - sample_ids
        
        if missing_ids:
            print(f"ERROR: Missing {len(missing_ids)} IDs from sample submission")
            print(f"  First 5 missing: {list(missing_ids)[:5]}")
        else:
            print("PASS: Submission contains all required IDs")
        
        if extra_ids:
            print(f"WARNING: Submission has {len(extra_ids)} extra IDs not in sample")
    
    # Plot prediction distribution
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(submission['Pred'], bins=50, alpha=0.7)
        plt.title('Distribution of Predicted Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300)
        
        # Create separate plots for men's and women's predictions
        if len(men_teams) > 0:
            men_indices = men_teams.index
            plt.figure(figsize=(10, 6))
            plt.hist(submission.loc[men_indices, 'Pred'], bins=50, alpha=0.7)
            plt.title("Distribution of Men's Predictions")
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'mens_prediction_distribution.png'), dpi=300)
        
        if len(women_teams) > 0:
            women_indices = women_teams.index
            plt.figure(figsize=(10, 6))
            plt.hist(submission.loc[women_indices, 'Pred'], bins=50, alpha=0.7)
            plt.title("Distribution of Women's Predictions")
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'womens_prediction_distribution.png'), dpi=300)
    
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate Kaggle March Madness submission file')
    parser.add_argument('submission_path', type=str, help='Path to submission CSV file')
    parser.add_argument('--sample', type=str, default=None, help='Path to sample submission file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for validation plots')
    
    args = parser.parse_args()
    validate_kaggle_submission(args.submission_path, args.sample, args.output)
