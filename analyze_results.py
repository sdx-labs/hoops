import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def analyze_predictions(submission_path, output_dir=None):
    """
    Analyze prediction results to understand model behavior
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load submission
    print(f"Loading submission from {submission_path}")
    
    try:
        predictions = pd.read_csv(submission_path)
    except Exception as e:
        print(f"Error loading submission: {str(e)}")
        return None
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Parse ID components
    id_parts = predictions['ID'].str.split('_', expand=True)
    id_parts.columns = ['Season', 'Team1', 'Team2']
    
    # Convert columns to appropriate types
    id_parts['Season'] = id_parts['Season'].astype(int)
    id_parts['Team1'] = id_parts['Team1'].astype(int)
    id_parts['Team2'] = id_parts['Team2'].astype(int)
    
    # Add parsed components to the predictions
    predictions = pd.concat([predictions, id_parts], axis=1)
    
    # Separate men's and women's predictions
    mens_predictions = predictions[(predictions['Team1'] >= 1000) & (predictions['Team1'] < 2000) & 
                                 (predictions['Team2'] >= 1000) & (predictions['Team2'] < 2000)]
    
    womens_predictions = predictions[(predictions['Team1'] >= 3000) & (predictions['Team1'] < 4000) & 
                                   (predictions['Team2'] >= 3000) & (predictions['Team2'] < 4000)]
    
    print(f"Men's predictions: {len(mens_predictions)}")
    print(f"Women's predictions: {len(womens_predictions)}")
    
    # Check prediction distributions
    print("\nPrediction Statistics:")
    print(f"  Overall: min={predictions['Pred'].min():.4f}, max={predictions['Pred'].max():.4f}, mean={predictions['Pred'].mean():.4f}")
    print(f"  Men's: min={mens_predictions['Pred'].min():.4f}, max={mens_predictions['Pred'].max():.4f}, mean={mens_predictions['Pred'].mean():.4f}")
    print(f"  Women's: min={womens_predictions['Pred'].min():.4f}, max={womens_predictions['Pred'].max():.4f}, mean={womens_predictions['Pred'].mean():.4f}")
    
    # Calculate team strength scores
    print("\nCalculating team strength indicators...")
    
    # For each team, calculate average predicted win probability against all opponents
    team_strength = {}
    
    # Process men's predictions
    for _, row in mens_predictions.iterrows():
        team1, team2 = row['Team1'], row['Team2']
        pred = row['Pred']
        
        # Team1 is the lower ID and pred is their win probability
        if team1 not in team_strength:
            team_strength[team1] = {'wins': 0, 'losses': 0, 'total_win_prob': 0, 'total_loss_prob': 0}
        if team2 not in team_strength:
            team_strength[team2] = {'wins': 0, 'losses': 0, 'total_win_prob': 0, 'total_loss_prob': 0}
        
        team_strength[team1]['wins'] += 1
        team_strength[team1]['total_win_prob'] += pred
        
        team_strength[team2]['losses'] += 1
        team_strength[team2]['total_loss_prob'] += (1 - pred)
    
    # Convert to DataFrame
    strength_data = []
    for team, stats in team_strength.items():
        avg_win_prob = stats['total_win_prob'] / stats['wins'] if stats['wins'] > 0 else 0
        avg_loss_prob = stats['total_loss_prob'] / stats['losses'] if stats['losses'] > 0 else 0
        
        strength_data.append({
            'TeamID': team,
            'AvgWinProb': avg_win_prob,
            'AvgLossProb': avg_loss_prob,
            'OverallStrength': avg_win_prob - (1 - avg_loss_prob)
        })
    
    strength_df = pd.DataFrame(strength_data)
    strength_df = strength_df.sort_values('OverallStrength', ascending=False)
    
    print("\nTop 10 Strongest Teams:")
    for i, (_, row) in enumerate(strength_df.head(10).iterrows()):
        print(f"  {i+1}. Team {int(row['TeamID'])}: {row['OverallStrength']:.4f}")
    
    # Plot prediction distribution
    if output_dir:
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions['Pred'], bins=50, kde=True)
        plt.title('Distribution of Predicted Win Probabilities')
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300)
        
        # Plot team strength distribution
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='AvgWinProb', y='AvgLossProb', data=strength_df)
        plt.title('Team Strength Comparison')
        plt.xlabel('Average Win Probability')
        plt.ylabel('Average Loss Prevention')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'team_strength.png'), dpi=300)
        
        # Plot top 20 teams
        top_teams = strength_df.head(20)
        plt.figure(figsize=(14, 8))
        sns.barplot(x='TeamID', y='OverallStrength', data=top_teams)
        plt.title('Top 20 Teams by Overall Strength')
        plt.xlabel('Team ID')
        plt.ylabel('Overall Strength')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_teams.png'), dpi=300)
        
        # Save team strength data
        strength_df.to_csv(os.path.join(output_dir, 'team_strength.csv'), index=False)
    
    return strength_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze March Madness prediction results")
    parser.add_argument('--submission', type=str, 
                      default='/Volumes/MINT/projects/model/submissions/ensemble_submission.csv',
                      help='Path to submission file')
    parser.add_argument('--output', type=str, 
                      default='/Volumes/MINT/projects/model/evaluation/prediction_analysis',
                      help='Output directory for analysis results')
    
    args = parser.parse_args()
    analyze_predictions(args.submission, args.output)
