import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def load_or_create_processed_data(data_dir='/Volumes/MINT/projects/model/data/processed', force_recreate=False):
    """
    Load already processed data or create simple processed data if doesn't exist
    """
    training_path = os.path.join(data_dir, 'training_matchups.csv')
    prediction_path = os.path.join(data_dir, 'prediction_matchups.csv')
    
    # Check if processed data exists
    if os.path.exists(training_path) and os.path.exists(prediction_path) and not force_recreate:
        print(f"Loading existing processed data from {data_dir}")
        training_data = pd.read_csv(training_path)
        prediction_data = pd.read_csv(prediction_path)
        return training_data, prediction_data
    
    # If we need to create data, let's use the test data generator for quick results
    print("No processed data found. Generating test data...")
    
    # Add src directory to path to access modules
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # Use test data generator (assumed to be quick)
    from generate_test_data import generate_test_kaggle_data, generate_test_kenpom_data
    
    test_dir = os.path.join(os.path.dirname(__file__), 'data/test-data')
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate test data
    print("Generating test Kaggle data...")
    test_data = generate_test_kaggle_data(test_dir)
    generate_test_kenpom_data(test_dir, test_data)
    
    # Now use simplified feature engineering and matchup creation
    print("Creating simplified matchups...")
    training_data, prediction_data = create_simple_matchups(test_data)
    
    # Save the data
    os.makedirs(data_dir, exist_ok=True)
    training_data.to_csv(training_path, index=False)
    prediction_data.to_csv(prediction_path, index=False)
    
    return training_data, prediction_data

def create_simple_matchups(test_data):
    """
    Create simplified matchups from test data
    """
    # Extract team stats
    teams = test_data['MTeams']
    results = pd.DataFrame(test_data['MRegularSeasonCompactResults'])
    
    # Create team stats
    team_stats = {}
    
    # For each team, calculate basic win rate stats
    for team_id in teams['TeamID']:
        wins = results[results['WTeamID'] == team_id]
        losses = results[results['LTeamID'] == team_id]
        
        win_count = len(wins)
        loss_count = len(losses)
        
        if win_count + loss_count == 0:
            continue
            
        # Calculate win rate
        win_rate = win_count / (win_count + loss_count)
        
        # Calculate points scored
        points_scored = wins['WScore'].sum() + losses['LScore'].sum()
        points_allowed = wins['LScore'].sum() + losses['WScore'].sum()
        
        avg_points = points_scored / (win_count + loss_count)
        avg_points_allowed = points_allowed / (win_count + loss_count)
        
        team_stats[team_id] = {
            'win_rate': win_rate,
            'avg_points': avg_points, 
            'avg_points_allowed': avg_points_allowed,
            'point_diff': avg_points - avg_points_allowed
        }
    
    # Create training matchups
    training_matchups = []
    for _, game in results.iterrows():
        winner_id = game['WTeamID']
        loser_id = game['LTeamID']
        
        # Skip if we don't have stats for both teams
        if winner_id not in team_stats or loser_id not in team_stats:
            continue
        
        # Create matchup with lower ID as Team1
        if winner_id < loser_id:
            team1, team2 = winner_id, loser_id
            target = 1  # Team1 won
        else:
            team1, team2 = loser_id, winner_id
            target = 0  # Team1 lost
            
        team1_stats = team_stats[team1]
        team2_stats = team_stats[team2]
        
        matchup = {
            'ID': f"{game['Season']}_{team1}_{team2}",
            'Season': game['Season'],
            'Team1': team1,
            'Team2': team2,
            'Target': target
        }
        
        # Add team stats
        for stat, value in team1_stats.items():
            matchup[f'Team1_{stat}'] = value
            
        for stat, value in team2_stats.items():
            matchup[f'Team2_{stat}'] = value
            
        # Add stat differences
        for stat in team1_stats:
            matchup[f'Diff_{stat}'] = team1_stats[stat] - team2_stats[stat]
            
        training_matchups.append(matchup)
    
    # Create prediction matchups (2025 season)
    prediction_season = 2025
    prediction_matchups = []
    
    # Get teams
    all_teams = sorted([t for t in team_stats.keys()])
    
    # Create all possible matchups between teams
    for i, team1 in enumerate(all_teams):
        for team2 in all_teams[i+1:]:
            matchup_id = f"{prediction_season}_{team1}_{team2}"
            
            # Basic matchup data
            matchup = {
                'ID': matchup_id,
                'Season': prediction_season,
                'Team1': team1,
                'Team2': team2
            }
            
            # Add team stats
            for stat, value in team_stats[team1].items():
                matchup[f'Team1_{stat}'] = value
                
            for stat, value in team_stats[team2].items():
                matchup[f'Team2_{stat}'] = value
                
            # Add stat differences
            for stat in team_stats[team1]:
                matchup[f'Diff_{stat}'] = team_stats[team1][stat] - team_stats[team2][stat]
                
            prediction_matchups.append(matchup)
    
    return pd.DataFrame(training_matchups), pd.DataFrame(prediction_matchups)

def quick_train_and_predict(model_type='rf', n_estimators=50, max_features=8, test_size=0.2, data_dir='/Volumes/MINT/projects/model/data/processed'):
    """
    Quick training and prediction with minimal processing
    
    Arguments:
    - model_type: 'rf' (Random Forest) or 'dt' (Decision Tree)
    - n_estimators: Number of trees for Random Forest
    - max_features: Maximum number of features to use in the model
    - test_size: Portion of data to use for testing
    - data_dir: Directory containing processed data
    
    Returns:
    - Trained model
    - Feature importances
    - Evaluation metrics
    - Submission dataframe
    """
    start_time = time.time()
    print(f"=== Quick Model Training started at {datetime.now()} ===")
    
    # 1. Load or create processed data
    training_data, prediction_data = load_or_create_processed_data(data_dir)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Prediction data shape: {prediction_data.shape}")
    
    # 2. Prepare features and target
    feature_cols = [col for col in training_data.columns if col not in ['ID', 'Season', 'Team1', 'Team2', 'Target']]
    print(f"Using {len(feature_cols)} features")
    
    # Limit features if needed
    if len(feature_cols) > max_features:
        print(f"Limiting to top {max_features} features")
        # Use diff features first as they tend to be most predictive
        diff_features = sorted([col for col in feature_cols if col.startswith('Diff_')])[:max_features//2]
        team1_features = sorted([col for col in feature_cols if col.startswith('Team1_')])[:max_features//4]
        team2_features = sorted([col for col in feature_cols if col.startswith('Team2_')])[:max_features//4]
        
        feature_cols = diff_features + team1_features + team2_features
    
    X = training_data[feature_cols]
    y = training_data['Target']
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 4. Train a simple model
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    else:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
        
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)
    log_loss_score = log_loss(y_test, y_pred_proba)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {log_loss_score:.4f}")
    
    # 6. Feature importance
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n=== Top Features ===")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # 7. Make predictions on all possible matchups
    print("\n=== Generating Predictions ===")
    X_pred = prediction_data[feature_cols]
    pred_probs = model.predict_proba(X_pred)[:, 1]
    
    # Create submission
    submission = pd.DataFrame({
        'ID': prediction_data['ID'],
        'Pred': pred_probs
    })
    
    # Save submission
    output_dir = os.path.join(os.path.dirname(__file__), 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, f'quick_submission_{model_type}.csv')
    submission.to_csv(submission_path, index=False)
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'quick_model_{model_type}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Record elapsed time
    elapsed_time = time.time() - start_time
    print(f"\n=== Quick model training completed in {elapsed_time:.2f} seconds ===")
    print(f"Submission saved to {submission_path}")
    print(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'importances': importances, 
        'metrics': {
            'accuracy': accuracy,
            'brier_score': brier,
            'log_loss': log_loss_score
        },
        'submission': submission
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick model training for March Madness predictions")
    parser.add_argument('--model', type=str, choices=['rf', 'dt'], default='rf', 
                       help='Model type (rf: Random Forest, dt: Decision Tree)')
    parser.add_argument('--trees', type=int, default=50,
                       help='Number of trees for Random Forest model')
    parser.add_argument('--features', type=int, default=8,
                       help='Maximum number of features to use')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test size for evaluation')
    
    args = parser.parse_args()
    
    quick_train_and_predict(
        model_type=args.model,
        n_estimators=args.trees,
        max_features=args.features,
        test_size=args.test_size
    )
