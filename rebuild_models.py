#!/usr/bin/env python3
import os
import sys
import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the EnsembleModel class definition
from src.modeling.train_model import EnsembleModel

def rebuild_models(output_dir='/Volumes/MINT/projects/model/models', 
                  training_data_path='/Volumes/MINT/projects/model/data/processed/training_matchups.csv',
                  n_estimators=50):
    """
    Rebuild basic models from scratch using training data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("===== Rebuilding Models =====")
    
    if not os.path.exists(training_data_path):
        print(f"ERROR: Training data not found at {training_data_path}")
        return False
    
    # Load training data
    try:
        print(f"Loading training data from {training_data_path}")
        training_data = pd.read_csv(training_data_path)
        print(f"Loaded training data with {len(training_data)} rows")
    except Exception as e:
        print(f"ERROR loading training data: {str(e)}")
        return False
    
    # Prepare features and target
    feature_cols = [col for col in training_data.columns 
                   if col not in ['ID', 'Season', 'Team1', 'Team2', 'Target']]
    
    # In case we don't have enough features, use diff_ features
    if len(feature_cols) < 5:
        print("WARNING: Not enough features, creating difference features")
        team1_cols = [col for col in training_data.columns if col.startswith('Team1_')]
        team2_cols = [col for col in training_data.columns if col.startswith('Team2_')]
        
        for col1, col2 in zip(team1_cols, team2_cols):
            feature = col1.replace('Team1_', '')
            training_data[f'Diff_{feature}'] = training_data[col1] - training_data[col2]
        
        feature_cols = [col for col in training_data.columns 
                       if col not in ['ID', 'Season', 'Team1', 'Team2', 'Target']]
    
    X = training_data[feature_cols]
    y = training_data['Target']
    
    print(f"Selected {len(feature_cols)} features")
    
    # Save features list
    features_path = os.path.join(output_dir, 'selected_features.csv')
    pd.DataFrame({'Feature': feature_cols, 'Importance': np.ones(len(feature_cols))}).to_csv(features_path, index=False)
    print(f"Saved feature list to {features_path}")
    
    # Train individual models
    models = {}
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X, y)
    models['rf'] = rf_model
    
    rf_path = os.path.join(output_dir, 'rf_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Saved Random Forest model to {rf_path}")
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting model...")
    gbm_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    gbm_model.fit(X, y)
    models['gbm'] = gbm_model
    
    gbm_path = os.path.join(output_dir, 'gbm_model.pkl')
    with open(gbm_path, 'wb') as f:
        pickle.dump(gbm_model, f)
    print(f"Saved Gradient Boosting model to {gbm_path}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X, y)
    models['lr'] = lr_model
    
    lr_path = os.path.join(output_dir, 'lr_model.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"Saved Logistic Regression model to {lr_path}")
    
    # Create ensemble model
    weights = {'rf': 1.0, 'gbm': 1.2, 'lr': 0.8}
    ensemble_model = EnsembleModel(models, weights)
    
    # Save ensemble model
    ensemble_path = os.path.join(output_dir, 'ensemble_model.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    print(f"Saved Ensemble model to {ensemble_path}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild models from training data")
    parser.add_argument('--output', type=str, default='/Volumes/MINT/projects/model/models',
                      help='Output directory for model files')
    parser.add_argument('--data', type=str, default='/Volumes/MINT/projects/model/data/processed/training_matchups.csv',
                      help='Path to training data CSV file')
    parser.add_argument('--trees', type=int, default=50,
                      help='Number of trees to use (for faster training)')
    
    args = parser.parse_args()
    
    success = rebuild_models(args.output, args.data, args.trees)
    sys.exit(0 if success else 1)
