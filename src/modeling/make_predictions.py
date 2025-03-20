import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_prediction_data(matchups_df, feature_columns):
    """
    Prepare data for making predictions
    """
    # Check if required columns exist
    missing_cols = [col for col in feature_columns if col not in matchups_df.columns]
    if missing_cols:
        print(f"WARNING: Missing {len(missing_cols)} required feature columns")
        print(f"First 5 missing: {missing_cols[:5]}")
        
        # Use only available columns to avoid errors
        available_cols = [col for col in feature_columns if col in matchups_df.columns]
        print(f"Proceeding with {len(available_cols)} available columns")
        X = matchups_df[available_cols].copy()
    else:
        # All columns are available
        X = matchups_df[feature_columns].copy()
    
    # Handle missing values - impute with mean to avoid errors
    for col in X.columns:
        if X[col].isnull().any():
            mean_val = X[col].mean()
            if pd.isna(mean_val):  # If mean is also NA, use 0
                mean_val = 0
            X[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in {col} with {mean_val:.4f}")
    
    # Keep track of matchup IDs
    ids = matchups_df['ID'].values if 'ID' in matchups_df.columns else None
    
    if ids is None:
        print("WARNING: ID column missing from matchups dataframe")
        # Create dummy IDs if none available
        ids = [f"2025_{i+1000}_{i+1001}" for i in range(len(X))]
    
    return X, ids

def make_predictions(model, X, ids):
    """
    Make predictions using the trained model
    
    Returns predicted probabilities that Team1 (lower TeamID) wins
    """
    # Check if model is valid
    if not hasattr(model, 'predict_proba'):
        print("ERROR: Model doesn't have predict_proba method")
        # Return a baseline prediction of 0.5 for all samples
        print("Falling back to baseline predictions (0.5)")
        y_pred_proba = np.full(len(X), 0.5)
    else:
        try:
            # Generate probability predictions
            y_pred_proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"ERROR during prediction: {str(e)}")
            # Fall back to baseline predictions
            print("Falling back to baseline predictions (0.5)")
            y_pred_proba = np.full(len(X), 0.5)
    
    # Create submission dataframe in Kaggle format
    submission = pd.DataFrame({
        'ID': ids,
        'Pred': y_pred_proba
    })
    
    return submission

def validate_submission(submission_df):
    """
    Validate submission format according to Kaggle requirements
    """
    # Check column names
    if list(submission_df.columns) != ['ID', 'Pred']:
        print("WARNING: Submission columns should be ['ID', 'Pred']")
    
    # Check ID format
    id_pattern = r'^\d{4}_\d{4}_\d{4}$'
    invalid_ids = submission_df[~submission_df['ID'].str.match(id_pattern)]
    if len(invalid_ids) > 0:
        print(f"WARNING: {len(invalid_ids)} IDs don't match the pattern SSSS_XXXX_YYYY")
        print("First 5 examples:", invalid_ids['ID'].head(5).tolist())
    
    # Check that predictions are probabilities
    if submission_df['Pred'].min() < 0 or submission_df['Pred'].max() > 1:
        print(f"WARNING: Predictions should be probabilities (0-1), but range is {submission_df['Pred'].min()}-{submission_df['Pred'].max()}")
    
    # Verify that the lower TeamID is always first in the ID string
    for idx, row in submission_df.head(10).iterrows():
        id_parts = row['ID'].split('_')
        team1, team2 = int(id_parts[1]), int(id_parts[2])
        if team1 >= team2:
            print(f"WARNING: Team IDs in wrong order: {row['ID']}, Team1 should be less than Team2")
            break
    
    print(f"Submission has {len(submission_df)} predictions")
    print(f"Sample predictions: {submission_df['Pred'].describe()}")
    
    return True

def generate_submission_file(model, matchups_df, feature_columns, output_path):
    """
    Generate a submission file from a model and matchups data
    """
    # Prepare data
    X, ids = prepare_prediction_data(matchups_df, feature_columns)
    
    # Make predictions
    submission = make_predictions(model, X, ids)
    
    # Validate submission format
    validate_submission(submission)
    
    # Save submission file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    print(f"Contains {len(submission)} predictions")
    
    return submission

def generate_ensemble_submission(models, weights, matchups_df, feature_columns, output_path):
    """
    Generate a submission using an ensemble of models
    """
    # Prepare data
    X, ids = prepare_prediction_data(matchups_df, feature_columns)
    
    # Initialize predictions
    weighted_preds = np.zeros(len(X))
    sum_weights = sum(weights.values())
    
    # Generate weighted predictions
    for model_type, model in models.items():
        weight = weights.get(model_type, 1)
        model_preds = model.predict_proba(X)[:, 1]
        weighted_preds += model_preds * (weight / sum_weights)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': ids,
        'Pred': weighted_preds
    })
    
    # Save submission file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Ensemble submission file saved to {output_path}")
    print(f"Contains {len(submission)} predictions")
    
    return submission
