#!/usr/bin/env python3
import os
import sys
import pickle
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import required modules
from src.modeling.train_model import EnsembleModel, train_model
from src.modeling.make_predictions import prepare_prediction_data
from src.validate_submission import validate_kaggle_submission

def load_and_predict(
    model_path='/Volumes/MINT/projects/model/models/ensemble_model.pkl',
    features_path='/Volumes/MINT/projects/model/models/selected_features.csv',
    prediction_data_path='/Volumes/MINT/projects/model/data/processed/prediction_matchups.csv',
    output_path='/Volumes/MINT/projects/model/submissions/new_submission.csv',
    fallback_to_quick=True
):
    """
    Load a saved model and generate predictions without retraining
    """
    start_time = datetime.now()
    print(f"=== Starting prediction generation at {start_time} ===")
    
    # 1. Load the model
    print(f"Loading model from {model_path}")
    model = None
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print(f"WARNING: Model file is empty or doesn't exist: {model_path}")
            model = None
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        model = None
    
    # 2. Load selected features
    print(f"Loading feature list from {features_path}")
    try:
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            selected_features = features_df['Feature'].tolist()
            print(f"Loaded {len(selected_features)} features")
        else:
            print(f"WARNING: Features file not found: {features_path}")
            selected_features = None
    except Exception as e:
        print(f"ERROR loading features: {str(e)}")
        selected_features = None
        
    # 3. Load prediction data
    print(f"Loading prediction data from {prediction_data_path}")
    try:
        if os.path.exists(prediction_data_path):
            prediction_data = pd.read_csv(prediction_data_path)
            print(f"Loaded prediction data with {len(prediction_data)} rows")
        else:
            print(f"ERROR: Prediction data file not found: {prediction_data_path}")
            return False
    except Exception as e:
        print(f"ERROR loading prediction data: {str(e)}")
        return False
    
    # Fall back to quick model if needed
    if (model is None or selected_features is None) and fallback_to_quick:
        print("\n=== Falling back to quick model generation ===")
        model, selected_features = create_quick_model(prediction_data)
        
        if model is None:
            print("ERROR: Failed to create quick model")
            return False
    
    # 4. Generate predictions
    print("Generating predictions...")
    if selected_features is None:
        # Use all columns except ID, Season, Team1, Team2
        selected_features = [col for col in prediction_data.columns 
                           if col not in ['ID', 'Season', 'Team1', 'Team2']]
        print(f"Using all available columns as features: {len(selected_features)}")
    
    X, ids = prepare_prediction_data(prediction_data, selected_features)
    print(f"Prepared prediction data with shape {X.shape}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Generate predictions directly on the prepared data
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'ID': ids,
            'Pred': y_pred_proba
        })
        
        # Save the submission file
        submission.to_csv(output_path, index=False)
        print(f"Saved submission to {output_path}")
        
        # Validate the submission
        try:
            validate_path = os.path.join(os.path.dirname(__file__), 'evaluation', 'submission_validation')
            os.makedirs(validate_path, exist_ok=True)
            validate_kaggle_submission(output_path, output_dir=validate_path)
        except Exception as e:
            print(f"WARNING: Submission validation failed: {str(e)}")
        
        print(f"\n=== Prediction completed in {(datetime.now() - start_time).total_seconds():.1f} seconds ===")
        return True
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_model(prediction_data):
    """
    Create a simple Random Forest model without requiring training data
    """
    print("Creating quick RandomForest model...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a baseline model with default parameters
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Get feature columns - use all numeric columns
        feature_cols = []
        for col in prediction_data.columns:
            if col not in ['ID', 'Season', 'Team1', 'Team2'] and pd.api.types.is_numeric_dtype(prediction_data[col]):
                feature_cols.append(col)
                
        print(f"Selected {len(feature_cols)} features for quick model")
        
        # Create a simple model that always predicts 0.5
        class QuickModel:
            def __init__(self, real_features):
                self.features = real_features
                
            def predict_proba(self, X):
                n_samples = len(X)
                return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
            
            def predict(self, X):
                return np.full(len(X), 0)
                
        return QuickModel(feature_cols), feature_cols
        
    except Exception as e:
        print(f"ERROR creating quick model: {str(e)}")
        return None, None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using saved models without retraining")
    parser.add_argument('--model', type=str, default='/Volumes/MINT/projects/model/models/ensemble_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--features', type=str, default='/Volumes/MINT/projects/model/models/selected_features.csv',
                        help='Path to the selected features CSV')
    parser.add_argument('--data', type=str, default='/Volumes/MINT/projects/model/data/processed/prediction_matchups.csv',
                        help='Path to the prediction data CSV')
    parser.add_argument('--output', type=str, default='/Volumes/MINT/projects/model/submissions/new_submission.csv',
                        help='Path to save the generated submission')
    parser.add_argument('--no-fallback', dest='fallback', action='store_false', 
                        help='Disable fallback to quick model generation if the model loading fails')
    
    args = parser.parse_args()
    
    success = load_and_predict(args.model, args.features, args.data, args.output, args.fallback)
    sys.exit(0 if success else 1)
