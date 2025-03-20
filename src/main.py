import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
from data_processing.load_data import load_kaggle_data, load_kenpom_data
from feature_engineering.build_features import TeamFeatureBuilder
from feature_engineering.conference_tourney_features import build_conference_tourney_features, add_conference_features_to_team_data
from feature_engineering.create_matchups import create_historical_matchups, create_tournament_predictions, save_datasets
from modeling.train_model import prepare_model_data, select_features, train_ensemble, evaluate_model, save_model
from modeling.make_predictions import generate_submission_file, generate_ensemble_submission

def verify_kaggle_compliance(submission_path, sample_submission_path=None):
    """
    Verify that our submission complies with Kaggle requirements
    """
    submission = pd.read_csv(submission_path)
    print(f"\n--- Verifying Kaggle submission compliance ---")
    
    # Check basic format
    print(f"Submission format: {submission.shape[0]} rows, {submission.shape[1]} columns")
    print(f"Column names: {submission.columns.tolist()}")
    
    # Verify prediction values are in [0,1]
    pred_min = submission['Pred'].min()
    pred_max = submission['Pred'].max()
    print(f"Prediction range: [{pred_min:.4f}, {pred_max:.4f}]")
    
    if pred_min < 0 or pred_max > 1:
        print("ERROR: Predictions must be in the range [0,1]")
    
    # Check ID format
    id_sample = submission['ID'].iloc[0]
    print(f"ID format sample: {id_sample}")
    
    # Parse a sample ID to verify format
    parts = id_sample.split('_')
    if len(parts) != 3:
        print("ERROR: ID should have format SSSS_XXXX_YYYY")
    else:
        season, team1, team2 = parts
        print(f"  Season: {season}")
        print(f"  Team1 (lower ID): {team1}")
        print(f"  Team2 (higher ID): {team2}")
        
        # Verify team1 < team2
        if int(team1) >= int(team2):
            print("ERROR: Team1 should have lower ID than Team2")
    
    # Compare with sample submission if provided
    if sample_submission_path and os.path.exists(sample_submission_path):
        sample = pd.read_csv(sample_submission_path)
        print(f"\nSample submission: {sample.shape[0]} rows, {sample.shape[1]} columns")
        
        # Check if all required IDs are present
        missing_ids = set(sample['ID']) - set(submission['ID'])
        extra_ids = set(submission['ID']) - set(sample['ID'])
        
        if missing_ids:
            print(f"ERROR: Missing {len(missing_ids)} IDs from sample submission")
            print(f"  First 5 missing: {list(missing_ids)[:5]}")
        
        if extra_ids:
            print(f"WARNING: Submission has {len(extra_ids)} extra IDs not in sample")
    
    print("\nVerification completed.")
    
    return submission

def main(
    data_path='/Volumes/MINT/projects/model/data',
    output_path='/Volumes/MINT/projects/model',
    tournament_season=2025,
    train_seasons=range(2003, 2024),
    use_kenpom=True
):
    """
    Main function to run the entire model pipeline:
    1. Load data
    2. Build team features
    3. Create matchups dataset
    4. Train models
    5. Generate tournament predictions
    """
    print(f"======= March Madness Prediction Pipeline =======")
    print(f"Starting time: {datetime.now()}")
    
    # 1. Load data
    print("\n--- Loading data ---")
    kaggle_data = load_kaggle_data(os.path.join(data_path, 'kaggle-data'))
    
    kenpom_data = None
    if use_kenpom:
        kenpom_data = load_kenpom_data(os.path.join(data_path, 'kenpom/historical_kenpom_data.csv'))
    
    # 2. Build team features
    print("\n--- Building team features ---")
    feature_builder = TeamFeatureBuilder(kaggle_data, kenpom_data)
    
    # 2a. Build basic stats
    print("Building basic team statistics...")
    basic_features = feature_builder.build_basic_stats(train_seasons)
    
    # 2b. Build advanced stats
    print("Building advanced team statistics...")
    advanced_features = feature_builder.build_advanced_stats(train_seasons)
    
    # 2c. Integrate KenPom data if available
    if kenpom_data is not None:
        print("Integrating KenPom statistics...")
        kenpom_features = feature_builder.integrate_kenpom_data(train_seasons)
    
    # 2d. Combine all features
    print("Combining all feature sets...")
    team_features = feature_builder.combine_all_features(list(train_seasons) + [tournament_season])
    
    # 2e. Add conference tournament features
    print("Building conference tournament features...")
    conf_features = build_conference_tourney_features(kaggle_data, 
                                                   seasons=list(train_seasons) + [tournament_season])
    
    # 2f. Combine with conference features
    print("Adding conference tournament features...")
    enhanced_features = add_conference_features_to_team_data(team_features, conf_features)
    
    # Save all features
    feature_builder.save_features(os.path.join(output_path, 'features'))
    
    # 3. Create matchups datasets
    print("\n--- Creating matchup datasets ---")
    
    # 3a. Training data from historical matchups
    print("Creating historical matchups dataset...")
    historical_matchups = create_historical_matchups(kaggle_data, enhanced_features)
    
    # 3b. Tournament prediction data
    print(f"Creating {tournament_season} tournament prediction dataset...")
    tournament_matchups = create_tournament_predictions(kaggle_data, enhanced_features, season=tournament_season)
    
    # Save datasets
    save_datasets(historical_matchups, tournament_matchups, os.path.join(output_path, 'data/processed'))
    
    # 4. Train model
    print("\n--- Training prediction model ---")
    
    # 4a. Prepare model data
    X, y = prepare_model_data(historical_matchups)
    print(f"Training data shape: {X.shape}")
    
    # Handle empty dataset scenario
    if len(X) == 0 or len(np.unique(y)) < 2:
        print("CRITICAL ERROR: Insufficient data for training models")
        print("Using a baseline model that predicts 0.5 probability for all matchups")
        
        # Create a baseline model class
        class BaselineModel:
            def predict_proba(self, X):
                n_samples = len(X)
                return np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])
            
            def predict(self, X):
                return np.full(len(X), 0.5)
        
        # Skip the regular model training and use baseline
        ensemble_model = BaselineModel()
        selected_features = X.columns.tolist()
        metrics = {'accuracy': 0.5, 'log_loss': 0.693, 'brier_score': 0.25}
    else:
        # 4b. Feature selection
        print("Selecting most important features...")
        print(f"Input features available: {X.columns.tolist()[:10]}... (and {len(X.columns)-10} more)")
        print(f"Feature stats: min={X.min().min()}, max={X.max().max()}, mean={X.mean().mean()}")
        
        selected_features, importances = select_features(X, y, n_features=50)
        print(f"Selected {len(selected_features)} features")
        print("Top 10 features:")
        for i, feature in enumerate(selected_features[:10]):
            print(f"  {i+1}. {feature} - {importances[i]:.4f}")
        
        # 4c. Train models
        print("Training ensemble model...")
        X_selected = X[selected_features]
        ensemble_model, trained_models, metrics, cv_results = train_ensemble(X_selected, y)
        
        # 4d. Save models
        models_dir = os.path.join(output_path, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in trained_models.items():
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            save_model(model, model_path, model_name)
        
        # Save ensemble
        ensemble_path = os.path.join(models_dir, "ensemble_model.pkl")
        save_model(ensemble_model, ensemble_path, "ensemble")
        
        # Save feature list
        feature_path = os.path.join(models_dir, "selected_features.csv")
        pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances[np.argsort(importances)[::-1]][:len(selected_features)]
        }).to_csv(feature_path, index=False)
    
    # 5. Generate tournament predictions
    print("\n--- Generating tournament predictions ---")
    print(f"Selected features shape: {X_selected.shape}")
    print(f"Top features statistics: {X_selected[selected_features[:5]].describe().loc['mean']}")
    X_tourney = tournament_matchups[selected_features]
    
    # 5a. Individual model predictions
    submissions_dir = os.path.join(output_path, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)
    
    for model_name, model in trained_models.items():
        submission_path = os.path.join(submissions_dir, f"{model_name}_submission.csv")
        generate_submission_file(model, tournament_matchups, selected_features, submission_path)
    
    # 5b. Ensemble predictions
    weights = {
        'rf': 1.0,
        'gbm': 1.2,
        'lr': 0.8
    }
    
    ensemble_submission_path = os.path.join(submissions_dir, f"ensemble_submission.csv")
    ensemble_submission = generate_ensemble_submission(trained_models, weights, tournament_matchups, 
                                                     selected_features, ensemble_submission_path)
    
    # 6. Verify Kaggle compliance
    print("\n--- Verifying submission compliance with Kaggle format ---")
    
    # Path to Kaggle sample submission if available
    sample_submission_path = os.path.join(data_path, 'kaggle-data', 'SampleSubmissionStage1.csv')
    
    verify_kaggle_compliance(ensemble_submission_path, sample_submission_path)
    
    print(f"\n======= Pipeline completed: {datetime.now()} =======")
    return ensemble_model, selected_features, metrics

if __name__ == "__main__":
    main()
