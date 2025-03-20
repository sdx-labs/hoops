import os
import sys
import pickle
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from data_processing.load_data import load_kaggle_data, load_kenpom_data
from feature_engineering.build_features import TeamFeatureBuilder
from feature_engineering.conference_tourney_features import build_conference_tourney_features, add_conference_features_to_team_data
from feature_engineering.create_matchups import create_tournament_predictions
from modeling.make_predictions import generate_submission_file
from validate_submission import validate_kaggle_submission

def generate_predictions_only(
    model_path='/Volumes/MINT/projects/model/models/ensemble_model.pkl',
    features_path='/Volumes/MINT/projects/model/models/selected_features.csv',
    output_path='/Volumes/MINT/projects/model/submissions/final_submission.csv',
    tournament_season=2025
):
    """
    Generate predictions using a previously trained model
    """
    print(f"=== Generating predictions for {tournament_season} tournament ===")
    
    # 1. Load the trained model
    print("Loading trained model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load selected features
    print("Loading feature list...")
    selected_features = pd.read_csv(features_path)['Feature'].tolist()
    print(f"Using {len(selected_features)} features")
    
    # 3. Load Kaggle data
    print("Loading Kaggle data...")
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    kaggle_data = load_kaggle_data(os.path.join(data_path, 'kaggle-data'))
    
    # 4. Load KenPom data if available
    kenpom_data = load_kenpom_data(os.path.join(data_path, 'kenpom/historical_kenpom_data.csv'))
    
    # 5. Build team features
    print("Building team features...")
    feature_builder = TeamFeatureBuilder(kaggle_data, kenpom_data)
    basic_features = feature_builder.build_basic_stats([tournament_season])
    advanced_features = feature_builder.build_advanced_stats([tournament_season])
    if kenpom_data is not None:
        kenpom_features = feature_builder.integrate_kenpom_data([tournament_season])
    team_features = feature_builder.combine_all_features([tournament_season])
    
    # 6. Build conference tourney features
    print("Building conference tournament features...")
    conf_features = build_conference_tourney_features(kaggle_data, seasons=[tournament_season])
    enhanced_features = add_conference_features_to_team_data(team_features, conf_features)
    
    # 7. Create tournament prediction matchups
    print("Creating tournament matchup predictions...")
    tournament_matchups = create_tournament_predictions(kaggle_data, enhanced_features, season=tournament_season)
    
    # 8. Generate predictions
    print("Generating predictions...")
    submission = generate_submission_file(model, tournament_matchups, selected_features, output_path)
    
    # 9. Validate submission
    print("Validating submission...")
    sample_submission_path = os.path.join(data_path, 'kaggle-data', 'SampleSubmissionStage1.csv')
    validation_output_dir = os.path.join(os.path.dirname(__file__), 'evaluation', 'submission_validation')
    validate_kaggle_submission(
        submission_path=output_path,
        sample_path=sample_submission_path,
        output_dir=validation_output_dir
    )
    
    print(f"\n=== Prediction generation complete ===")
    print(f"Submission file saved to: {output_path}")
    print(f"Contains {len(submission)} predictions")
    
    return submission

if __name__ == "__main__":
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'ensemble_model.pkl')
    features_path = os.path.join(os.path.dirname(__file__), 'models', 'selected_features.csv')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run the full pipeline first using run_pipeline.py")
        sys.exit(1)
        
    if not os.path.exists(features_path):
        print(f"ERROR: Selected features not found at {features_path}")
        print("Run the full pipeline first using run_pipeline.py")
        sys.exit(1)
    
    output_path = os.path.join(os.path.dirname(__file__), 'submissions', 'final_submission.csv')
    generate_predictions_only(model_path, features_path, output_path)
