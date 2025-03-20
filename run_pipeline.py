import os
import sys
import time
import argparse
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from main import main
from validate_submission import validate_kaggle_submission

def run_full_pipeline(use_kenpom=True, tournament_season=2025, train_seasons=range(2003, 2024), prepare_kenpom=False):
    """
    Run the complete March Madness prediction pipeline
    """
    start_time = time.time()
    print(f"=== Starting March Madness Prediction Pipeline at {datetime.now()} ===")
    print(f"Using KenPom data: {use_kenpom}")
    
    # Define paths
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    output_path = os.path.dirname(__file__)
    
    # Check if KenPom pre-processing is needed
    if use_kenpom and prepare_kenpom:
        try:
            print("\n--- Pre-processing KenPom data ---")
            
            # First check if the file exists
            kenpom_path = os.path.join(data_path, 'kenpom/historical_kenpom_data.csv')
            if not os.path.exists(kenpom_path):
                print(f"WARNING: KenPom file not found at {kenpom_path}")
                print("Continuing without KenPom data...")
                use_kenpom = False
            else:
                # Check if the enhanced KenPom data already exists
                enhanced_path = os.path.join(data_path, 'kenpom/enhanced_kenpom_data.csv')
                if os.path.exists(enhanced_path):
                    print(f"Enhanced KenPom data already exists at {enhanced_path}")
                    use_existing = input("Use existing enhanced data? [y/n]: ").lower() == 'y'
                    if not use_existing:
                        # Run the team name matcher to create mapping
                        import team_name_matcher
                        kaggle_dir = os.path.join(data_path, 'kaggle-data')
                        mapping_path = os.path.join(data_path, 'kenpom/team_mapping.csv')
                        
                        print("Creating team name mapping...")
                        team_name_matcher.match_team_names(
                            kenpom_path=kenpom_path,
                            kaggle_dir=kaggle_dir,
                            output_path=mapping_path,
                            threshold=85
                        )
                else:
                    # Run the team name matcher to create mapping
                    import team_name_matcher
                    kaggle_dir = os.path.join(data_path, 'kaggle-data')
                    mapping_path = os.path.join(data_path, 'kenpom/team_mapping.csv')
                    
                    print("Creating team name mapping...")
                    team_name_matcher.match_team_names(
                        kenpom_path=kenpom_path,
                        kaggle_dir=kaggle_dir,
                        output_path=mapping_path,
                        threshold=85
                    )
        except Exception as e:
            print(f"Error during KenPom pre-processing: {str(e)}")
            print("Continuing without KenPom data...")
            use_kenpom = False
    
    # Run main pipeline
    try:
        print("\n1. Running main prediction pipeline...")
        ensemble_model, selected_features, metrics = main(
            data_path=data_path,
            output_path=output_path,
            tournament_season=tournament_season,
            train_seasons=train_seasons,
            use_kenpom=use_kenpom
        )
        
        # Get path to the generated submission file
        submission_path = os.path.join(output_path, 'submissions', 'ensemble_submission.csv')
        sample_submission_path = os.path.join(data_path, 'kaggle-data', 'SampleSubmissionStage1.csv')
        
        # Validate the submission in a more comprehensive way
        print("\n2. Running comprehensive submission validation...")
        validation_output_dir = os.path.join(output_path, 'evaluation', 'submission_validation')
        validate_kaggle_submission(
            submission_path=submission_path,
            sample_path=sample_submission_path,
            output_dir=validation_output_dir
        )
        
        # Print final summary
        elapsed_time = (time.time() - start_time) / 60  # in minutes
        print(f"\n=== Pipeline completed successfully in {elapsed_time:.2f} minutes ===")
        print(f"Submission file saved to: {submission_path}")
        print("\nModel performance metrics:")
        print(f"  Brier score: {metrics.get('brier_score', 'N/A'):.6f}")
        print(f"  Log loss: {metrics.get('log_loss', 'N/A'):.6f}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.6f}")
        print("\nNext steps:")
        print("1. Verify the submission file with 'python check_submission.py'")
        print("2. Upload the submission to Kaggle")
        print("3. Fine-tune models if needed")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the March Madness prediction pipeline')
    parser.add_argument('--no-kenpom', dest='use_kenpom', action='store_false', 
                        help='Skip using KenPom data (use this if you have KenPom data issues)')
    parser.add_argument('--prepare-kenpom', action='store_true',
                        help='Run KenPom preprocessing to create team mapping before pipeline')
    parser.add_argument('--season', type=int, default=2025,
                        help='Tournament season to predict (default: 2025)')
    
    args = parser.parse_args()
    
    # Convert training seasons to a range
    train_seasons = range(2003, args.season)
    
    success = run_full_pipeline(
        use_kenpom=args.use_kenpom,
        tournament_season=args.season,
        train_seasons=train_seasons,
        prepare_kenpom=args.prepare_kenpom
    )
    
    sys.exit(0 if success else 1)
