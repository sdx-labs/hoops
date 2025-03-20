import os
import sys
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from main import main
from validate_submission import validate_kaggle_submission

def run_full_pipeline():
    """
    Run the complete March Madness prediction pipeline
    """
    start_time = time.time()
    print(f"=== Starting March Madness Prediction Pipeline at {datetime.now()} ===")
    
    # Define paths
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    output_path = os.path.dirname(__file__)
    
    # Run main pipeline
    try:
        print("\n1. Running main prediction pipeline...")
        ensemble_model, selected_features, metrics = main(
            data_path=data_path,
            output_path=output_path,
            tournament_season=2025,
            train_seasons=range(2003, 2024),
            use_kenpom=True
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
        print("1. Verify the submission file with 'src/validate_submission.py'")
        print("2. Upload the submission to Kaggle")
        print("3. Fine-tune models if needed")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
