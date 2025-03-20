import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import validation module
from validate_submission import validate_kaggle_submission

def main():
    parser = argparse.ArgumentParser(description='Check Kaggle March Madness submission for format compliance')
    parser.add_argument('--submission', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'submissions', 'ensemble_submission.csv'),
                        help='Path to the submission file to check')
    parser.add_argument('--sample', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'data', 'kaggle-data', 'SampleSubmissionStage1.csv'),
                        help='Path to Kaggle sample submission')
    parser.add_argument('--output', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'evaluation', 'submission_check'),
                        help='Directory to save validation outputs')
    
    args = parser.parse_args()
    
    print("=== Checking Kaggle Submission Format ===")
    
    # Verify submission file exists
    if not os.path.exists(args.submission):
        print(f"ERROR: Submission file not found at {args.submission}")
        return False
    
    # Validate the submission
    validate_kaggle_submission(args.submission, args.sample, args.output)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
