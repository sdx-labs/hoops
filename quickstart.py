#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def setup_environment():
    """Set up the project environment"""
    print("=== Setting up project environment ===")
    
    # Create directory structure
    directories = [
        'data/kaggle-data',
        'data/kenpom',
        'data/processed',
        'features',
        'models',
        'submissions',
        'evaluation',
        'evaluation/submission_validation'
    ]
    
    for directory in directories:
        path = os.path.join(os.path.dirname(__file__), directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    # Install required packages
    print("\nInstalling required packages...")
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'fuzzywuzzy',
        'python-Levenshtein'  # For faster fuzzy matching
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"  Installed {package}")
        except subprocess.CalledProcessError:
            print(f"  Failed to install {package}")
    
    return True

def check_data():
    """Check if required data files are present"""
    print("\n=== Checking for required data files ===")
    
    kaggle_dir = os.path.join(os.path.dirname(__file__), 'data/kaggle-data')
    
    # Check for essential Kaggle files
    essential_files = [
        'MTeams.csv',
        'WTeams.csv',
        'MNCAATourneyDetailedResults.csv',
        'MRegularSeasonDetailedResults.csv',
        'SampleSubmissionStage1.csv'
    ]
    
    missing_files = []
    for filename in essential_files:
        path = os.path.join(kaggle_dir, filename)
        if not os.path.exists(path):
            missing_files.append(filename)
    
    if missing_files:
        print("MISSING DATA FILES:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download the competition data from Kaggle and place the files in:")
        print(f"  {kaggle_dir}")
        return False
    
    print("All essential Kaggle data files found!")
    
    # Check for KenPom data
    kenpom_path = os.path.join(os.path.dirname(__file__), 'data/kenpom/historical_kenpom_data.csv')
    if os.path.exists(kenpom_path):
        print(f"KenPom data found at: {kenpom_path}")
        return True
    else:
        print(f"KenPom data not found at: {kenpom_path}")
        print("The pipeline can still run without KenPom data, but performance may be reduced.")
        return True

def run_pipeline(args):
    """Run the March Madness prediction pipeline"""
    print("\n=== Running March Madness Prediction Pipeline ===")
    
    # Build pipeline command
    cmd = [sys.executable, 'run_pipeline.py']
    
    if args.no_kenpom:
        cmd.append('--no-kenpom')
    
    if args.prepare_kenpom:
        cmd.append('--prepare-kenpom')
    
    if args.season:
        cmd.extend(['--season', str(args.season)])
    
    # Execute the pipeline
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Pipeline execution failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Quick start the March Madness prediction pipeline')
    parser.add_argument('--skip-setup', action='store_true', help='Skip environment setup')
    parser.add_argument('--skip-checks', action='store_true', help='Skip data checks')
    parser.add_argument('--no-kenpom', action='store_true', help='Run without KenPom data')
    parser.add_argument('--prepare-kenpom', action='store_true', help='Preprocess KenPom data before running pipeline')
    parser.add_argument('--season', type=int, default=2025, help='Tournament season to predict')
    
    args = parser.parse_args()
    
    # Step 1: Setup environment
    if not args.skip_setup:
        if not setup_environment():
            print("Failed to set up environment. Exiting.")
            return 1
    
    # Step 2: Check data files
    if not args.skip_checks:
        if not check_data():
            print("Data check failed. Exiting.")
            return 1
    
    # Step 3: Run the pipeline
    if not run_pipeline(args):
        print("Pipeline execution failed. Exiting.")
        return 1
    
    print("\n=== Quick start completed successfully! ===")
    print("Your submission file is ready at: submissions/ensemble_submission.csv")
    print("\nNext steps:")
    print("1. Verify the submission with: python check_submission.py")
    print("2. Upload the submission to Kaggle")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
