# March Madness Prediction Pipeline

This project provides a complete pipeline for generating predictions for the NCAA March Madness basketball tournament in Kaggle's 2025 competition.

## Project Structure

- `data/` - Contains all data files
  - `kaggle-data/` - Kaggle competition data files
  - `kenpom/` - KenPom basketball statistics data
  - `processed/` - Generated datasets
- `src/` - Source code
  - `data_processing/` - Data loading modules
  - `feature_engineering/` - Feature creation modules
  - `modeling/` - Model training and prediction modules
  - `evaluation/` - Model evaluation modules
- `features/` - Saved feature sets
- `models/` - Trained models
- `submissions/` - Generated submission files
- `evaluation/` - Evaluation results

## Step-by-Step Guide

### 1. Setup Project Environment

#### 1.1. Create required directories
```bash
mkdir -p data/kaggle-data data/kenpom data/processed features models submissions evaluation
```

#### 1.2. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fuzzywuzzy python-Levenshtein
```

### 2. Prepare Data

#### 2.1. Download Kaggle Competition Data
1. Go to the Kaggle March Madness competition page
2. Download the competition data files
3. Place all CSV files in the `data/kaggle-data/` directory

#### 2.2. Prepare KenPom Data (Optional)
1. Place your KenPom data file at `data/kenpom/historical_kenpom_data.csv`
2. Run the team name matcher to create mappings:
   ```bash
   python team_name_matcher.py
   ```
   This will:
   - Match KenPom team names to Kaggle TeamIDs
   - Create a mapping file at `data/kenpom/team_mapping.csv`
   - Create an enhanced KenPom file with TeamIDs at `data/kenpom/enhanced_kenpom_data.csv`

### 3. Run the Pipeline

#### 3.1. Full Pipeline
To run the complete pipeline (data loading, feature engineering, model training, prediction):

```bash
python run_pipeline.py
```

Options:
- `--no-kenpom` - Skip using KenPom data
- `--prepare-kenpom` - Run KenPom preprocessing before the main pipeline
- `--season YEAR` - Specify the tournament season to predict (default: 2025)

Example:
```bash
python run_pipeline.py --prepare-kenpom --season 2025
```

#### 3.2. Prediction Only
If you've already trained models and just want to generate predictions:

```bash
python generate_predictions.py
```

This will:
1. Load the trained ensemble model
2. Build features for the current season teams
3. Create matchups
4. Generate predictions
5. Save a submission file to `submissions/final_submission.csv`

### 4. Verify Submission

To check if your submission complies with the Kaggle format requirements:

```bash
python check_submission.py --submission submissions/ensemble_submission.csv
```

This will:
1. Verify the ID format (SSSS_XXXX_YYYY)
2. Ensure Team1 ID is less than Team2 ID
3. Check that predictions are within the [0,1] range
4. Verify no mixed gender matchups exist
5. Generate visualizations of the prediction distributions

### 5. Understanding the Competition

#### 5.1. Submission Format
As per Kaggle requirements:
- Each game has a unique ID: "SSSS_XXXX_YYYY"
  - SSSS is the season (2025)
  - XXXX is the lower-ID team
  - YYYY is the higher-ID team
- You must predict the probability (0-1) that the team with the lower ID wins
- Men's teams have IDs 1000-1999
- Women's teams have IDs 3000-3999
- No matchups between men's and women's teams

#### 5.2. Evaluation
- Submissions are evaluated using the Brier score
- Lower scores are better
- The Brier score measures the mean squared error between predictions and actual outcomes

### 6. Troubleshooting

#### 6.1. KenPom Integration Issues
If you're having issues with KenPom data:
1. Run `python examine_kenpom.py` to better understand the data structure
2. Run `python team_name_matcher.py` to create proper mappings
3. If problems persist, use the `--no-kenpom` flag with `run_pipeline.py`

#### 6.2. Missing Dependencies
If you encounter import errors:
```bash
pip install fuzzywuzzy python-Levenshtein pandas numpy scikit-learn matplotlib seaborn
```

#### 6.3. Memory Issues
If the pipeline uses too much memory:
1. Reduce the number of seasons used for training
2. Reduce the number of features used in model training
