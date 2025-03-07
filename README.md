# March Machine Learning Mania 2025

This project aims to predict the outcomes of both the men's and women's 2025 collegiate basketball tournaments for the Kaggle Machine Learning Mania competition.

## Project Setup

1. Initialize the project structure:
```bash
python setup_project.py
```

2. Download competition data:
```bash
python -m src.data.download
```

3. Preprocess the data:
```bash
python -m src.data.preprocess
```

## Project Structure

- `config.yaml`: Project configuration
- `data/`: Data directory
  - `raw/`: Original competition data
  - `processed/`: Cleaned data
  - `features/`: Engineered features
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks for exploration and development
- `src/`: Source code
  - `data/`: Data processing scripts
  - `features/`: Feature engineering
  - `models/`: Model training and prediction

## Modeling Approach

The competition requires predicting the probability that one team beats another for all possible matchups in the tournaments. Our approach will include:

1. **Data Understanding**: Analyze historical tournament data and team statistics
2. **Feature Engineering**: Create informative features from the raw data
3. **Model Development**: Train and tune models to predict game outcomes
4. **Submission Generation**: Create predictions for all possible matchups

## Evaluation

Submissions are evaluated using the Brier score (mean squared error) between predicted probabilities and actual outcomes.

## Competition Timeline

- February 10, 2025: Start Date
- February 18-21, 2025: 2025 Tournament Submission File Available
- March 20, 2025 4PM UTC: Final Submission Deadline
- March 20 - April 8, 2025: Tournament results play out

## Next Steps

1. Explore the data in `notebooks/01_data_exploration.ipynb`
2. Develop features in `notebooks/02_feature_engineering.ipynb`
3. Train baseline models in `notebooks/03_model_development.ipynb`
4. Generate submission file in `notebooks/04_submission_generation.ipynb`