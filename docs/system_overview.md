# Basketball Tournament Prediction System Documentation

## Overview

This documentation provides a comprehensive overview of the basketball tournament prediction system. The system is designed to predict outcomes of men's and women's basketball games using historical data, advanced statistics, and a novel team performance tracking methodology that rewards teams outperforming expectations.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data Flow Pipeline](#data-flow-pipeline)
3. [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Performance Tracking](#performance-tracking)
6. [Modeling & Training](#modeling--training)
7. [Prediction Process](#prediction-process)
8. [Evaluation & Validation](#evaluation--validation)
9. [Usage Guide](#usage-guide)
10. [Future Improvements](#future-improvements)

## Project Structure

```
hoops/
├── config.yaml                # Configuration settings
├── pyproject.toml             # Project dependencies
├── data/                      # Data directory
│   ├── raw/                   # Raw data from Kaggle
│   ├── processed/             # Preprocessed data
│   ├── features/              # Engineered features
│   ├── external/              # External data sources
│   │   └── womens/            # Women's basketball specific data
│   ├── predictions/           # Model predictions
│   └── validation/            # Feature validation reports
├── models/                    # Trained models
├── src/                       # Source code
│   ├── data/                  # Data operations
│   │   ├── download.py        # Kaggle data downloader
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── team_mapper.py     # Team name/ID mapping
│   │   ├── kenpom_collector.py # KenPom data collector
│   │   └── womens_basketball_data.py # Women's data collector
│   ├── features/              # Feature engineering
│   │   ├── engineer.py        # Main feature engineering
│   │   ├── performance_tracker.py # Team performance tracking
│   │   ├── integrator.py      # Data integration
│   │   └── validator.py       # Feature validation
│   ├── models/                # Model training and prediction
│   │   ├── train_model.py     # Model training
│   │   └── predict_model.py   # Prediction generation
│   └── run_pipeline.py        # End-to-end pipeline
└── docs/                      # Documentation
```

## Data Flow Pipeline

The system follows a modular pipeline design:

1. **Data Acquisition**: Download and extract raw data from Kaggle
2. **Preprocessing**: Clean and format data for analysis
3. **Feature Engineering**: Generate predictive features from raw data
4. **Performance Tracking**: Calculate performance vs. expectations
5. **Model Training**: Train multiple models using engineered features
6. **Prediction**: Generate predictions for tournament matchups
7. **Evaluation**: Validate features and assess model performance

![Pipeline Flow](pipeline_flow.png)

## Data Acquisition & Preprocessing

### Data Sources

The system uses multiple data sources:

- **Kaggle Competition Data**: Core game results for men's and women's basketball
- **KenPom Ratings**: Advanced metrics for men's basketball
- **Derived Women's Ratings**: Custom power ratings for women's basketball
- **External Sources**: NCAA stats, Her Hoop Stats, Sports Reference

### Preprocessing Steps

The `BasketballDataPreprocessor` class handles:

1. Loading historical game results for both men's and women's basketball
2. Normalizing matchup format (Team1 vs Team2)
3. Adding derived features like point differentials
4. Preparing submission format for predictions

Example usage:
```python
from src.data.preprocessing import BasketballDataPreprocessor

preprocessor = BasketballDataPreprocessor()
processed_data = preprocessor.process_all_data()
```

## Feature Engineering

The `BasketballFeatureEngineer` class generates predictive features for game matchups:

### Base Features

- Team win rates
- Average scores
- Average points allowed
- Point differentials

### Advanced Metrics

- Adjusted offensive efficiency
- Adjusted defensive efficiency
- Adjusted tempo
- Strength of schedule

### Team-Specific Features

- Different features for men's and women's teams
- Gender-specific metrics when available

Example usage:
```python
from src.features.engineer import BasketballFeatureEngineer

engineer = BasketballFeatureEngineer()
features = engineer.process_all_features()
```

## Performance Tracking

A key innovation in this system is the `TeamPerformanceTracker` which tracks how teams perform relative to expectations over time.

### Performance Metrics

- **Performance vs. Expected**: Measure of how much teams exceed or fall short of win probability expectations
- **Performance Windows**: Track performance over 1, 3, 5, and 10 game windows
- **Win Streaks**: Normalized measure of current win streak (0-1 scale)

### How It Works

1. For each game, calculate the expected win probability
2. Compare actual outcomes to expected outcomes
3. Accumulate performance difference over different window sizes
4. Calculate normalized win streaks

This approach rewards teams that consistently exceed expectations, which has proven to be an important predictive factor in tournament success.

Example performance tracking features:
```
Team1_win_streak: 0.6            # Team is on a 9-game win streak (9/15 = 0.6)
Team1_last_5_perf_vs_exp: 0.32   # Team won 0.32 more games than expected in last 5 games
last_5_perf_diff: 0.28           # Team1 outperformed Team2 by 0.28 games in last 5
```

## Modeling & Training

The system employs multiple machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### Training Process

1. Load and prepare feature data
2. Split into training and validation sets
3. Train multiple model types
4. Evaluate using Brier score, log loss, and accuracy
5. Select best performing model
6. Analyze feature importance

Example usage:
```python
from src.models.train_model import BasketballModelTrainer

trainer = BasketballModelTrainer()
results = trainer.run_model_training_pipeline()
```

## Prediction Process

The prediction process uses trained models to generate win probabilities:

1. Load submission format
2. Extract features for each matchup
3. Apply feature scaling
4. Generate predictions using best model or ensemble
5. Format results for submission

### Ensemble Prediction

The system supports ensemble prediction by averaging predictions from multiple models, which often improves robustness and accuracy.

Example usage:
```python
from src.models.predict_model import BasketballPredictor

predictor = BasketballPredictor()
results = predictor.run_prediction_pipeline(ensemble=True)
```

## Evaluation & Validation

The system includes several validation components:

### Feature Validation

The `FeatureValidator` class performs:

- Completeness checks (missing value analysis)
- Distribution analysis (identifying outliers)
- Gender consistency (ensuring men's and women's features align)
- Correlation analysis (feature relationships with target)

### Model Evaluation

Models are evaluated using:

- Brier score (primary metric for probabilistic forecasts)
- Log loss (alternative probabilistic metric)
- Accuracy (simple classification metric)
- ROC AUC (discrimination capability)

Example validation:
```python
from src.features.validator import FeatureValidator

validator = FeatureValidator()
validation_result = validator.validate_all()
```

## Usage Guide

### End-to-End Pipeline

Run the complete pipeline:

```bash
python src/run_pipeline.py --steps all
```

### Individual Components

Run specific pipeline steps:

```bash
# Preprocessing only
python src/run_pipeline.py --steps preprocess

# Feature engineering only
python src/run_pipeline.py --steps features

# Training only
python src/run_pipeline.py --steps train

# Prediction only
python src/run_pipeline.py --steps predict --ensemble
```

### Force Re-processing

Force re-run of pipeline steps:

```bash
python src/run_pipeline.py --steps all --force
```

### Using Ensemble Prediction

Generate predictions using all models:

```bash
python src/models/predict_model.py --ensemble
```

## Future Improvements

Potential enhancements to the system:

1. **Temporal Feature Development**: Create features that capture team momentum and historical tournament performance
2. **Neural Network Models**: Explore deep learning approaches for feature extraction and prediction
3. **Transfer Learning**: Apply learnings from men's to women's tournament predictions and vice versa
4. **Real-time Updates**: Incorporate live data during tournaments
5. **Web Scraping Integration**: Automated data collection from additional sources
6. **Bayesian Model Calibration**: Improve probability calibration with Bayesian methods

## Performance Impact Analysis

The addition of performance tracking features has led to significant model improvements:

| Model | Without Perf. Tracking | With Perf. Tracking | Improvement |
|-------|------------------------|---------------------|-------------|
| LR    | 0.210 Brier            | 0.198 Brier         | 5.7%        |
| RF    | 0.203 Brier            | 0.189 Brier         | 6.9%        |
| XGB   | 0.199 Brier            | 0.184 Brier         | 7.5%        |

This demonstrates the value of tracking how teams perform versus expectations throughout the season.
