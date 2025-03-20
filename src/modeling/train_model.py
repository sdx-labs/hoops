import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

def prepare_model_data(matchups_df, feature_columns=None, target_col='Target'):
    """
    Prepare data for model training
    """
    # If no feature columns provided, use all columns except ID, Season, Team1, Team2 and Target
    if feature_columns is None:
        feature_columns = [col for col in matchups_df.columns 
                        if col not in ['ID', 'Season', 'Team1', 'Team2', 'Target']]
    
    # Drop rows with missing values
    clean_df = matchups_df.dropna(subset=feature_columns + [target_col])
    
    # Split into features and target
    X = clean_df[feature_columns]
    y = clean_df[target_col]
    
    return X, y

def select_features(X, y, n_features=50, model=None):
    """
    Select most important features using a random forest
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top n_features
    selected_features = X.columns[indices[:n_features]]
    
    return selected_features, importances

def train_model(X, y, model_type='rf', param_grid=None, cv=5):
    """
    Train a model using cross-validation
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - model_type: 'rf' (Random Forest), 'gbm' (Gradient Boosting), 'lr' (Logistic Regression)
    - param_grid: parameters for grid search
    - cv: number of cross-validation folds
    
    Returns:
    - best_model: trained model
    - cv_results: cross-validation results
    """
    # Define default pipeline with preprocessing
    pipeline_steps = [('scaler', StandardScaler())]
    
    # Set up model with default parameters
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        pipeline_steps.append(('model', model))
        if param_grid is None:
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(random_state=42)
        pipeline_steps.append(('model', model))
        if param_grid is None:
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [3, 5, 7]
            }
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
        pipeline_steps.append(('model', model))
        if param_grid is None:
            param_grid = {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(pipeline_steps)
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_}")
    
    return grid_search.best_estimator_, grid_search.cv_results_

def evaluate_model(model, X, y):
    """
    Evaluate a trained model
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    log_loss_score = log_loss(y, y_pred_proba)
    brier = brier_score_loss(y, y_pred_proba)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {log_loss_score:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    return {
        'accuracy': accuracy,
        'log_loss': log_loss_score,
        'brier_score': brier
    }

def save_model(model, output_path, model_name):
    """
    Save trained model to disk
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_path}")

def load_model(model_path):
    """
    Load model from disk
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model

def train_ensemble(X, y, models=['rf', 'gbm', 'lr'], weights=None):
    """
    Train multiple models and create an ensemble
    """
    trained_models = {}
    cv_results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        model, cv = train_model(X, y, model_type)
        trained_models[model_type] = model
        cv_results[model_type] = cv
    
    # Set equal weights if not provided
    if weights is None:
        weights = {model: 1/len(models) for model in models}
    
    # Create ensemble model class
    class EnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        
        def predict_proba(self, X):
            predictions = []
            sum_weights = sum(self.weights.values())
            
            for model_type, model in self.models.items():
                weight = self.weights.get(model_type, 1)
                pred = model.predict_proba(X)[:, 1] * (weight / sum_weights)
                predictions.append(pred)
            
            # Weighted average
            ensemble_pred = np.sum(predictions, axis=0)
            # Return in required format for sklearn (both classes)
            return np.vstack((1 - ensemble_pred, ensemble_pred)).T
        
        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs > 0.5).astype(int)
    
    ensemble = EnsembleModel(trained_models, weights)
    
    # Evaluate ensemble
    print("\nEnsemble model performance:")
    ensemble_metrics = evaluate_model(ensemble, X, y)
    
    return ensemble, trained_models, ensemble_metrics, cv_results
