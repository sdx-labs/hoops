import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasketballModelTrainer:
    def __init__(self, features_dir="data/features", models_dir="models"):
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_data(self):
        """Load feature data for training"""
        try:
            file_path = self.features_dir / "combined_matchup_features.csv"
            if not file_path.exists():
                logger.error("Combined features file not found")
                return None
                
            features_df = pd.read_csv(file_path)
            logger.info(f"Loaded training data with {len(features_df)} rows")
            return features_df
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return None
            
    def prepare_training_data(self, features_df):
        """
        Prepare data for training by splitting into X and y, and handling missing values
        """
        if features_df is None or len(features_df) == 0:
            logger.error("No feature data provided")
            return None, None
        
        # Define features to use
        basic_features = [
            'Team1_WinRate', 'Team2_WinRate',
            'Team1_AvgScore', 'Team2_AvgScore',
            'Team1_AvgAllowed', 'Team2_AvgAllowed',
            'Team1_AvgPointDiff', 'Team2_AvgPointDiff',
            'WinRate_Diff', 'AvgScore_Diff', 'AvgAllowed_Diff', 'AvgPointDiff_Diff'
        ]
        
        advanced_features = [
            'Team1_AdjO', 'Team2_AdjO', 'Team1_AdjD', 'Team2_AdjD',
            'Team1_AdjTempo', 'Team2_AdjTempo', 'Team1_Luck', 'Team2_Luck',
            'AdjO_Diff', 'AdjD_Diff'
        ]
        
        # Add performance metrics from our new tracker
        performance_features = [
            'Team1_win_streak', 'Team2_win_streak', 'win_streak_diff',
            'Team1_last_1_perf_vs_exp', 'Team2_last_1_perf_vs_exp', 'last_1_perf_diff',
            'Team1_last_3_perf_vs_exp', 'Team2_last_3_perf_vs_exp', 'last_3_perf_diff',
            'Team1_last_5_perf_vs_exp', 'Team2_last_5_perf_vs_exp', 'last_5_perf_diff',
            'Team1_last_10_perf_vs_exp', 'Team2_last_10_perf_vs_exp', 'last_10_perf_diff',
            'Team1_last_1_win_pct', 'Team2_last_1_win_pct',
            'Team1_last_3_win_pct', 'Team2_last_3_win_pct',
            'Team1_last_5_win_pct', 'Team2_last_5_win_pct',
            'Team1_last_10_win_pct', 'Team2_last_10_win_pct'
        ]
        
        # Combine all feature types
        all_features = basic_features.copy()
        
        # Check which advanced features are present
        available_advanced = [f for f in advanced_features if f in features_df.columns]
        if available_advanced:
            all_features.extend(available_advanced)
            logger.info(f"Added {len(available_advanced)} advanced features")
            
        # Check which performance metrics are present
        available_performance = [f for f in performance_features if f in features_df.columns]
        if available_performance:
            all_features.extend(available_performance)
            logger.info(f"Added {len(available_performance)} performance features")
            
        # Check if gender-specific features should be included
        if 'Gender' in features_df.columns:
            features_df['is_men'] = (features_df['Gender'] == 'M').astype(int)
            all_features.append('is_men')
        
        # Ensure all selected features exist in the dataframe
        actual_features = [f for f in all_features if f in features_df.columns]
        logger.info(f"Using {len(actual_features)} features for training")
        
        # Select features and target
        X = features_df[actual_features].copy()
        y = features_df['Result'].copy()
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Save the scaler for later use
        with open(self.models_dir / "scaler.pkl", 'wb') as file:
            pickle.dump(scaler, file)
            
        logger.info("Prepared training data and saved scaler")
        return X_scaled, y, actual_features
        
    def train_models(self, X, y, models=None):
        """
        Train multiple models and select the best one
        
        Args:
            X: Feature matrix
            y: Target variable
            models: Dictionary of model instances to train
        """
        if X is None or y is None:
            logger.error("No valid training data provided")
            return None
            
        if models is None:
            # Default models to try
            models = {
                'logistic': LogisticRegression(max_iter=1000, C=1.0),
                'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                'gbm': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42)
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate each model
        results = {}
        best_score = float('inf')  # Lower is better for Brier score
        best_model_name = None
        best_model = None
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                brier = brier_score_loss(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                accuracy = accuracy_score(y_test, y_pred_proba > 0.5)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'brier_score': brier,
                    'log_loss': logloss,
                    'accuracy': accuracy,
                    'auc': auc
                }
                
                logger.info(f"{name} results: Brier={brier:.4f}, LogLoss={logloss:.4f}, Acc={accuracy:.4f}, AUC={auc:.4f}")
                
                # Update best model if this is better
                if brier < best_score:
                    best_score = brier
                    best_model_name = name
                    best_model = model
                    
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}")
                
        # Save all models
        for name, result in results.items():
            model_path = self.models_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as file:
                pickle.dump(result['model'], file)
            logger.info(f"Saved {name} model to {model_path}")
            
        # Save the best model separately
        if best_model is not None:
            best_model_path = self.models_dir / "best_model.pkl"
            with open(best_model_path, 'wb') as file:
                pickle.dump(best_model, file)
            logger.info(f"Best model ({best_model_name}) saved with Brier score: {best_score:.4f}")
            
        return results, best_model_name
    
    def analyze_feature_importance(self, model_name="best_model", feature_names=None):
        """
        Analyze feature importance for the specified model
        
        Args:
            model_name: Name of the model to analyze
            feature_names: List of feature names
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Extract feature importance (method depends on model type)
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Cannot extract feature importance from {model_name}")
            return None
            
        if importances is None or feature_names is None:
            return None
            
        # Ensure we have the right number of feature names
        if len(importances) != len(feature_names):
            logger.warning(f"Feature count mismatch: {len(importances)} vs {len(feature_names)}")
            return None
            
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        output_path = self.models_dir / f"{model_name}_feature_importance.csv"
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Saved feature importance to {output_path}")
        
        return importance_df
    
    def run_model_training_pipeline(self):
        """Run the complete model training pipeline"""
        # Load data
        features_df = self.load_training_data()
        
        # Prepare data for training
        X, y, feature_names = self.prepare_training_data(features_df)
        
        # Train models
        results, best_model = self.train_models(X, y)
        
        # Analyze feature importance
        importance = self.analyze_feature_importance(best_model, feature_names)
        
        return {
            'results': results,
            'best_model': best_model,
            'feature_importance': importance
        }

if __name__ == "__main__":
    trainer = BasketballModelTrainer()
    training_results = trainer.run_model_training_pipeline()
    
    # Print top features
    if training_results['feature_importance'] is not None:
        top_features = training_results['feature_importance'].head(10)
        print("\nTop 10 features:")
        for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
            print(f"{i+1}. {feature}: {importance:.4f}")
