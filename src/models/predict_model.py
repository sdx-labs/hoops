import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasketballPredictor:
    def __init__(self, features_dir="data/features", models_dir="models", output_dir="data/predictions"):
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_submission_features(self):
        """Load submission features for prediction"""
        file_path = self.features_dir / "submission_features.csv"
        if not file_path.exists():
            logger.error("Submission features file not found")
            return None
            
        df = pd.read_csv(file_path)
        logger.info(f"Loaded submission features with {len(df)} rows")
        return df
        
    def load_model(self, model_name="best_model"):
        """Load trained model for prediction"""
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        logger.info(f"Loaded model: {model_name}")
        return model
        
    def load_scaler(self):
        """Load feature scaler"""
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            logger.warning("Scaler file not found, will not scale features")
            return None
            
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
        logger.info("Loaded feature scaler")
        return scaler
        
    def prepare_features(self, features_df, scaler=None):
        """Prepare features for prediction"""
        if features_df is None or len(features_df) == 0:
            logger.error("No feature data provided")
            return None
            
        # Define features to use (must match what was used in training)
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
        
        # Performance metrics from our tracker
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
        
        # Check for recent win tracking
        recent_win_features = [
            'Team1_RecentWins', 'Team2_RecentWins', 'RecentWins_Diff'
        ]
        
        # Combine all feature types
        all_features = basic_features.copy()
        
        # Include any available advanced features
        available_advanced = [f for f in advanced_features if f in features_df.columns]
        if available_advanced:
            all_features.extend(available_advanced)
            
        # Include any available performance metrics
        available_performance = [f for f in performance_features if f in features_df.columns]
        if available_performance:
            all_features.extend(available_performance)
        
        # Include any available recent win features
        available_win_features = [f for f in recent_win_features if f in features_df.columns]
        if available_win_features:
            all_features.extend(available_win_features)
            
        # Check if gender-specific features should be included
        if 'Gender' in features_df.columns:
            features_df['is_men'] = (features_df['Gender'] == 'M').astype(int)
            if 'is_men' in features_df.columns:
                all_features.append('is_men')
        
        # Ensure all selected features exist in the dataframe
        actual_features = [f for f in all_features if f in features_df.columns]
        logger.info(f"Using {len(actual_features)} features for prediction")
        
        # Select features
        X = features_df[actual_features].copy()
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Apply scaling if scaler is provided
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                logger.info("Applied feature scaling")
            except Exception as e:
                logger.error(f"Error applying scaler: {str(e)}")
                logger.info("Using unscaled features")
        
        return X
        
    def make_predictions(self, model_name="best_model", ensemble=False):
        """
        Make predictions using the specified model
        
        Args:
            model_name: Name of the model to use ("best_model" or specific model name)
            ensemble: Whether to use ensemble of multiple models
            
        Returns:
            DataFrame with predictions
        """
        # Load submission features
        features_df = self.load_submission_features()
        if features_df is None:
            return None
            
        # Load scaler
        scaler = self.load_scaler()
        
        # Load model(s)
        if ensemble:
            # Use an ensemble of models
            models = {}
            for model_file in self.models_dir.glob("*_model.pkl"):
                if "best_model" not in model_file.name:  # Skip the best model file
                    model_name = model_file.stem.replace("_model", "")
                    model = self.load_model(model_name)
                    if model is not None:
                        models[model_name] = model
                        
            if not models:
                logger.error("No models found for ensemble")
                return None
                
            logger.info(f"Using ensemble of {len(models)} models: {', '.join(models.keys())}")
        else:
            # Use a single model
            model = self.load_model(model_name)
            if model is None:
                return None
        
        # Prepare features
        X = self.prepare_features(features_df, scaler)
        if X is None:
            return None
            
        # Make predictions
        try:
            if ensemble:
                # Combine predictions from multiple models
                all_preds = []
                for name, model in models.items():
                    preds = model.predict_proba(X)[:, 1]
                    all_preds.append(preds)
                    logger.info(f"Made predictions with {name} model")
                    
                # Average predictions
                predictions = np.mean(all_preds, axis=0)
                logger.info("Combined ensemble predictions")
            else:
                # Single model prediction
                predictions = model.predict_proba(X)[:, 1]
                logger.info(f"Made predictions with {model_name} model")
                
            # Create results DataFrame
            results_df = pd.DataFrame({
                'ID': features_df['ID'],
                'Pred': predictions
            })
            
            logger.info(f"Generated {len(results_df)} predictions")
            return results_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def save_submission(self, results_df, filename=None):
        """
        Save predictions in submission format
        
        Args:
            results_df: DataFrame with ID and Pred columns
            filename: Custom filename (default: submission_YYYYMMDD_HHMMSS.csv)
        """
        if results_df is None or len(results_df) == 0:
            logger.error("No predictions to save")
            return False
            
        if 'ID' not in results_df.columns or 'Pred' not in results_df.columns:
            logger.error("Results dataframe must contain 'ID' and 'Pred' columns")
            return False
            
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"
            
        # Ensure it has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Save submission
        output_path = self.output_dir / filename
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved submission to {output_path}")
        
        return True
        
    def analyze_predictions(self, results_df):
        """
        Analyze prediction distribution and identify potential issues
        
        Args:
            results_df: DataFrame with predictions
        """
        if results_df is None or len(results_df) == 0 or 'Pred' not in results_df.columns:
            logger.error("No valid predictions to analyze")
            return
            
        predictions = results_df['Pred']
        
        # Calculate statistics
        stats = {
            'count': len(predictions),
            'mean': predictions.mean(),
            'median': predictions.median(),
            'min': predictions.min(),
            'max': predictions.max(),
            'std': predictions.std()
        }
        
        # Count predictions in each range
        ranges = {
            '0.0-0.1': ((predictions >= 0.0) & (predictions < 0.1)).sum(),
            '0.1-0.2': ((predictions >= 0.1) & (predictions < 0.2)).sum(),
            '0.2-0.3': ((predictions >= 0.2) & (predictions < 0.3)).sum(),
            '0.3-0.4': ((predictions >= 0.3) & (predictions < 0.4)).sum(),
            '0.4-0.5': ((predictions >= 0.4) & (predictions < 0.5)).sum(),
            '0.5-0.6': ((predictions >= 0.5) & (predictions < 0.6)).sum(),
            '0.6-0.7': ((predictions >= 0.6) & (predictions < 0.7)).sum(),
            '0.7-0.8': ((predictions >= 0.7) & (predictions < 0.8)).sum(),
            '0.8-0.9': ((predictions >= 0.8) & (predictions < 0.9)).sum(),
            '0.9-1.0': ((predictions >= 0.9) & (predictions <= 1.0)).sum()
        }
        
        # Log statistics
        logger.info("Prediction Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("Prediction Distribution:")
        for range_name, count in ranges.items():
            pct = count / stats['count'] * 100
            logger.info(f"  {range_name}: {count} ({pct:.1f}%)")
            
        # Check for potential issues
        if stats['min'] < 0 or stats['max'] > 1:
            logger.warning("Predictions outside valid probability range [0,1]")
            
        if stats['std'] < 0.05:
            logger.warning("Low prediction variance, may indicate underfitting")
            
        if ranges['0.4-0.6'] / stats['count'] > 0.5:
            logger.warning("More than 50% of predictions are near 0.5, model may lack confidence")
        
        # Save analysis report
        report_path = self.output_dir / "prediction_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(f"Prediction Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("Prediction Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
                
            f.write("\nPrediction Distribution:\n")
            for range_name, count in ranges.items():
                pct = count / stats['count'] * 100
                f.write(f"  {range_name}: {count} ({pct:.1f}%)\n")
        
        logger.info(f"Saved prediction analysis to {report_path}")
    
    def run_prediction_pipeline(self, model_name="best_model", ensemble=False, filename=None):
        """Run the complete prediction pipeline"""
        # Make predictions
        results_df = self.make_predictions(model_name, ensemble)
        
        # Save submission
        if results_df is not None:
            self.save_submission(results_df, filename)
            
            # Analyze predictions
            self.analyze_predictions(results_df)
            
        return results_df

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate basketball tournament predictions')
    parser.add_argument('--model', default='best_model', help='Model to use for predictions')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of all available models')
    parser.add_argument('--output', '-o', help='Output filename for submission')
    parser.add_argument('--no-download', action='store_true', help='Do not download data even if missing')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predictor = BasketballPredictor()
    results = predictor.run_prediction_pipeline(
        model_name=args.model,
        ensemble=args.ensemble,
        filename=args.output
    )
    
    print(f"Prediction process completed with {len(results) if results is not None else 0} predictions")
