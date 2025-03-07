import logging
import argparse
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import BasketballDataPreprocessor
from src.features.engineer import BasketballFeatureEngineer
from src.models.train_model import BasketballModelTrainer
from src.models.predict_model import BasketballPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run the full basketball prediction pipeline')
    parser.add_argument('--steps', nargs='+', default=['all'], 
                        choices=['preprocess', 'features', 'train', 'predict', 'all'],
                        help='Pipeline steps to run')
    parser.add_argument('--force', action='store_true', help='Force rerun of all steps even if data exists')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble for prediction')
    parser.add_argument('--output', help='Custom filename for submission file')
    parser.add_argument('--no-download', action='store_true', help='Skip downloading data even if missing')
    return parser.parse_args()

def run_preprocessing(force=False, download=True):
    """Run preprocessing step"""
    logger.info("Starting data preprocessing")
    
    # Check if processed data already exists
    processed_dir = Path("data/processed")
    if not force and processed_dir.exists() and list(processed_dir.glob("*.csv")):
        logger.info("Processed data already exists. Use --force to reprocess.")
        return True
    
    # Run preprocessing
    preprocessor = BasketballDataPreprocessor()
    result = preprocessor.process_all_data(download=download, force_download=force)
    
    if result:
        logger.info("Preprocessing completed successfully")
        return True
    else:
        logger.error("Preprocessing failed")
        return False

def run_feature_engineering(force=False):
    """Run feature engineering step"""
    logger.info("Starting feature engineering")
    
    # Check if features already exist
    features_dir = Path("data/features")
    if not force and features_dir.exists() and list(features_dir.glob("*matchup_features.csv")):
        logger.info("Features already exist. Use --force to regenerate.")
        return True
    
    # Run feature engineering
    engineer = BasketballFeatureEngineer()
    result = engineer.process_all_features()
    
    if result:
        logger.info("Feature engineering completed successfully")
        return True
    else:
        logger.error("Feature engineering failed")
        return False

def run_model_training(force=False):
    """Run model training step"""
    logger.info("Starting model training")
    
    # Check if models already exist
    models_dir = Path("models")
    if not force and models_dir.exists() and list(models_dir.glob("*.pkl")):
        logger.info("Models already exist. Use --force to retrain.")
        return True
    
    # Run training
    trainer = BasketballModelTrainer()
    result = trainer.run_model_training_pipeline()
    
    if result:
        logger.info("Model training completed successfully")
        return True
    else:
        logger.error("Model training failed")
        return False

def run_prediction(ensemble=False, output=None):
    """Run prediction step"""
    logger.info("Starting prediction")
    
    # Run prediction
    predictor = BasketballPredictor()
    result = predictor.run_prediction_pipeline(
        ensemble=ensemble,
        filename=output
    )
    
    if result is not None:
        logger.info("Prediction completed successfully")
        return True
    else:
        logger.error("Prediction failed")
        return False

def run_full_pipeline(args):
    """Run the full pipeline or selected steps"""
    steps = args.steps
    
    # If 'all' is selected, run all steps
    run_all = 'all' in steps
    
    # Run preprocessing
    if run_all or 'preprocess' in steps:
        if not run_preprocessing(args.force, not args.no_download):
            return False
    
    # Run feature engineering
    if run_all or 'features' in steps:
        if not run_feature_engineering(args.force):
            return False
    
    # Run model training
    if run_all or 'train' in steps:
        if not run_model_training(args.force):
            return False
    
    # Run prediction
    if run_all or 'predict' in steps:
        if not run_prediction(args.ensemble, args.output):
            return False
    
    return True

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run the pipeline
    if run_full_pipeline(args):
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        sys.exit(1)
