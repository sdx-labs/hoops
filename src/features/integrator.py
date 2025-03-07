import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import our data collectors
from src.data.womens_basketball_data import WomensBasketballDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegrator:
    def __init__(self, features_dir="data/features", external_data_dir="data/external", output_dir="data/integrated"):
        self.features_dir = Path(features_dir)
        self.external_data_dir = Path(external_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collectors
        self.womens_data_collector = WomensBasketballDataCollector(
            data_dir=self.external_data_dir / "womens"
        )
        
    def get_available_data_sources(self):
        """List available data sources for integration"""
        sources = []
        
        # Check for base features
        if (self.features_dir / "combined_matchup_features.csv").exists():
            sources.append("base_features")
            
        # Check for KenPom data
        if (self.external_data_dir / "kenpom_historical_ratings.csv").exists() or \
           (self.external_data_dir / "kenpom_sample_data.csv").exists():
            sources.append("kenpom")
            
        # Check for women's data
        womens_path = self.external_data_dir / "womens" / "womens_historical_ratings.csv"
        if womens_path.exists():
            sources.append("womens_ratings")
            
        # Check for NCAA stats for women
        if list(Path(self.external_data_dir / "womens").glob("ncaa_sample_stats_*.csv")):
            sources.append("ncaa_womens_stats")
            
        # Check for Her Hoop Stats data
        if list(Path(self.external_data_dir / "womens").glob("her_hoop_stats_sample_*.csv")):
            sources.append("her_hoop_stats")
            
        # Check for Sports Reference data
        if list(Path(self.external_data_dir / "womens").glob("sports_reference_sample_*.csv")):
            sources.append("sports_reference")
            
        logger.info(f"Found {len(sources)} available data sources: {sources}")
        return sources
        
    def integrate_data_sources(self, selected_sources=None):
        """Integrate selected data sources"""
        available_sources = self.get_available_data_sources()
        
        if selected_sources is None:
            selected_sources = available_sources
            
        if not selected_sources:
            logger.warning("No data sources available for integration")
            return None
            
        # Start with base features if available
        if "base_features" in selected_sources and "base_features" in available_sources:
            base_features_path = self.features_dir / "combined_matchup_features.csv"
            integrated_df = pd.read_csv(base_features_path)
            logger.info(f"Loaded base features with {len(integrated_df)} rows")
        else:
            logger.error("Base features not available - required for integration")
            return None
        
        # Process the data by gender
        m_features = integrated_df[integrated_df['Gender'] == 'M'].copy()
        w_features = integrated_df[integrated_df['Gender'] == 'W'].copy()
        
        logger.info(f"Split features by gender: {len(m_features)} men's, {len(w_features)} women's")
        
        # Process men's data sources
        if "kenpom" in selected_sources and "kenpom" in available_sources:
            try:
                # KenPom integration should already be done in the feature engineering step
                # but we can check if the columns exist and log them
                kenpom_columns = ['Team1_AdjO', 'Team1_AdjD', 'Team1_AdjTempo', 'Team1_Luck',
                                'Team2_AdjO', 'Team2_AdjD', 'Team2_AdjTempo', 'Team2_Luck']
                                
                kenpom_cols_present = [col for col in kenpom_columns if col in m_features.columns]
                logger.info(f"KenPom features present: {len(kenpom_cols_present)}/{len(kenpom_columns)}")
                
                # If needed, we could add additional KenPom-derived features here
                
            except Exception as e:
                logger.error(f"Error checking KenPom features: {str(e)}")
        
        # Process women's data sources
        womens_sources = ["womens_ratings", "ncaa_womens_stats", "her_hoop_stats", "sports_reference"]
        available_womens_sources = [s for s in womens_sources if s in selected_sources and s in available_sources]
        
        if available_womens_sources:
            try:
                # Similar to KenPom, these should already be integrated in feature engineering
                # but we can check if the columns exist
                womens_columns = ['Team1_OffensiveRating', 'Team1_DefensiveRating', 
                                 'Team2_OffensiveRating', 'Team2_DefensiveRating']
                                 
                womens_cols_present = [col for col in womens_columns if col in w_features.columns]
                logger.info(f"Women's advanced metrics present: {len(womens_cols_present)}/{len(womens_columns)}")
                
                # Check for any missing advanced metrics and try to add them
                if len(womens_cols_present) < len(womens_columns):
                    logger.info("Some women's advanced metrics are missing, attempting to add them")
                    # Here we could add logic to merge in missing metrics
                    
            except Exception as e:
                logger.error(f"Error processing women's data sources: {str(e)}")
        
        # Recombine the processed dataframes
        integrated_df = pd.concat([m_features, w_features], ignore_index=True)
        
        # Generate integrated feature importance report
        try:
            # This would analyze which features are most important for each gender
            self._generate_feature_importance_report(integrated_df)
        except Exception as e:
            logger.error(f"Error generating feature importance report: {str(e)}")
        
        # Save integrated dataset
        output_path = self.output_dir / "integrated_features.csv"
        integrated_df.to_csv(output_path, index=False)
        logger.info(f"Saved integrated features to {output_path}")
        
        return integrated_df
    
    def _generate_feature_importance_report(self, df):
        """Generate a report on feature importance by gender"""
        # This would use a simple model to determine feature importance
        # For now, we'll just log the number of features
        m_features = df[df['Gender'] == 'M']
        w_features = df[df['Gender'] == 'W']
        
        logger.info(f"Men's features dataset: {len(m_features)} rows, {m_features.shape[1]} columns")
        logger.info(f"Women's features dataset: {len(w_features)} rows, {w_features.shape[1]} columns")
        
        # In a real implementation, this would fit a model and extract feature importances
        
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        # 1. Ensure women's basketball data is available
        self.womens_data_collector.create_derived_power_ratings()
        
        # 2. Generate women's ratings for recent seasons
        current_year = datetime.now().year
        for season in range(current_year - 5, current_year + 1):
            self.womens_data_collector.get_womens_ratings(season)
            
        # 3. Integrate all available data sources
        integrated_data = self.integrate_data_sources()
        
        return integrated_data

if __name__ == "__main__":
    integrator = DataIntegrator()
    integrated_data = integrator.integrate_data_sources()
    
    # Check distribution of features by gender
    if integrated_data is not None:
        gender_counts = integrated_data['Gender'].value_counts()
        logger.info(f"Data distribution by gender: {gender_counts.to_dict()}")
