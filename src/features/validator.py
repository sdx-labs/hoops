import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureValidator:
    def __init__(self, features_dir="data/features", output_dir="data/validation"):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_feature_data(self):
        """Load all feature data for validation"""
        try:
            # Load combined features
            combined_path = self.features_dir / "combined_matchup_features.csv"
            if not combined_path.exists():
                logger.error("Combined matchup features file not found")
                return None
                
            combined_features = pd.read_csv(combined_path)
            logger.info(f"Loaded combined features with {len(combined_features)} rows")
            
            # Split by gender
            m_features = combined_features[combined_features['Gender'] == 'M']
            w_features = combined_features[combined_features['Gender'] == 'W']
            
            logger.info(f"Split by gender: {len(m_features)} men's features, {len(w_features)} women's features")
            
            return {
                'combined': combined_features,
                'mens': m_features,
                'womens': w_features
            }
            
        except Exception as e:
            logger.error(f"Error loading feature data: {str(e)}")
            return None
    
    def validate_feature_completeness(self, features_dict=None):
        """
        Check for missing values in critical features
        """
        if features_dict is None:
            features_dict = self.load_feature_data()
            
        if features_dict is None:
            return False
            
        # Define critical features for each gender
        mens_critical_features = [
            'Team1_WinRate', 'Team2_WinRate', 
            'Team1_AdjO', 'Team1_AdjD', 'Team2_AdjO', 'Team2_AdjD'
        ]
        
        womens_critical_features = [
            'Team1_WinRate', 'Team2_WinRate',
            'Team1_AdjO', 'Team1_AdjD', 'Team2_AdjO', 'Team2_AdjD'
        ]
        
        # Check men's features
        m_features = features_dict['mens']
        m_missing = {}
        for feature in mens_critical_features:
            if feature in m_features.columns:
                missing_count = m_features[feature].isna().sum()
                missing_pct = missing_count / len(m_features) * 100
                m_missing[feature] = (missing_count, missing_pct)
            else:
                logger.warning(f"Critical men's feature missing: {feature}")
                m_missing[feature] = (len(m_features), 100.0)
        
        # Check women's features
        w_features = features_dict['womens']
        w_missing = {}
        for feature in womens_critical_features:
            if feature in w_features.columns:
                missing_count = w_features[feature].isna().sum()
                missing_pct = missing_count / len(w_features) * 100
                w_missing[feature] = (missing_count, missing_pct)
            else:
                logger.warning(f"Critical women's feature missing: {feature}")
                w_missing[feature] = (len(w_features), 100.0)
        
        # Log the results
        logger.info("Men's feature completeness:")
        for feature, (count, pct) in m_missing.items():
            logger.info(f"  {feature}: {count} missing ({pct:.2f}%)")
            
        logger.info("Women's feature completeness:")
        for feature, (count, pct) in w_missing.items():
            logger.info(f"  {feature}: {count} missing ({pct:.2f}%)")
            
        # Determine if validation passed
        mens_complete = all(pct < 5.0 for _, pct in m_missing.values())
        womens_complete = all(pct < 5.0 for _, pct in w_missing.values())
        
        # Generate missing values report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mens_features_complete': mens_complete,
            'womens_features_complete': womens_complete,
            'mens_missing': m_missing,
            'womens_missing': w_missing
        }
        
        # Save report
        report_path = self.output_dir / "missing_values_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Feature Completeness Report - {report['timestamp']}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Men's features complete: {report['mens_features_complete']}\n")
            f.write("Men's missing values:\n")
            for feature, (count, pct) in report['mens_missing'].items():
                f.write(f"  {feature}: {count} missing ({pct:.2f}%)\n")
            f.write("\n")
            
            f.write(f"Women's features complete: {report['womens_features_complete']}\n")
            f.write("Women's missing values:\n")
            for feature, (count, pct) in report['womens_missing'].items():
                f.write(f"  {feature}: {count} missing ({pct:.2f}%)\n")
            
        logger.info(f"Saved missing values report to {report_path}")
        
        return mens_complete and womens_complete
    
    def validate_feature_distributions(self, features_dict=None):
        """
        Check feature distributions and identify potential issues
        """
        if features_dict is None:
            features_dict = self.load_feature_data()
            
        if features_dict is None:
            return False
        
        # Numeric features to check
        numeric_features = [
            'Team1_WinRate', 'Team2_WinRate',
            'Team1_AvgScore', 'Team2_AvgScore',
            'Team1_AvgAllowed', 'Team2_AvgAllowed',
            'Team1_AdjO', 'Team2_AdjO',
            'Team1_AdjD', 'Team2_AdjD',
            'WinRate_Diff', 'AvgScore_Diff', 'AvgAllowed_Diff',
            'AdjO_Diff', 'AdjD_Diff'
        ]
        
        # Generate distribution statistics
        distribution_stats = {'mens': {}, 'womens': {}}
        
        # Check men's distributions
        m_features = features_dict['mens']
        for feature in numeric_features:
            if feature in m_features.columns:
                feature_data = m_features[feature].dropna()
                if len(feature_data) > 0:
                    distribution_stats['mens'][feature] = {
                        'mean': feature_data.mean(),
                        'median': feature_data.median(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'skew': feature_data.skew(),
                        'kurtosis': feature_data.kurtosis(),
                        'outliers': sum((feature_data - feature_data.mean()).abs() > 3 * feature_data.std())
                    }
        
        # Check women's distributions
        w_features = features_dict['womens']
        for feature in numeric_features:
            if feature in w_features.columns:
                feature_data = w_features[feature].dropna()
                if len(feature_data) > 0:
                    distribution_stats['womens'][feature] = {
                        'mean': feature_data.mean(),
                        'median': feature_data.median(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'skew': feature_data.skew(),
                        'kurtosis': feature_data.kurtosis(),
                        'outliers': sum((feature_data - feature_data.mean()).abs() > 3 * feature_data.std())
                    }
        
        # Generate distribution report
        report_path = self.output_dir / "distribution_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Feature Distribution Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            # Men's section
            f.write("Men's Feature Distributions:\n")
            f.write("-"*40 + "\n")
            for feature, stats in distribution_stats['mens'].items():
                f.write(f"{feature}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
                f.write("\n")
            
            # Women's section
            f.write("\nWomen's Feature Distributions:\n")
            f.write("-"*40 + "\n")
            for feature, stats in distribution_stats['womens'].items():
                f.write(f"{feature}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
                f.write("\n")
                
        logger.info(f"Saved distribution report to {report_path}")
        
        # Create visualization of distributions
        self._plot_distributions(features_dict, numeric_features[:6])  # Plot first 6 features
        
        # Check for extreme outliers or anomalies
        has_issues = False
        for gender in ['mens', 'womens']:
            for feature, stats in distribution_stats[gender].items():
                # Check for high skew
                if abs(stats['skew']) > 5:
                    logger.warning(f"{gender.title()} feature '{feature}' has high skew: {stats['skew']}")
                    has_issues = True
                
                # Check for extreme outliers (>5% of data)
                outlier_pct = stats['outliers'] / len(features_dict[gender])
                if outlier_pct > 0.05:
                    logger.warning(f"{gender.title()} feature '{feature}' has many outliers: {outlier_pct:.2%}")
                    has_issues = True
                    
                # Check for unrealistic values (e.g., win rates > 1)
                if 'WinRate' in feature:
                    if stats['max'] > 1.1 or stats['min'] < -0.1:
                        logger.warning(f"{gender.title()} feature '{feature}' has unrealistic values: min={stats['min']}, max={stats['max']}")
                        has_issues = True
                
        return not has_issues
    
    def _plot_distributions(self, features_dict, features_to_plot):
        """Plot feature distributions by gender"""
        try:
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Create a combined plot with multiple subplots
            n_features = len(features_to_plot)
            fig, axes = plt.subplots(n_features, 2, figsize=(15, 4*n_features))
            
            # Plot each feature's distribution
            for i, feature in enumerate(features_to_plot):
                # Men's data
                if feature in features_dict['mens'].columns:
                    m_data = features_dict['mens'][feature].dropna()
                    axes[i, 0].hist(m_data, bins=30, alpha=0.7)
                    axes[i, 0].set_title(f"Men's {feature}")
                    axes[i, 0].set_xlabel(feature)
                    axes[i, 0].set_ylabel("Frequency")
                
                # Women's data
                if feature in features_dict['womens'].columns:
                    w_data = features_dict['womens'][feature].dropna()
                    axes[i, 1].hist(w_data, bins=30, alpha=0.7, color='orange')
                    axes[i, 1].set_title(f"Women's {feature}")
                    axes[i, 1].set_xlabel(feature)
                    axes[i, 1].set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_distributions.png")
            plt.close()
            
            # Create boxplots for comparing men's and women's distributions
            fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
            
            for i, feature in enumerate(features_to_plot):
                data_to_plot = []
                labels = []
                
                if feature in features_dict['mens'].columns:
                    data_to_plot.append(features_dict['mens'][feature].dropna())
                    labels.append("Men")
                    
                if feature in features_dict['womens'].columns:
                    data_to_plot.append(features_dict['womens'][feature].dropna())
                    labels.append("Women")
                
                if data_to_plot:
                    axes[i].boxplot(data_to_plot, labels=labels)
                    axes[i].set_title(f"Distribution of {feature}")
                    axes[i].set_ylabel(feature)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_comparisons.png")
            plt.close()
            
            logger.info(f"Saved distribution plots to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
    
    def validate_gender_consistency(self, features_dict=None):
        """
        Validate that men's and women's data are consistently structured
        """
        if features_dict is None:
            features_dict = self.load_feature_data()
            
        if features_dict is None:
            return False
        
        m_features = features_dict['mens']
        w_features = features_dict['womens']
        
        # Check column consistency
        m_cols = set(m_features.columns)
        w_cols = set(w_features.columns)
        
        common_cols = m_cols.intersection(w_cols)
        m_only_cols = m_cols - w_cols
        w_only_cols = w_cols - m_cols
        
        logger.info(f"Common columns: {len(common_cols)}")
        if m_only_cols:
            logger.warning(f"Men-only columns: {m_only_cols}")
        if w_only_cols:
            logger.warning(f"Women-only columns: {w_only_cols}")
        
        # Check data types consistency for common columns
        dtype_mismatches = []
        for col in common_cols:
            m_dtype = m_features[col].dtype
            w_dtype = w_features[col].dtype
            if m_dtype != w_dtype:
                dtype_mismatches.append((col, m_dtype, w_dtype))
                logger.warning(f"Data type mismatch for column '{col}': men={m_dtype}, women={w_dtype}")
        
        # Check value ranges for key numeric features
        range_issues = []
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(m_features[col]) and pd.api.types.is_numeric_dtype(w_features[col]):
                m_min, m_max = m_features[col].min(), m_features[col].max()
                w_min, w_max = w_features[col].min(), w_features[col].max()
                
                # Check if ranges are significantly different
                # This is a simple heuristic - might need adjustment
                if abs(m_min - w_min) > 0.5 * max(abs(m_min), abs(w_min)) or \
                   abs(m_max - w_max) > 0.5 * max(abs(m_max), abs(w_max)):
                    range_issues.append((col, (m_min, m_max), (w_min, w_max)))
                    logger.warning(f"Value range differs significantly for '{col}': men=[{m_min}, {m_max}], women=[{w_min}, {w_max}]")
        
        # Generate gender consistency report
        report_path = self.output_dir / "gender_consistency_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Gender Consistency Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Common columns: {len(common_cols)}\n")
            f.write(f"Men-only columns: {len(m_only_cols)}\n")
            for col in sorted(m_only_cols):
                f.write(f"  {col}\n")
            
            f.write(f"\nWomen-only columns: {len(w_only_cols)}\n")
            for col in sorted(w_only_cols):
                f.write(f"  {col}\n")
            
            f.write(f"\nData type mismatches: {len(dtype_mismatches)}\n")
            for col, m_dtype, w_dtype in dtype_mismatches:
                f.write(f"  {col}: men={m_dtype}, women={w_dtype}\n")
            
            f.write(f"\nValue range issues: {len(range_issues)}\n")
            for col, m_range, w_range in range_issues:
                f.write(f"  {col}: men={m_range}, women={w_range}\n")
        
        logger.info(f"Saved gender consistency report to {report_path}")
        
        # Return True if there are no major issues
        return len(m_only_cols) == 0 and len(w_only_cols) == 0 and len(dtype_mismatches) == 0 and len(range_issues) == 0
    
    def check_feature_correlation_with_target(self, features_dict=None):
        """
        Check correlation of features with the target variable (Result)
        """
        if features_dict is None:
            features_dict = self.load_feature_data()
            
        if features_dict is None:
            return False
        
        # Features to check
        numeric_features = [
            'Team1_WinRate', 'Team2_WinRate',
            'Team1_AvgScore', 'Team2_AvgScore',
            'Team1_AvgAllowed', 'Team2_AvgAllowed',
            'WinRate_Diff', 'AvgScore_Diff', 'AvgAllowed_Diff',
            'Team1_AdjO', 'Team2_AdjO', 
            'Team1_AdjD', 'Team2_AdjD',
            'AdjO_Diff', 'AdjD_Diff'
        ]
        
        correlations = {'mens': {}, 'womens': {}}
        
        # Calculate correlations for men's data
        m_features = features_dict['mens']
        for feature in numeric_features:
            if feature in m_features.columns:
                correlations['mens'][feature] = m_features[['Result', feature]].corr().iloc[0, 1]
        
        # Calculate correlations for women's data
        w_features = features_dict['womens']
        for feature in numeric_features:
            if feature in w_features.columns:
                correlations['womens'][feature] = w_features[['Result', feature]].corr().iloc[0, 1]
        
        # Sort by absolute correlation
        correlations['mens'] = dict(sorted(correlations['mens'].items(), 
                                          key=lambda x: abs(x[1]), 
                                          reverse=True))
        correlations['womens'] = dict(sorted(correlations['womens'].items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True))
        
        # Generate correlation report
        report_path = self.output_dir / "feature_correlation_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Feature Correlation with Target Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("Men's Feature Correlations with Result:\n")
            f.write("-"*50 + "\n")
            for feature, corr in correlations['mens'].items():
                f.write(f"{feature}: {corr:.4f}\n")
            
            f.write("\nWomen's Feature Correlations with Result:\n")
            f.write("-"*50 + "\n")
            for feature, corr in correlations['womens'].items():
                f.write(f"{feature}: {corr:.4f}\n")
        
        logger.info(f"Saved feature correlation report to {report_path}")
        
        # Plot correlation heatmaps
        self._plot_correlation_heatmaps(features_dict, numeric_features)
        
        # Check if we have good predictive features
        mens_max_corr = max(abs(corr) for corr in correlations['mens'].values()) if correlations['mens'] else 0
        womens_max_corr = max(abs(corr) for corr in correlations['womens'].values()) if correlations['womens'] else 0
        
        logger.info(f"Max correlation with target: men={mens_max_corr:.4f}, women={womens_max_corr:.4f}")
        
        # Return True if we have at least some meaningful correlations
        return mens_max_corr >= 0.2 and womens_max_corr >= 0.2
    
    def _plot_correlation_heatmaps(self, features_dict, features_to_plot):
        """Plot correlation heatmaps for both genders"""
        try:
            # Create plots directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Features to include in correlation matrix
            selected_features = ['Result'] + [f for f in features_to_plot 
                                           if f in features_dict['mens'].columns 
                                           and f in features_dict['womens'].columns]
            
            # Plot men's correlation heatmap
            plt.figure(figsize=(12, 10))
            m_corr = features_dict['mens'][selected_features].corr()
            sns.heatmap(m_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Men's Feature Correlations")
            plt.tight_layout()
            plt.savefig(plots_dir / "mens_correlation_heatmap.png")
            plt.close()
            
            # Plot women's correlation heatmap
            plt.figure(figsize=(12, 10))
            w_corr = features_dict['womens'][selected_features].corr()
            sns.heatmap(w_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Women's Feature Correlations")
            plt.tight_layout()
            plt.savefig(plots_dir / "womens_correlation_heatmap.png")
            plt.close()
            
            logger.info(f"Saved correlation heatmaps to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmaps: {str(e)}")
    
    def validate_all(self):
        """Run all validation checks"""
        # Load data
        features_dict = self.load_feature_data()
        if features_dict is None:
            logger.error("Failed to load feature data for validation")
            return False
        
        # Run all validation checks
        completeness_valid = self.validate_feature_completeness(features_dict)
        distribution_valid = self.validate_feature_distributions(features_dict)
        gender_consistency_valid = self.validate_gender_consistency(features_dict)
        correlation_valid = self.check_feature_correlation_with_target(features_dict)
        
        # Overall validation result
        all_valid = completeness_valid and distribution_valid and gender_consistency_valid and correlation_valid
        
        # Generate overall validation report
        report_path = self.output_dir / "validation_summary.txt"
        with open(report_path, 'w') as f:
            f.write(f"Feature Validation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall validation result: {'PASSED' if all_valid else 'FAILED'}\n\n")
            f.write(f"1. Feature completeness: {'PASSED' if completeness_valid else 'FAILED'}\n")
            f.write(f"2. Feature distributions: {'PASSED' if distribution_valid else 'FAILED'}\n")
            f.write(f"3. Gender consistency: {'PASSED' if gender_consistency_valid else 'FAILED'}\n")
            f.write(f"4. Feature correlations: {'PASSED' if correlation_valid else 'FAILED'}\n")
        
        logger.info(f"Overall feature validation: {'PASSED' if all_valid else 'FAILED'}")
        logger.info(f"Saved validation summary to {report_path}")
        
        return all_valid

if __name__ == "__main__":
    validator = FeatureValidator()
    validation_result = validator.validate_all()
    
    print(f"Feature validation {'passed' if validation_result else 'failed'}")
