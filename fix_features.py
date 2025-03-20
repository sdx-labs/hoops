import os
import pandas as pd
import numpy as np
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.load_data import load_kaggle_data, load_kenpom_data
from feature_engineering.build_features import TeamFeatureBuilder

def diagnose_features():
    """
    Diagnose feature engineering issues and check feature values
    """
    print("=== Feature Engineering Diagnosis ===")
    
    # Load data
    print("\n1. Loading data...")
    kaggle_data = load_kaggle_data()
    kenpom_data = load_kenpom_data()
    
    # Initialize feature builder
    print("\n2. Building basic features...")
    feature_builder = TeamFeatureBuilder(kaggle_data, kenpom_data)
    
    # Get a sample of seasons to diagnose
    test_seasons = range(2020, 2023)
    
    # Build and examine features
    basic_features = feature_builder.build_basic_stats(test_seasons)
    
    # Check basic features
    print("\n3. Examining basic features:")
    if basic_features:
        season = list(basic_features.keys())[0]
        team_id = list(basic_features[season].keys())[0]
        print(f"Sample basic features for team {team_id} in season {season}:")
        for feature, value in basic_features[season][team_id].items():
            print(f"  {feature}: {value}")
    else:
        print("ERROR: No basic features were created")
    
    # Build advanced features
    print("\n4. Building advanced features...")
    advanced_features = feature_builder.build_advanced_stats(test_seasons)
    
    # Check advanced features
    print("\n5. Examining advanced features:")
    if advanced_features:
        season = list(advanced_features.keys())[0]
        if advanced_features[season]:  # Check if the season has any teams
            team_id = list(advanced_features[season].keys())[0]
            print(f"Sample advanced features for team {team_id} in season {season}:")
            for feature, value in advanced_features[season][team_id].items():
                print(f"  {feature}: {value}")
        else:
            print("WARNING: No teams found in advanced features for this season")
    else:
        print("WARNING: No advanced features were created")
    
    # Build and examine KenPom features
    if kenpom_data is not None:
        print("\n6. Building KenPom features...")
        kenpom_features = feature_builder.integrate_kenpom_data(test_seasons)
        
        print("\n7. Examining KenPom features:")
        if kenpom_features and any(kenpom_features.get(season, {}) for season in test_seasons):
            for season in test_seasons:
                if season in kenpom_features and kenpom_features[season]:
                    team_id = list(kenpom_features[season].keys())[0]
                    print(f"Sample KenPom features for team {team_id} in season {season}:")
                    for feature, value in list(kenpom_features[season][team_id].items())[:5]:
                        print(f"  {feature}: {value}")
                    if len(kenpom_features[season][team_id]) > 5:
                        print(f"  ... and {len(kenpom_features[season][team_id])-5} more features")
                    break
            else:
                print("WARNING: No KenPom features found for test seasons")
        else:
            print("WARNING: No KenPom features were created")
    
    # Test combining all features
    print("\n8. Testing combined features...")
    combined_features = feature_builder.combine_all_features(test_seasons)
    
    # Convert to DataFrame for easier analysis
    print("\n9. Converting features to DataFrame for analysis...")
    rows = []
    for season in combined_features:
        for team_id in combined_features[season]:
            row = {'Season': season, 'TeamID': team_id}
            row.update(combined_features[season][team_id])
            rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Check for missing values or constant columns
    null_counts = df.isnull().sum()
    print(f"\nColumns with missing values: {sum(null_counts > 0)}")
    print("Top 5 columns with most missing values:")
    for col, count in null_counts.nlargest(5).items():
        print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    # Check for constant values
    constant_cols = []
    for col in df.columns:
        if col not in ['Season', 'TeamID'] and df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\nFound {len(constant_cols)} columns with constant values:")
        for col in constant_cols[:5]:
            print(f"  {col}: {df[col].iloc[0]}")
        if len(constant_cols) > 5:
            print(f"  ... and {len(constant_cols)-5} more")
    
    # Check feature variability
    print("\nFeature statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  Numeric features: {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std()
        
        print("Features with highest mean values:")
        for col, mean in means.nlargest(5).items():
            print(f"  {col}: {mean:.4f}")
        
        print("Features with highest standard deviation:")
        for col, std in stds.nlargest(5).items():
            print(f"  {col}: {std:.4f}")
    
    print("\n=== Diagnosis Complete ===")
    return df

if __name__ == "__main__":
    feature_df = diagnose_features()
    
    # Save diagnostic results to a file
    output_path = os.path.join(os.path.dirname(__file__), 'evaluation', 'feature_diagnosis.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
        feature_df.head(100).to_csv(output_path, index=False)
        print(f"\nSaved sample feature data to {output_path}")
    
    print("\nTo fix feature issues, run:")
    print("  python run_pipeline.py --prepare-kenpom")
    print("This will rebuild the features and ensure proper data is used for modeling.")
