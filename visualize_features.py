import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import pickle
import argparse

def plot_feature_importance(features_path, model_path=None, output_dir=None, top_n=30):
    """
    Visualize feature importance from the trained model
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load feature importance data
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found at {features_path}")
        return False
    
    try:
        features_df = pd.read_csv(features_path)
        print(f"Loaded feature data with {len(features_df)} features")
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        return False
    
    # If model path is provided, extract feature importance from the model
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # For ensemble models, try to extract importance from component models
            if hasattr(model, 'models'):
                print("Ensemble model detected, extracting importances from component models")
                for model_name, component_model in model.models.items():
                    if hasattr(component_model, 'feature_importances_') or hasattr(component_model, 'coef_'):
                        print(f"Extracting importances from {model_name} model")
                        
                        if hasattr(component_model, 'feature_importances_'):
                            importances = component_model.feature_importances_
                        else:
                            importances = np.abs(component_model.coef_[0])
                        
                        # Create a DataFrame for this model's importances
                        model_importances = pd.DataFrame({
                            'Feature': features_df['Feature'],
                            'Importance': importances,
                            'Model': model_name
                        })
                        
                        # Plot top features for this model
                        if output_dir:
                            plt.figure(figsize=(12, 10))
                            sns.barplot(
                                x='Importance', 
                                y='Feature', 
                                data=model_importances.nlargest(top_n, 'Importance')
                            )
                            plt.title(f'Top {top_n} Features - {model_name} Model')
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'), dpi=300)
        except Exception as e:
            print(f"Error processing model for feature importance: {str(e)}")
    
    # Plot feature importance from the features file
    print(f"Creating feature importance visualization for top {top_n} features")
    plt.figure(figsize=(12, 10))
    
    # Use top_n features for visualization
    top_features = features_df.nlargest(top_n, 'Importance')
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Features by Importance')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        print(f"Saved feature importance visualization to {output_dir}")
    
    plt.figure(figsize=(12, 8))
    
    # Group features by type
    feature_types = []
    for feature in top_features['Feature']:
        if feature.startswith('Team1_'):
            feature_types.append('Team1')
        elif feature.startswith('Team2_'):
            feature_types.append('Team2')
        elif feature.startswith('Diff_'):
            feature_types.append('Difference')
        elif feature.startswith('Ratio_'):
            feature_types.append('Ratio')
        else:
            feature_types.append('Other')
    
    top_features['Type'] = feature_types
    
    # Create grouped plot
    sns.countplot(x='Type', data=top_features)
    plt.title('Feature Types in Top Features')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_type_distribution.png'), dpi=300)
    
    # Save the table of top features
    if output_dir:
        top_features.to_csv(os.path.join(output_dir, 'top_features.csv'), index=False)
    
    return top_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize feature importance")
    parser.add_argument('--features', type=str, 
                      default='/Volumes/MINT/projects/model/models/selected_features.csv',
                      help='Path to features CSV file')
    parser.add_argument('--model', type=str, 
                      default='/Volumes/MINT/projects/model/models/ensemble_model.pkl',
                      help='Path to model pickle file')
    parser.add_argument('--output', type=str, 
                      default='/Volumes/MINT/projects/model/evaluation/feature_analysis',
                      help='Output directory for visualizations')
    parser.add_argument('--top', type=int, default=30,
                      help='Number of top features to visualize')
    
    args = parser.parse_args()
    plot_feature_importance(args.features, args.model, args.output, args.top)
