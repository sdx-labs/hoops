import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, confusion_matrix
from sklearn.calibration import calibration_curve

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return metrics
    """
    cv_scores = {}
    
    # Brier score
    brier_scores = cross_val_score(
        model, X, y, cv=cv, scoring='neg_brier_score'
    )
    cv_scores['brier'] = -brier_scores.mean()
    
    # Log loss
    log_loss_scores = cross_val_score(
        model, X, y, cv=cv, scoring='neg_log_loss'
    )
    cv_scores['log_loss'] = -log_loss_scores.mean()
    
    # Accuracy
    acc_scores = cross_val_score(
        model, X, y, cv=cv, scoring='accuracy'
    )
    cv_scores['accuracy'] = acc_scores.mean()
    
    # Print results
    print("Cross-validation results:")
    print(f"  Brier Score: {cv_scores['brier']:.4f} (±{np.std(-brier_scores):.4f})")
    print(f"  Log Loss: {cv_scores['log_loss']:.4f} (±{np.std(-log_loss_scores):.4f})")
    print(f"  Accuracy: {cv_scores['accuracy']:.4f} (±{np.std(acc_scores):.4f})")
    
    return cv_scores

def plot_calibration_curve(model, X, y, n_bins=10, output_path=None):
    """
    Plot the calibration curve of the model
    """
    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=n_bins)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration line
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('Calibration Curve')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()

def plot_feature_importance(model, features, top_n=20, output_path=None):
    """
    Plot feature importances for tree-based models
    """
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            importances = model.steps[-1][1].feature_importances_
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'coef_'):
            importances = np.abs(model.steps[-1][1].coef_[0])
        else:
            print("Model does not have feature_importances_ or coef_ attribute")
            return

    # Sort and select top_n
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    top_features = np.array(features)[top_indices]
    top_importances = importances[top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_importances)), top_importances, align='center')
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()

def analyze_upsets(model, X, y, output_path=None):
    """
    Analyze model performance on upsets vs non-upsets
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Create upset indicator (assuming lower seeded team should win)
    # For now, we'll just use a random indicator as example since we don't have seeds
    np.random.seed(42)
    is_upset = np.random.choice([0, 1], size=len(y), p=[0.8, 0.2])
    
    # Calculate metrics for upsets vs non-upsets
    upset_acc = accuracy_score(y[is_upset == 1], y_pred[is_upset == 1])
    nonupset_acc = accuracy_score(y[is_upset == 0], y_pred[is_upset == 0])
    
    upset_brier = brier_score_loss(y[is_upset == 1], y_pred_proba[is_upset == 1])
    nonupset_brier = brier_score_loss(y[is_upset == 0], y_pred_proba[is_upset == 0])
    
    # Print results
    print("Model performance on upsets vs non-upsets:")
    print(f"  Upset Accuracy: {upset_acc:.4f}")
    print(f"  Non-Upset Accuracy: {nonupset_acc:.4f}")
    print(f"  Upset Brier Score: {upset_brier:.4f}")
    print(f"  Non-Upset Brier Score: {nonupset_brier:.4f}")
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    ax[0].bar(['Upsets', 'Non-Upsets'], [upset_acc, nonupset_acc])
    ax[0].set_ylim([0, 1])
    ax[0].set_title('Accuracy')
    ax[0].grid(axis='y')
    
    # Brier score comparison
    ax[1].bar(['Upsets', 'Non-Upsets'], [upset_brier, nonupset_brier])
    ax[1].set_ylim([0, 0.5])
    ax[1].set_title('Brier Score (lower is better)')
    ax[1].grid(axis='y')
    
    plt.suptitle('Model Performance on Upsets vs Non-Upsets')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    
    plt.show()

def run_validation(model, X, y, features, output_dir=None):
    """
    Run all validation analyses
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Cross Validation ===")
    cv_scores = cross_validate_model(model, X, y)
    
    print("\n=== Train-Test Split Validation ===")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Train on the train set
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_log_loss = log_loss(y_test, y_pred_proba)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"Test set results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Log Loss: {test_log_loss:.4f}")
    print(f"  Brier Score: {test_brier:.4f}")
    
    # Plot calibration curve
    print("\n=== Calibration Analysis ===")
    cal_path = os.path.join(output_dir, 'calibration_curve.png') if output_dir else None
    plot_calibration_curve(model, X_test, y_test, output_path=cal_path)
    
    # Plot feature importance
    print("\n=== Feature Importance Analysis ===")
    feat_path = os.path.join(output_dir, 'feature_importance.png') if output_dir else None
    plot_feature_importance(model, features, output_path=feat_path)
    
    # Analyze upsets
    print("\n=== Upset Analysis ===")
    upset_path = os.path.join(output_dir, 'upset_analysis.png') if output_dir else None
    analyze_upsets(model, X_test, y_test, output_path=upset_path)
    
    return {
        'cv_scores': cv_scores,
        'test_accuracy': test_acc,
        'test_log_loss': test_log_loss,
        'test_brier': test_brier
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    features = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Run validation
    run_validation(model, X, y, features, '/Volumes/MINT/projects/model/evaluation')
