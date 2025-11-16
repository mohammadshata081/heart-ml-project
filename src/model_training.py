"""
Machine Learning Models Module
Implements Logistic Regression and Random Forest classifiers with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from config import (
    MODEL_TRAINED_DIR,
    LOGISTIC_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_SEED
)


def prepare_features(df, target_col='target'):
    """
    Prepare features and target for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    tuple
        X (features) and y (target)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    C : float
        Inverse of regularization strength
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained model
    """
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                        min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required at a leaf node
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance with comprehensive metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return metrics


def plot_roc_curve(y_test, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """
    Plot ROC curve for the model.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(conf_matrix, model_name="Model", figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    -----------
    conf_matrix : array-like
        Confusion matrix
    model_name : str
        Name of the model
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def compare_models(metrics_lr, metrics_rf):
    """
    Compare two models and provide insights.
    
    Parameters:
    -----------
    metrics_lr : dict
        Metrics for Logistic Regression
    metrics_rf : dict
        Metrics for Random Forest
        
    Returns:
    --------
    dict
        Comparison results and insights
    """
    comparison = {
        'metrics_comparison': pd.DataFrame({
            'Logistic Regression': [
                metrics_lr['accuracy'],
                metrics_lr['precision'],
                metrics_lr['recall'],
                metrics_lr['f1_score'],
                metrics_lr['roc_auc']
            ],
            'Random Forest': [
                metrics_rf['accuracy'],
                metrics_rf['precision'],
                metrics_rf['recall'],
                metrics_rf['f1_score'],
                metrics_rf['roc_auc']
            ]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
        'best_model': None,
        'insights': []
    }
    
    # Determine best model based on multiple metrics
    lr_score = (metrics_lr['accuracy'] + metrics_lr['f1_score'] + metrics_lr['roc_auc']) / 3
    rf_score = (metrics_rf['accuracy'] + metrics_rf['f1_score'] + metrics_rf['roc_auc']) / 3
    
    if lr_score > rf_score:
        comparison['best_model'] = 'Logistic Regression'
    else:
        comparison['best_model'] = 'Random Forest'
    
    # Generate insights
    insights = []
    insights.append(f"Accuracy: {'Logistic Regression' if metrics_lr['accuracy'] > metrics_rf['accuracy'] else 'Random Forest'} performs better ({max(metrics_lr['accuracy'], metrics_rf['accuracy']):.3f})")
    insights.append(f"F1 Score: {'Logistic Regression' if metrics_lr['f1_score'] > metrics_rf['f1_score'] else 'Random Forest'} performs better ({max(metrics_lr['f1_score'], metrics_rf['f1_score']):.3f})")
    insights.append(f"ROC AUC: {'Logistic Regression' if metrics_lr['roc_auc'] > metrics_rf['roc_auc'] else 'Random Forest'} performs better ({max(metrics_lr['roc_auc'], metrics_rf['roc_auc']):.3f})")
    
    insights.append("\nModel Strengths:")
    insights.append("Logistic Regression:")
    insights.append("- Simpler, more interpretable model")
    insights.append("- Faster training and prediction")
    insights.append("- Less prone to overfitting")
    insights.append("- Good baseline model")
    
    insights.append("\nRandom Forest:")
    insights.append("- Can capture non-linear relationships")
    insights.append("- Handles feature interactions well")
    insights.append("- More robust to outliers")
    insights.append("- Can provide feature importance")
    
    comparison['insights'] = insights
    
    return comparison


def save_model(model, filepath=None, model_name="model"):
    """
    Save trained model to file.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to save
    filepath : str, optional
        Path to save the model. If None, uses default path from config based on model_name
    model_name : str
        Name of the model ('logistic_regression' or 'random_forest')
    """
    # Use default path from config if filepath not provided
    if filepath is None:
        if 'logistic' in model_name.lower() or 'lr' in model_name.lower():
            filepath = LOGISTIC_REGRESSION_MODEL_PATH
        elif 'random' in model_name.lower() or 'rf' in model_name.lower() or 'forest' in model_name.lower():
            filepath = RANDOM_FOREST_MODEL_PATH
        else:
            # Default to model directory with model name
            filepath = os.path.join(MODEL_TRAINED_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[OK] {model_name} saved to {filepath}")


def load_model(filepath):
    """
    Load saved model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    sklearn model
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"[OK] Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Test ML models
    print("=" * 50)
    print("ML Models Module Test")
    print("=" * 50)
    
    from src import data_preprocessing as dp
    
    df = dp.load_data()
    if df is not None:
        df = dp.preprocess_data(df)
        X, y = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        print(f"Accuracy: {lr_metrics['accuracy']:.3f}")
        print(f"F1 Score: {lr_metrics['f1_score']:.3f}")
        print(f"ROC AUC: {lr_metrics['roc_auc']:.3f}")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        print(f"Accuracy: {rf_metrics['accuracy']:.3f}")
        print(f"F1 Score: {rf_metrics['f1_score']:.3f}")
        print(f"ROC AUC: {rf_metrics['roc_auc']:.3f}")
        
        # Compare models
        print("\nModel Comparison:")
        comparison = compare_models(lr_metrics, rf_metrics)
        print(comparison['metrics_comparison'])
        print(f"\nBest Model: {comparison['best_model']}")

