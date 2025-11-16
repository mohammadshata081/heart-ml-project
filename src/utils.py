"""
Utility Functions Module
Helper functions for loading saved models, making predictions, and formatting output.
"""

import joblib
import os
import pandas as pd
import numpy as np
from config import (
    LOGISTIC_REGRESSION_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    MODEL_TRAINED_DIR
)


def load_model(model_type='logistic_regression'):
    """
    Load a saved trained model from disk.
    
    Parameters:
    -----------
    model_type : str
        Type of model to load ('logistic_regression' or 'random_forest')
        
    Returns:
    --------
    model : sklearn model object
        Loaded trained model, or None if file not found
    """
    try:
        if model_type.lower() == 'logistic_regression':
            model_path = LOGISTIC_REGRESSION_MODEL_PATH
        elif model_type.lower() == 'random_forest':
            model_path = RANDOM_FOREST_MODEL_PATH
        else:
            print(f"[ERROR] Unknown model type: {model_type}")
            return None
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        print(f"[OK] Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        return None


def load_model_from_path(model_path):
    """
    Load a model from a specific file path.
    
    Parameters:
    -----------
    model_path : str
        Full path to the model file
        
    Returns:
    --------
    model : sklearn model object
        Loaded trained model, or None if file not found
    """
    try:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        print(f"[OK] Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        return None


def make_prediction(model, features):
    """
    Make predictions using a trained model.
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model
    features : pd.DataFrame, np.array, or dict
        Feature values for prediction
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'prediction': predicted class
        - 'probability': prediction probabilities
        - 'probability_class_0': probability of class 0
        - 'probability_class_1': probability of class 1
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            prob_class_0 = probabilities[0]
            prob_class_1 = probabilities[1]
        else:
            # For models without predict_proba, use decision function
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(features)[0]
                prob_class_1 = 1 / (1 + np.exp(-decision))  # Sigmoid transformation
                prob_class_0 = 1 - prob_class_1
            else:
                prob_class_0 = None
                prob_class_1 = None
        
        return {
            'prediction': int(prediction),
            'probability': probabilities if hasattr(model, 'predict_proba') else None,
            'probability_class_0': float(prob_class_0) if prob_class_0 is not None else None,
            'probability_class_1': float(prob_class_1) if prob_class_1 is not None else None
        }
    
    except Exception as e:
        print(f"[ERROR] Error making prediction: {str(e)}")
        return None


def format_prediction_output(prediction_result, class_names=None):
    """
    Format prediction output for display.
    
    Parameters:
    -----------
    prediction_result : dict
        Result from make_prediction function
    class_names : dict, optional
        Mapping of class numbers to names (e.g., {0: 'No Disease', 1: 'Disease'})
        
    Returns:
    --------
    str
        Formatted prediction output string
    """
    if prediction_result is None:
        return "Error: Could not make prediction"
    
    prediction = prediction_result['prediction']
    prob_class_0 = prediction_result.get('probability_class_0')
    prob_class_1 = prediction_result.get('probability_class_1')
    
    # Use class names if provided
    if class_names:
        class_name = class_names.get(prediction, f"Class {prediction}")
    else:
        class_name = f"Class {prediction}"
    
    output = f"Prediction: {class_name}\n"
    
    if prob_class_0 is not None and prob_class_1 is not None:
        output += f"Probability:\n"
        if class_names:
            output += f"  - {class_names.get(0, 'Class 0')}: {prob_class_0:.2%}\n"
            output += f"  - {class_names.get(1, 'Class 1')}: {prob_class_1:.2%}\n"
        else:
            output += f"  - Class 0: {prob_class_0:.2%}\n"
            output += f"  - Class 1: {prob_class_1:.2%}\n"
    
    return output


def format_metrics_output(metrics_dict):
    """
    Format model metrics for display.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing metric names and values
        
    Returns:
    --------
    str
        Formatted metrics output string
    """
    output = "Model Performance Metrics:\n"
    output += "=" * 40 + "\n"
    
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, float):
            output += f"{metric_name.capitalize()}: {metric_value:.4f}\n"
        else:
            output += f"{metric_name.capitalize()}: {metric_value}\n"
    
    return output


def check_model_files():
    """
    Check if trained model files exist.
    
    Returns:
    --------
    dict
        Dictionary with status of each model file
    """
    models_status = {
        'logistic_regression': os.path.exists(LOGISTIC_REGRESSION_MODEL_PATH),
        'random_forest': os.path.exists(RANDOM_FOREST_MODEL_PATH)
    }
    
    return models_status


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, create if it doesn't.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory
        
    Returns:
    --------
    bool
        True if directory exists or was created successfully, False otherwise
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"[OK] Created directory: {directory_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error creating directory {directory_path}: {str(e)}")
        return False


if __name__ == "__main__":
    # Test utility functions
    print("=" * 50)
    print("Utility Functions Module Test")
    print("=" * 50)
    
    # Check model files
    print("\nChecking model files...")
    models_status = check_model_files()
    for model_name, exists in models_status.items():
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"  {model_name}: {status}")
    
    # Test directory creation
    print("\nTesting directory creation...")
    test_dir = os.path.join(MODEL_TRAINED_DIR, 'test')
    ensure_directory_exists(test_dir)
    
    print("\n[OK] Utility functions test completed!")

