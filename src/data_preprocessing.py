"""
Data Preprocessing Module
Handles data loading, validation, and preprocessing for the heart disease dataset.
"""

import pandas as pd
import numpy as np
import os
from config import DATASET_PATH, PROCESSED_DATA_PATH, DATA_PROCESSED_DIR


def load_data(file_path=None):
    """
    Load the heart disease dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file. If None, uses path from config.py
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if file_path is None:
        file_path = DATASET_PATH
    
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {str(e)}")
        return None


def validate_data(df):
    """
    Validate data types and check for data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    validation_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else None
    }
    
    return validation_results


def get_summary_statistics(df):
    """
    Generate comprehensive summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    return df.describe()


def preprocess_data(df):
    """
    Main preprocessing function that handles missing values and data cleaning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset
    """
    df_processed = df.copy()
    
    # Check for missing values
    missing_count = df_processed.isnull().sum().sum()
    if missing_count > 0:
        print(f"[WARNING] Found {missing_count} missing values. Handling them...")
        # For this dataset, we'll use forward fill for numeric columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    else:
        print("[OK] No missing values found in the dataset")
    
    # Remove duplicates if any
    duplicates = df_processed.duplicated().sum()
    if duplicates > 0:
        print(f"[WARNING] Found {duplicates} duplicate rows. Removing them...")
        df_processed = df_processed.drop_duplicates()
        print(f"[OK] Dataset after removing duplicates: {df_processed.shape}")
    
    return df_processed


def save_processed_data(df_processed, file_path=None):
    """
    Save processed dataset to CSV file in data/processed/ directory.
    
    Parameters:
    -----------
    df_processed : pd.DataFrame
        Processed dataset to save
    file_path : str, optional
        Path to save the processed data. If None, uses path from config.py
        
    Returns:
    --------
    bool
        True if saved successfully, False otherwise
    """
    if file_path is None:
        file_path = PROCESSED_DATA_PATH
    
    try:
        # Ensure directory exists
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        
        # Save to CSV
        df_processed.to_csv(file_path, index=False)
        print(f"[OK] Processed data saved successfully to {file_path}")
        print(f"     Shape: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving processed data: {str(e)}")
        return False


def split_data(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to split
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    from config import TARGET_VARIABLE
    
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[OK] Data split completed:")
    print(f"     Training set: {X_train.shape[0]} samples")
    print(f"     Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the preprocessing functions
    print("=" * 50)
    print("Data Preprocessing Module Test")
    print("=" * 50)
    
    df = load_data()
    if df is not None:
        validation = validate_data(df)
        print("\nValidation Results:")
        print(f"Shape: {validation['shape']}")
        print(f"Missing values: {sum(validation['missing_values'].values())}")
        print(f"Duplicates: {validation['duplicates']}")
        
        df_processed = preprocess_data(df)
        print(f"\nProcessed dataset shape: {df_processed.shape}")
        
        # Save processed data
        print("\n" + "=" * 50)
        print("Saving Processed Data")
        print("=" * 50)
        save_processed_data(df_processed)

