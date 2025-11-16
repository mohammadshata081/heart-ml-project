"""
Configuration Module
Central configuration file for dataset paths, model parameters, and project settings.
"""

import os

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET_PATH = os.path.join('data', 'raw', 'dataset.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'processed.csv')

# ============================================================================
# Target Variable Configuration
# ============================================================================
TARGET_VARIABLE = 'target'
TARGET_NAME = 'target'

# ============================================================================
# Feature Names (Update based on your dataset)
# ============================================================================
# Common heart disease dataset features (update as needed)
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# ============================================================================
# Model Parameters
# ============================================================================

# Logistic Regression Parameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42
}

# Random Forest Parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Train-Test Split Parameters
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# ============================================================================
# Random Seed for Reproducibility
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
MODEL_TRAINED_DIR = os.path.join(BASE_DIR, 'models', 'trained_models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# ============================================================================
# Model Save Paths
# ============================================================================
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODEL_TRAINED_DIR, 'logistic_regression.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_TRAINED_DIR, 'random_forest.pkl')

# ============================================================================
# Output File Paths
# ============================================================================
HISTOGRAM_PATH = os.path.join(OUTPUT_FIGURES_DIR, 'histograms.png')
BOXPLOT_PATH = os.path.join(OUTPUT_FIGURES_DIR, 'boxplots.png')
CORRELATION_HEATMAP_PATH = os.path.join(OUTPUT_FIGURES_DIR, 'correlation_heatmap.png')
TARGET_DISTRIBUTION_PATH = os.path.join(OUTPUT_FIGURES_DIR, 'target_distribution.png')

# ============================================================================
# EDA Configuration
# ============================================================================
NUM_HISTOGRAMS = 3  # Minimum number of histograms to generate
NUM_BOXPLOTS = 3    # Minimum number of boxplots to generate
FIGURE_DPI = 150    # Resolution for saved figures

# ============================================================================
# File Paths
# ============================================================================
FILE_PATHS = {
    'dataset': DATASET_PATH,
    'processed_data': PROCESSED_DATA_PATH,
    'logistic_regression_model': LOGISTIC_REGRESSION_MODEL_PATH,
    'random_forest_model': RANDOM_FOREST_MODEL_PATH,
    'scaler': os.path.join(MODEL_TRAINED_DIR, 'scaler.pkl'),
    'model_metadata': os.path.join(MODEL_TRAINED_DIR, 'model_metadata.json'),
    'eda_report': os.path.join(BASE_DIR, 'outputs', 'results', 'eda_report.txt'),
    'model_comparison': os.path.join(BASE_DIR, 'outputs', 'results', 'model_comparison.txt')
}

