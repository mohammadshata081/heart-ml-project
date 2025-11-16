# Heart Disease Prediction System

A comprehensive data science project that combines Exploratory Data Analysis (EDA), Machine Learning model training, and an interactive Streamlit web application for heart disease prediction.

## 📋 Project Overview

This project implements a complete machine learning pipeline for predicting heart disease using patient medical data. It includes:

- **Data Preprocessing**: Robust data loading, validation, and preprocessing
- **Exploratory Data Analysis**: Comprehensive visualizations and insights
- **Machine Learning Models**: Logistic Regression and Random Forest classifiers
- **Interactive Web Application**: User-friendly Streamlit interface for data exploration, model training, and predictions

## 🎯 Features

### 1. Data View & EDA
- Dataset overview and statistics
- Missing value analysis
- Target variable distribution visualization
- Histograms for feature distributions
- Boxplots for outlier detection
- Correlation matrix heatmap
- Automated insights generation

### 2. Model Training
- **Logistic Regression**: Configurable hyperparameters (C, max_iter)
- **Random Forest**: Configurable hyperparameters (n_estimators, max_depth, min_samples_split)
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- ROC curve visualizations
- Confusion matrix heatmaps
- Model comparison and insights

### 3. Prediction Interface
- Interactive form for patient data input
- Real-time predictions with probabilities
- Visual probability display
- Support for both trained models

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd heart_disease_model
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure dataset is available**
   - The `heart_disease.csv` file should be in the project root directory

## 📊 Dataset

The project uses a heart disease dataset with the following features:

- **age**: Age in years
- **sex**: Sex (0 = Female, 1 = Male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (0 = No, 1 = Yes)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by flourosopy (0-4)
- **thal**: Thalassemia (0-3)
- **target**: Target variable (0 = No disease, 1 = Disease)

## 🏃 Running the Application

### Start the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Data View Page** 📊
   - Explore the dataset
   - View summary statistics
   - Analyze missing values
   - Examine visualizations (histograms, boxplots, correlation heatmap)
   - Read generated insights

2. **Model Training Page** 🤖
   - Select model(s) to train (Logistic Regression, Random Forest, or Both)
   - Adjust hyperparameters using sliders
   - Click "Train Model" to train and evaluate
   - View performance metrics and visualizations
   - Compare models side-by-side

3. **Prediction Interface** 🔮
   - Select a trained model
   - Enter patient information using the interactive form
   - Click "Predict" to get predictions
   - View predicted class and probabilities
   - See visual probability distribution

## 📁 Project Structure

```
heart_disease_model/
├── app.py                      # Main Streamlit application
├── data_preprocessing.py       # Data loading and preprocessing module
├── eda.py                      # Exploratory Data Analysis module
├── ml_models.py                # Machine Learning models module
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── heart_disease.csv           # Dataset file
└── PROJECT_OUTLINE.md          # Detailed project outline
```

## 🔧 Module Descriptions

### `data_preprocessing.py`
- `load_data()`: Load dataset from CSV
- `validate_data()`: Validate data types and quality
- `get_summary_statistics()`: Generate descriptive statistics
- `preprocess_data()`: Handle missing values and data cleaning

### `eda.py`
- `get_dataset_overview()`: Dataset structure and info
- `analyze_missing_values()`: Missing value analysis
- `analyze_target_variable()`: Target distribution analysis
- `create_histograms()`: Feature distribution visualizations
- `create_boxplots()`: Outlier detection visualizations
- `create_correlation_heatmap()`: Feature correlation analysis
- `generate_eda_insights()`: Automated insights generation

### `ml_models.py`
- `prepare_features()`: Feature and target preparation
- `train_logistic_regression()`: Train LR model
- `train_random_forest()`: Train RF model
- `evaluate_model()`: Comprehensive model evaluation
- `plot_roc_curve()`: ROC curve visualization
- `plot_confusion_matrix()`: Confusion matrix visualization
- `compare_models()`: Model comparison and insights
- `save_model()` / `load_model()`: Model persistence

## 📈 Model Performance

The models are evaluated using multiple metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

Both models generate:
- ROC curve plots
- Confusion matrices
- Detailed performance metrics

## 🎓 Key Insights

### Model Comparison
- **Logistic Regression**: 
  - Simpler, more interpretable
  - Faster training and prediction
  - Good baseline model
  
- **Random Forest**:
  - Captures non-linear relationships
  - Handles feature interactions
  - More robust to outliers

### Data Insights
- The dataset contains 1025 samples with 14 features
- No missing values (excellent data quality)
- Target variable distribution analysis
- Feature correlations and relationships

## 🛠️ Technical Requirements

- **Python**: 3.7+
- **Libraries**:
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - scikit-learn >= 1.2.0
  - matplotlib >= 3.6.0
  - streamlit >= 1.25.0
  - seaborn >= 0.12.0
  - joblib >= 1.2.0

## 📝 Usage Examples

### Running Individual Modules

You can test individual modules:

```bash
# Test data preprocessing
python data_preprocessing.py

# Test EDA
python eda.py

# Test ML models
python ml_models.py
```

### Custom Hyperparameters

In the Streamlit app, you can adjust:
- **Logistic Regression**: C (0.01-10.0), max_iter (100-5000)
- **Random Forest**: n_estimators (10-500), max_depth (1-20), min_samples_split (2-10)

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Solution: Install dependencies with `pip install -r requirements.txt`

2. **FileNotFoundError: heart_disease.csv**
   - Solution: Ensure the CSV file is in the project root directory

3. **Port already in use**
   - Solution: Streamlit will automatically use the next available port, or specify: `streamlit run app.py --server.port 8502`

## 📄 License

This project is created for educational and demonstration purposes.

## 👤 Author

Heart Disease Prediction System - Data Science Project

## 🙏 Acknowledgments

- Dataset: Heart Disease Dataset
- Libraries: pandas, scikit-learn, streamlit, matplotlib, seaborn

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Note**: This project is designed for educational purposes. For medical applications, consult healthcare professionals and ensure proper validation and regulatory compliance.

