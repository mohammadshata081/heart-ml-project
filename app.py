"""
Heart Disease Prediction - Streamlit Application
Main application file for the interactive heart disease prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src import data_preprocessing as dp
from src import eda
from src import model_training as ml

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'lr_metrics' not in st.session_state:
    st.session_state.lr_metrics = None
if 'rf_metrics' not in st.session_state:
    st.session_state.rf_metrics = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None


def load_data():
    """Load and preprocess data."""
    if st.session_state.df is None:
        with st.spinner("Loading dataset..."):
            df = dp.load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.df_processed = dp.preprocess_data(df)
                st.success("Dataset loaded successfully!")
            else:
                st.error("Failed to load dataset!")
    return st.session_state.df_processed


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 Data View", "🤖 Model Training", "🔮 Prediction Interface"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. **Data View**: Explore the dataset and EDA visualizations
    2. **Model Training**: Train and evaluate ML models
    3. **Prediction Interface**: Make predictions on new data
    """)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Please ensure the dataset is in the 'data/raw/' directory. Check config.py for the correct path.")
        return
    
    # Route to appropriate page
    if page == "📊 Data View":
        show_data_view(df)
    elif page == "🤖 Model Training":
        show_model_training(df)
    elif page == "🔮 Prediction Interface":
        show_prediction_interface()


def show_data_view(df):
    """Display data view page with EDA."""
    st.header("📊 Data View & Exploratory Data Analysis")
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Target Classes", df['target'].nunique())
    
    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width='stretch')
    
    # Data Types
    st.subheader("Data Types")
    st.dataframe(pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    }), width='stretch')
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), width='stretch')
    
    # Missing Values Analysis
    st.subheader("Missing Values Analysis")
    missing_analysis = eda.analyze_missing_values(df)
    if missing_analysis['total_missing'] == 0:
        st.success("No missing values found in the dataset!")
    else:
        st.warning(f"Found {missing_analysis['total_missing']} missing values")
        missing_fig = eda.visualize_missing_values(df)
        if missing_fig:
            st.pyplot(missing_fig)
    
    # Target Variable Analysis
    st.subheader("Target Variable Distribution")
    target_analysis = eda.analyze_target_variable(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Value Counts:**")
        st.write(target_analysis['value_counts'])
    with col2:
        st.write("**Percentages:**")
        st.write(target_analysis['percentages'])
    
    target_fig = eda.visualize_target_distribution(df)
    st.pyplot(target_fig)
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    # Histograms
    st.write("**Histograms - Feature Distributions**")
    hist_fig = eda.create_histograms(df, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    st.pyplot(hist_fig)
    st.caption("These histograms show the distribution of key numerical features. They help identify data skewness and outliers.")
    
    # Boxplots
    st.write("**Boxplots - Outlier Detection**")
    box_fig = eda.create_boxplots(df, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    st.pyplot(box_fig)
    st.caption("Boxplots help identify outliers and understand the spread of data across different features.")
    
    # Correlation Heatmap
    st.write("**Correlation Matrix Heatmap**")
    corr_fig = eda.create_correlation_heatmap(df)
    st.pyplot(corr_fig)
    st.caption("The correlation heatmap shows relationships between features. Strong correlations (close to ±1) indicate features that move together.")
    
    # EDA Insights
    st.subheader("Key Insights")
    insights = eda.generate_eda_insights(df)
    for key, insight in insights.items():
        st.text(insight)


def show_model_training(df):
    """Display model training page."""
    st.header("🤖 Model Training & Evaluation")
    
    # Model Selection
    st.subheader("Select Model to Train")
    model_choice = st.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Random Forest", "Both Models"]
    )
    
    # Prepare features
    X, y = ml.prepare_features(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    if st.session_state.X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
    
    st.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    
    if model_choice in ["Logistic Regression", "Both Models"]:
        st.write("**Logistic Regression Hyperparameters:**")
        col1, col2 = st.columns(2)
        with col1:
            lr_C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01, key="lr_C")
        with col2:
            lr_max_iter = st.slider("Max Iterations", 100, 5000, 1000, 100, key="lr_max_iter")
    
    if model_choice in ["Random Forest", "Both Models"]:
        st.write("**Random Forest Hyperparameters:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_n_estimators = st.slider("N Estimators", 10, 500, 100, 10, key="rf_n_est")
        with col2:
            rf_max_depth = st.slider("Max Depth", 1, 20, 10, 1, key="rf_max_depth")
            if rf_max_depth == 20:
                rf_max_depth = None
        with col3:
            rf_min_samples_split = st.slider("Min Samples Split", 2, 10, 2, 1, key="rf_min_split")
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        if model_choice in ["Logistic Regression", "Both Models"]:
            with st.spinner("Training Logistic Regression..."):
                lr_model = ml.train_logistic_regression(
                    X_train, y_train, C=lr_C, max_iter=lr_max_iter
                )
                lr_metrics = ml.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
                st.session_state.lr_model = lr_model
                st.session_state.lr_metrics = lr_metrics
                st.success("Logistic Regression trained successfully!")
        
        if model_choice in ["Random Forest", "Both Models"]:
            with st.spinner("Training Random Forest..."):
                rf_model = ml.train_random_forest(
                    X_train, y_train,
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_split=rf_min_samples_split
                )
                rf_metrics = ml.evaluate_model(rf_model, X_test, y_test, "Random Forest")
                st.session_state.rf_model = rf_model
                st.session_state.rf_metrics = rf_metrics
                st.success("Random Forest trained successfully!")
    
    # Display Results
    if st.session_state.lr_metrics is not None:
        st.subheader("📈 Logistic Regression Results")
        display_model_results(st.session_state.lr_metrics, "Logistic Regression")
    
    if st.session_state.rf_metrics is not None:
        st.subheader("📈 Random Forest Results")
        display_model_results(st.session_state.rf_metrics, "Random Forest")
    
    # Model Comparison
    if st.session_state.lr_metrics is not None and st.session_state.rf_metrics is not None:
        st.subheader("📊 Model Comparison")
        comparison = ml.compare_models(st.session_state.lr_metrics, st.session_state.rf_metrics)
        
        st.write("**Metrics Comparison:**")
        st.dataframe(comparison['metrics_comparison'], width='stretch')
        
        st.write(f"**Best Overall Model:** {comparison['best_model']}")
        
        st.write("**Insights:**")
        for insight in comparison['insights']:
            st.text(insight)


def display_model_results(metrics, model_name):
    """Display model evaluation results."""
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    with col5:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else "N/A")
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        roc_fig = ml.plot_roc_curve(metrics['y_test'], metrics['y_pred_proba'], model_name)
        st.pyplot(roc_fig)
    with col2:
        cm_fig = ml.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
        st.pyplot(cm_fig)


def show_prediction_interface():
    """Display prediction interface page."""
    st.header("🔮 Prediction Interface")
    
    if st.session_state.lr_model is None and st.session_state.rf_model is None:
        st.warning("Please train at least one model in the 'Model Training' page first!")
        return
    
    # Model Selection
    available_models = []
    if st.session_state.lr_model is not None:
        available_models.append("Logistic Regression")
    if st.session_state.rf_model is not None:
        available_models.append("Random Forest")
    
    selected_model = st.selectbox("Select a trained model:", available_models)
    
    # Get feature names
    if st.session_state.df_processed is not None:
        feature_cols = st.session_state.df_processed.drop(columns=['target']).columns.tolist()
    else:
        st.error("Please load data first!")
        return
    
    # Input form
    st.subheader("Enter Patient Information")
    
    # Organize inputs in columns
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    with col1:
        inputs['age'] = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
        inputs['sex'] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        inputs['cp'] = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
                                    format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        inputs['trestbps'] = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120, step=1)
        inputs['chol'] = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200, step=1)
    
    with col2:
        inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], 
                                    format_func=lambda x: "No" if x == 0 else "Yes")
        inputs['restecg'] = st.selectbox("Resting ECG (restecg)", [0, 1, 2],
                                        format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
        inputs['thalach'] = st.number_input("Maximum Heart Rate (thalach)", min_value=0, max_value=250, value=150, step=1)
        inputs['exang'] = st.selectbox("Exercise Induced Angina (exang)", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
        inputs['oldpeak'] = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    with col3:
        inputs['slope'] = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2],
                                      format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        inputs['ca'] = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        inputs['thal'] = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3],
                                     format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x] if x < 3 else "Unknown")
    
    # Prediction button
    if st.button("🔮 Predict", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame([inputs])
        input_data = input_data[feature_cols]  # Ensure correct column order
        
        # Get selected model
        if selected_model == "Logistic Regression":
            model = st.session_state.lr_model
        else:
            model = st.session_state.rf_model
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"**Predicted Class: Disease Present** ❤️‍🩹")
            else:
                st.success(f"**Predicted Class: No Disease** ✅")
        
        with col2:
            st.metric("Probability of Disease", f"{prediction_proba[1]:.3f}")
            st.metric("Probability of No Disease", f"{prediction_proba[0]:.3f}")
        
        # Probability visualization
        prob_fig, ax = plt.subplots(figsize=(8, 4))
        classes = ['No Disease', 'Disease']
        probs = [prediction_proba[0], prediction_proba[1]]
        colors = ['green' if p < 0.5 else 'red' for p in probs]
        ax.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            ax.text(i, prob + 0.02, f'{prob:.3f}', ha='center', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(prob_fig)


if __name__ == "__main__":
    main()

