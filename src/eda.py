"""
Exploratory Data Analysis Module
Provides comprehensive EDA functions including visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


def get_dataset_overview(df):
    """
    Get dataset overview including head, shape, and data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dictionary containing overview information
    """
    overview = {
        'head': df.head(10),
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'info': df.info()
    }
    return overview


def analyze_missing_values(df):
    """
    Analyze and visualize missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Missing value analysis results
    """
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    analysis = {
        'missing_count': missing_data.to_dict(),
        'missing_percent': missing_percent.to_dict(),
        'total_missing': missing_data.sum(),
        'columns_with_missing': missing_data[missing_data > 0].index.tolist()
    }
    
    return analysis


def visualize_missing_values(df, figsize=(10, 6)):
    """
    Create visualization for missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    missing_data.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_ylabel('Number of Missing Values', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig


def analyze_target_variable(df, target_col='target'):
    """
    Analyze target variable distribution and class balance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    dict
        Target variable analysis results
    """
    target_counts = df[target_col].value_counts()
    target_percent = df[target_col].value_counts(normalize=True) * 100
    
    analysis = {
        'value_counts': target_counts.to_dict(),
        'percentages': target_percent.to_dict(),
        'is_balanced': abs(target_percent.iloc[0] - target_percent.iloc[1]) < 10 if len(target_percent) == 2 else None
    }
    
    return analysis


def visualize_target_distribution(df, target_col='target', figsize=(10, 5)):
    """
    Visualize target variable distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Name of target column
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    target_counts = df[target_col].value_counts()
    axes[0].bar(target_counts.index.astype(str), target_counts.values, color=['skyblue', 'salmon'])
    axes[0].set_title('Target Variable Distribution (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Target Class', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Disease (0)', 'Disease (1)'])
    
    # Pie chart
    target_percent = df[target_col].value_counts(normalize=True) * 100
    axes[1].pie(target_percent.values, labels=['No Disease (0)', 'Disease (1)'], 
                autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    axes[1].set_title('Target Variable Distribution (Percentage)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_histograms(df, columns=None, figsize=(15, 10)):
    """
    Create histograms for key features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        List of columns to plot. If None, selects numeric columns automatically.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if columns is None:
        # Select numeric columns (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        columns = numeric_cols[:5]  # Select first 5 numeric columns
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_boxplots(df, columns=None, figsize=(15, 10)):
    """
    Create boxplots for outlier detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        List of columns to plot. If None, selects numeric columns automatically.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if columns is None:
        # Select numeric columns (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        columns = numeric_cols[:5]  # Select first 5 numeric columns
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            box_data = df[col].dropna()
            axes[idx].boxplot(box_data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[idx].set_title(f'Boxplot of {col}', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(col, fontsize=10)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(df, figsize=(12, 10)):
    """
    Create correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def generate_eda_insights(df, target_col='target'):
    """
    Generate written insights from EDA analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Name of target column
        
    Returns:
    --------
    dict
        Dictionary containing insights for each analysis
    """
    insights = {}
    
    # Target variable insights
    target_analysis = analyze_target_variable(df, target_col)
    target_counts = target_analysis['value_counts']
    target_percent = target_analysis['percentages']
    
    insights['target'] = f"""
    Target Variable Analysis:
    - Class 0 (No Disease): {target_counts.get(0, 0)} samples ({target_percent.get(0, 0):.1f}%)
    - Class 1 (Disease): {target_counts.get(1, 0)} samples ({target_percent.get(1, 0):.1f}%)
    - The dataset is {'relatively balanced' if target_analysis['is_balanced'] else 'imbalanced'}
    """
    
    # Missing values insights
    missing_analysis = analyze_missing_values(df)
    if missing_analysis['total_missing'] == 0:
        insights['missing_values'] = """
        Missing Values Analysis:
        - No missing values found in the dataset
        - Data quality is excellent, no imputation needed
        """
    else:
        insights['missing_values'] = f"""
        Missing Values Analysis:
        - Total missing values: {missing_analysis['total_missing']}
        - Columns with missing values: {', '.join(missing_analysis['columns_with_missing'])}
        """
    
    # Correlation insights
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    top_correlated = target_corr[target_corr.index != target_col].head(3)
    
    insights['correlation'] = f"""
    Correlation Analysis:
    - Top 3 features most correlated with target:
      1. {top_correlated.index[0]}: {top_correlated.iloc[0]:.3f}
      2. {top_correlated.index[1]}: {top_correlated.iloc[1]:.3f}
      3. {top_correlated.index[2]}: {top_correlated.iloc[2]:.3f}
    """
    
    return insights


if __name__ == "__main__":
    # Test EDA functions
    print("=" * 50)
    print("EDA Module Test")
    print("=" * 50)
    
    from src import data_preprocessing as dp
    
    df = dp.load_data()
    if df is not None:
        df = dp.preprocess_data(df)
        
        print("\nDataset Overview:")
        overview = get_dataset_overview(df)
        print(f"Shape: {overview['shape']}")
        
        print("\nMissing Values Analysis:")
        missing = analyze_missing_values(df)
        print(f"Total missing: {missing['total_missing']}")
        
        print("\nTarget Variable Analysis:")
        target_analysis = analyze_target_variable(df)
        print(target_analysis)
        
        print("\nGenerating visualizations...")
        # Create output directory for visualizations
        from config import OUTPUT_FIGURES_DIR
        output_dir = OUTPUT_FIGURES_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save visualizations
        try:
            # Target distribution
            target_fig = visualize_target_distribution(df)
            target_fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150, bbox_inches='tight')
            plt.close(target_fig)
            print(f"  [OK] Saved target_distribution.png to {output_dir}/")
            
            # Histograms
            hist_fig = create_histograms(df, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
            hist_fig.savefig(os.path.join(output_dir, "histograms.png"), dpi=150, bbox_inches='tight')
            plt.close(hist_fig)
            print(f"  [OK] Saved histograms.png to {output_dir}/")
            
            # Boxplots
            box_fig = create_boxplots(df, columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
            box_fig.savefig(os.path.join(output_dir, "boxplots.png"), dpi=150, bbox_inches='tight')
            plt.close(box_fig)
            print(f"  [OK] Saved boxplots.png to {output_dir}/")
            
            # Correlation heatmap
            corr_fig = create_correlation_heatmap(df)
            corr_fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close(corr_fig)
            print(f"  [OK] Saved correlation_heatmap.png to {output_dir}/")
            
            print(f"\n[OK] All visualizations saved successfully to '{output_dir}' directory!")
        except Exception as e:
            print(f"  [ERROR] Error generating visualizations: {e}")

