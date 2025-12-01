"""
DS341 - DATA MINING PROJECT
Section 4: EXPLORATORY DATA ANALYSIS (EDA)
Analyze patterns, trends, and relationships in the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imports_setup import print_section_header, print_subsection_header
from preprocessing import preprocess_complete_pipeline
from dataset_loading import load_dataset

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def descriptive_statistics(df):
    """
    Generate comprehensive descriptive statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_subsection_header("Descriptive Statistics - Numerical Features")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numerical_cols].describe()
    print(stats)
    
    return stats

# ============================================================================
# CATEGORICAL FEATURES ANALYSIS
# ============================================================================

def categorical_analysis(df):
    """
    Analyze categorical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_subsection_header("Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print(f"\n📊 {col}:")
        print(f"  Unique Values: {df[col].nunique()}")
        print(f"\n  Value Counts:")
        print(df[col].value_counts())
        print("-" * 60)

# ============================================================================
# NUMERICAL FEATURES VISUALIZATION
# ============================================================================

def plot_numerical_distributions(df, max_cols=4):
    """
    Visualize distributions of numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    max_cols : int
        Maximum number of columns to plot (default: 4)
    """
    print_subsection_header("Visualizing Numerical Features Distribution")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = min(len(numerical_cols), max_cols)
    
    if n_cols == 0:
        print("No numerical columns found")
        return
    
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols[:max_cols]):
        ax = axes[idx]
        ax.hist(df[col], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        ax.set_title(f'Distribution of {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Plotted {n_cols} numerical features")

# ============================================================================
# CATEGORICAL FEATURES VISUALIZATION
# ============================================================================

def plot_categorical_distributions(df, max_cols=4, max_categories=20):
    """
    Visualize distributions of categorical features (SAFE VERSION)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    max_cols : int
        Maximum number of columns to plot
    max_categories : int
        Maximum unique values allowed per column (prevents freeze)
    """
    print_subsection_header("Visualizing Categorical Features Distribution")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    cols_to_plot = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= max_categories:
            cols_to_plot.append(col)
        else:
            print(f"⚠️  Skipping '{col}' - too many unique values ({unique_count}) > {max_categories}")
    
    n_cols = min(len(cols_to_plot), max_cols)
    
    if n_cols == 0:
        print("No categorical columns with reasonable cardinality to plot")
        return
    
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, col in enumerate(cols_to_plot[:max_cols]):
        ax = axes[idx]
        counts = df[col].value_counts().head(20)  # Top 20 only
        counts.plot(kind='bar', ax=ax, color='coral', alpha=0.8, edgecolor='black')
        ax.set_title(f'Distribution of {col} (Top 20)', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Plotted {n_cols} categorical features safely")
# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(df):
    """
    Analyze correlation between numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_subsection_header("Correlation Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        print("Not enough numerical features for correlation analysis")
        return
    
    correlation_matrix = df[numerical_cols].corr()
    print(correlation_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    print_subsection_header("Strong Correlations (|r| > 0.7)")
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                strong_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    if strong_corr:
        for col1, col2, corr in strong_corr:
            print(f"  • {col1} ↔ {col2}: {corr:.4f}")
    else:
        print("  No strong correlations found")

# ============================================================================
# OUTLIERS ANALYSIS
# ============================================================================

def outliers_analysis(df):
    """
    Analyze outliers in numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_subsection_header("Outliers Analysis - Box Plots")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        print("No numerical columns found")
        return
    
    n_cols = min(len(numerical_cols), 4)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols[:4]):
        ax = axes[idx]
        ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(f'Box Plot: {col}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
    
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Box plots generated for {n_cols} numerical features")

# ============================================================================
# MISSING VALUES HEATMAP
# ============================================================================

def missing_values_heatmap(df):
    """
    Visualize missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset (before preprocessing)
    """
    print_subsection_header("Missing Values Heatmap")
    
    missing_data = df.isnull()
    
    if missing_data.sum().sum() == 0:
        print("✓ No missing values in the dataset")
        return
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(missing_data.iloc[:100, :], yticklabels=False, 
                cbar_kws={'label': 'Missing (White) vs Present (Blue)'}, cmap='viridis')
    plt.title('Missing Values Heatmap (First 100 rows)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

# ============================================================================
# SKEWNESS & KURTOSIS ANALYSIS
# ============================================================================

def skewness_kurtosis_analysis(df):
    """
    Analyze skewness and kurtosis of numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_subsection_header("Skewness & Kurtosis Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        print("No numerical columns found")
        return
    
    skew_kurt_df = pd.DataFrame({
        'Feature': numerical_cols,
        'Skewness': [df[col].skew() for col in numerical_cols],
        'Kurtosis': [df[col].kurtosis() for col in numerical_cols]
    })
    
    print(skew_kurt_df.to_string(index=False))
    print("\nInterpretation:")
    print("  • Skewness: -0.5 to 0.5 (Normal), > 0.5 (Right-skewed), < -0.5 (Left-skewed)")
    print("  • Kurtosis: ~3 (Normal), > 3 (Heavy-tailed), < 3 (Light-tailed)")

# ============================================================================
# COMPLETE EDA PIPELINE
# ============================================================================

def perform_eda(df):
    """
    Execute complete EDA pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_section_header("EXPLORATORY DATA ANALYSIS (EDA)")
    
    # 1. Descriptive Statistics
    descriptive_statistics(df)
    
    # 2. Categorical Analysis
    categorical_analysis(df)
    
    # 3. Numerical Distributions
    plot_numerical_distributions(df, max_cols=4)
    
    # 4. Categorical Distributions
    plot_categorical_distributions(df, max_cols=4,max_categories=30)
    
    # 5. Correlation Analysis
    correlation_analysis(df)
    
    # 6. Outliers Analysis
    outliers_analysis(df)
    
    # 7. Skewness & Kurtosis
    skewness_kurtosis_analysis(df)
    
    print_section_header("EDA COMPLETE")
    print("✓ Exploratory Data Analysis completed successfully")
    print("✓ Ready for Association Rule Mining")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 4: EXPLORATORY DATA ANALYSIS")
    
    # Load and preprocess dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        preprocessing_result = preprocess_complete_pipeline(df)
        df_cleaned = preprocessing_result['dataframe']
        
        # Perform EDA
        perform_eda(df_cleaned)
    else:
        print("Failed to load dataset.")