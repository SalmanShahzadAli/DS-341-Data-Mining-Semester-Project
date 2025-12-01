"""
DS341 - DATA MINING PROJECT
Section 6: CLASSIFICATION MODELS
Build and evaluate Decision Tree, Naive Bayes, and KNN classifiers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from imports_setup import print_section_header, print_subsection_header
from preprocessing import preprocess_complete_pipeline
from dataset_loading import load_dataset

# ============================================================================
# PREPARE DATA FOR CLASSIFICATION
# ============================================================================

def prepare_classification_data(df):
    """
    Prepare features and target variable for classification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    
    Returns:
    --------
    tuple : (X, y, column_info)
    """
    print_subsection_header("Preparing Data for Classification")
    
    df_classified = df.copy()
    numerical_cols = df_classified.select_dtypes(include=[np.number]).columns
    categorical_cols = df_classified.select_dtypes(include=['object']).columns
    
    # Create target variable (binary classification)
    # Use the first numerical column as basis for target
    if len(numerical_cols) > 0:
        target_col = numerical_cols[0]
        df_classified['target'] = (df_classified[target_col] > df_classified[target_col].quantile(0.75)).astype(int)
        print(f"Target Variable Created: High vs Low based on {target_col}")
        print(f"  Threshold: {df_classified[target_col].quantile(0.75):.2f}")
    else:
        print("No numerical columns found for target creation")
        return None, None, None
    
    # Class distribution
    print(f"\nClass Distribution:")
    print(f"  Class 0 (Low): {(df_classified['target'] == 0).sum()} ({(df_classified['target'] == 0).sum()/len(df_classified)*100:.1f}%)")
    print(f"  Class 1 (High): {(df_classified['target'] == 1).sum()} ({(df_classified['target'] == 1).sum()/len(df_classified)*100:.1f}%)")
    
    # Encode categorical variables
    print(f"\nEncoding Categorical Features:")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_classified[col] = le.fit_transform(df_classified[col].astype(str))
        le_dict[col] = le
        print(f"  ✓ {col}: encoded")
    
    # Prepare features and target
    X = df_classified.select_dtypes(include=[np.number]).drop('target', axis=1, errors='ignore')
    y = df_classified['target']
    
    print(f"\nFeatures Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")
    
    return X, y, {'numerical_cols': numerical_cols, 'categorical_cols': categorical_cols}

# ============================================================================
# TRAIN TEST SPLIT
# ============================================================================

def split_and_scale_data(X, y, test_size=0.3, random_state=42):
    """
    Split and scale data for training and testing
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float
        Test set size (default: 0.3)
    random_state : int
        Random seed (default: 42)
    
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print_subsection_header("Splitting and Scaling Data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training Set Size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test Set Size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Data scaled using StandardScaler")
    print(f"  Mean of scaled training features ≈ 0: ✓")
    print(f"  Std Dev of scaled training features ≈ 1: ✓")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# TRAIN CLASSIFIERS
# ============================================================================

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree classifier"""
    print_subsection_header("Decision Tree Classifier")
    
    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    
    return dt_clf, y_pred

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Train Naive Bayes classifier"""
    print_subsection_header("Naive Bayes Classifier")
    
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    
    return nb_clf, y_pred

def train_knn(X_train, y_train, X_test, y_test, k=5):
    """Train KNN classifier"""
    print_subsection_header(f"K-Nearest Neighbors (K={k}) Classifier")
    
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)
    
    return knn_clf, y_pred

# ============================================================================
# EVALUATE CLASSIFIER
# ============================================================================

def evaluate_classifier(clf_name, y_test, y_pred):
    """
    Evaluate classifier performance
    
    Parameters:
    -----------
    clf_name : str
        Classifier name
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predicted labels
    
    Returns:
    --------
    dict : Performance metrics
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\n{clf_name} Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification Report
    print(f"\nClassification Report for {clf_name}:")
    print(classification_report(y_test, y_pred))
    
    return metrics

# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

def plot_confusion_matrix(clf_name, y_test, y_pred):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    clf_name : str
        Classifier name
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.title(f'Confusion Matrix - {clf_name}', fontweight='bold', fontsize=12)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(results_dict):
    """
    Compare performance of all models
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for all classifiers
    """
    print_subsection_header("Model Comparison & Ranking")
    
    comparison_df = pd.DataFrame(results_dict).T
    print("\nPerformance Comparison Table:")
    print(comparison_df.round(4))
    
    # Visualize comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Classification Model Performance Comparison', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Rank models by F1 score
    print("\nModel Ranking (by F1-Score):")
    rankings = comparison_df['f1'].sort_values(ascending=False)
    for rank, (model, score) in enumerate(rankings.items(), 1):
        print(f"  {rank}. {model}: {score:.4f}")
    
    best_model = rankings.index[0]
    print(f"\n🏆 Best Model: {best_model} (F1-Score: {rankings.iloc[0]:.4f})")
    
    return best_model

# ============================================================================
# COMPLETE CLASSIFICATION PIPELINE
# ============================================================================

def perform_classification(df):
    """
    Execute complete classification pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_section_header("CLASSIFICATION MODELS")
    
    # Step 1: Prepare data
    X, y, col_info = prepare_classification_data(df)
    
    if X is None:
        print("Failed to prepare classification data")
        return None
    
    # Step 2: Split and scale
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Step 3: Train classifiers
    print_section_header("TRAINING CLASSIFIERS")
    
    classifiers = {}
    predictions = {}
    
    # Decision Tree
    dt_clf, dt_pred = train_decision_tree(X_train_scaled, y_train, X_test_scaled, y_test)
    classifiers['Decision Tree'] = dt_clf
    predictions['Decision Tree'] = dt_pred
    
    # Naive Bayes
    nb_clf, nb_pred = train_naive_bayes(X_train_scaled, y_train, X_test_scaled, y_test)
    classifiers['Naive Bayes'] = nb_clf
    predictions['Naive Bayes'] = nb_pred
    
    # KNN
    knn_clf, knn_pred = train_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=5)
    classifiers['KNN (k=5)'] = knn_clf
    predictions['KNN (k=5)'] = knn_pred
    
    # Step 4: Evaluate classifiers
    print_section_header("EVALUATING CLASSIFIERS")
    
    results = {}
    for clf_name, y_pred in predictions.items():
        results[clf_name] = evaluate_classifier(clf_name, y_test, y_pred)
    
    # Step 5: Plot confusion matrices
    print_section_header("CONFUSION MATRICES")
    for clf_name, y_pred in predictions.items():
        plot_confusion_matrix(clf_name, y_test, y_pred)
    
    # Step 6: Compare models
    print_section_header("MODEL COMPARISON")
    best_model = compare_models(results)
    
    print_section_header("CLASSIFICATION COMPLETE")
    print(f"✓ Models trained: {len(classifiers)}")
    print(f"✓ Best model: {best_model}")
    print("✓ Ready for Clustering & Customer Segmentation")
    
    return {
        'classifiers': classifiers,
        'predictions': predictions,
        'results': results,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 6: CLASSIFICATION MODELS")
    
    # Load and preprocess dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        preprocessing_result = preprocess_complete_pipeline(df)
        df_cleaned = preprocessing_result['dataframe']
        
        # Perform classification
        classification_result = perform_classification(df_cleaned)
    else:
        print("Failed to load dataset.")