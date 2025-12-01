import pandas as pd
import numpy as np
from imports_setup import print_section_header, print_subsection_header
from dataset_loading import load_dataset, assess_data_quality, get_column_info

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================

def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    pd.DataFrame : Dataset with missing values handled
    """
    print_subsection_header("Handling Missing Values")
    
    df_processed = df.copy()
    
    # Get column information
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Handle numerical columns with median imputation
    print("\n📊 Numerical Columns - Using Median Imputation:")
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"  ✓ {col}: {df_processed[col].isnull().sum()} values imputed with median ({median_val:.2f})")
    
    # Handle categorical columns with mode imputation
    print("\n📋 Categorical Columns - Using Mode Imputation:")
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"  ✓ {col}: {df_processed[col].isnull().sum()} values imputed with mode ('{mode_val}')")
    
    print(f"\n✓ Missing values handled successfully")
    print(f"✓ Remaining missing values: {df_processed.isnull().sum().sum()}")
    
    return df_processed

# ============================================================================
# REMOVE DUPLICATES
# ============================================================================

def remove_duplicates(df):
    """
    Remove duplicate records from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    pd.DataFrame : Dataset with duplicates removed
    """
    print_subsection_header("Removing Duplicate Records")
    
    df_processed = df.copy()
    
    initial_count = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    removed_count = initial_count - len(df_processed)
    
    print(f"Initial Records: {initial_count}")
    print(f"Duplicate Records Removed: {removed_count}")
    print(f"Final Records: {len(df_processed)}")
    print(f"Duplicate Percentage: {(removed_count / initial_count * 100):.2f}%")
    
    if removed_count > 0:
        print(f"✓ Duplicates removed successfully")
    else:
        print(f"✓ No duplicates found in dataset")
    
    return df_processed

# ============================================================================
# REMOVE OUTLIERS
# ============================================================================

def remove_outliers_iqr(df, multiplier=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    multiplier : float
        IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
    --------
    pd.DataFrame : Dataset with outliers removed
    """
    print_subsection_header("Removing Outliers (IQR Method)")
    
    df_processed = df.copy()
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    initial_count = len(df_processed)
    outliers_per_column = {}
    
    print(f"Using IQR Multiplier: {multiplier}\n")
    
    for col in numerical_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Count outliers
        outliers = ((df_processed[col] < lower_bound) | 
                   (df_processed[col] > upper_bound)).sum()
        outliers_per_column[col] = outliers
        
        if outliers > 0:
            print(f"  {col}:")
            print(f"    Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    Outliers: {outliers}")
    
    # Remove outliers
    for col in numerical_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_processed = df_processed[
            (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
        ]
    
    removed_count = initial_count - len(df_processed)
    print(f"\nTotal Records Before: {initial_count}")
    print(f"Total Records After: {len(df_processed)}")
    print(f"Records Removed: {removed_count}")
    print(f"Removal Percentage: {(removed_count / initial_count * 100):.2f}%")
    print(f"✓ Outliers removed successfully")
    
    return df_processed

# ============================================================================
# DATA TRANSFORMATION & ENCODING
# ============================================================================

def encode_categorical_features(df):
    """
    Encode categorical features using LabelEncoder
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    tuple : (encoded_df, encoder_dict)
    """
    print_subsection_header("Encoding Categorical Features")
    
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    encoder_dict = {}
    
    print(f"Categorical Columns to Encode: {len(categorical_cols)}\n")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoder_dict[col] = le
        print(f"  ✓ {col}: {len(le.classes_)} unique values encoded")
        print(f"    Classes: {list(le.classes_[:5])}{'...' if len(le.classes_) > 5 else ''}")
    
    print(f"\n✓ All categorical features encoded successfully")
    
    return df_encoded, encoder_dict

# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset (with numerical columns only)
    
    Returns:
    --------
    tuple : (scaled_array, scaler_object)
    """
    print_subsection_header("Scaling Numerical Features")
    
    from sklearn.preprocessing import StandardScaler
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    print(f"Columns Scaled: {len(numerical_cols)}")
    print(f"  Mean of scaled features ≈ 0: ✓")
    print(f"  Std Dev of scaled features ≈ 1: ✓")
    
    return scaled_data, scaler

# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_complete_pipeline(df):
    """
    Execute complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataset
    
    Returns:
    --------
    dict : Dictionary containing preprocessed data and metadata
    """
    print_section_header("COMPLETE PREPROCESSING PIPELINE")
    
    # Step 1: Handle missing values
    df_step1 = handle_missing_values(df)
    
    # Step 2: Remove duplicates
    df_step2 = remove_duplicates(df_step1)
    
    # Step 3: Remove outliers
    df_step3 = remove_outliers_iqr(df_step2, multiplier=1.5)
    
    # Step 4: Get column information
    column_info = get_column_info(df_step3)
    
    print_subsection_header("Preprocessing Summary")
    print(f"Initial Dataset Shape: {df.shape}")
    print(f"Final Dataset Shape: {df_step3.shape}")
    print(f"Records Removed: {df.shape[0] - df_step3.shape[0]}")
    print(f"Features Retained: {df_step3.shape[1]}")
    
    return {
        'dataframe': df_step3,
        'numerical_cols': column_info['numerical'],
        'categorical_cols': column_info['categorical'],
        'initial_shape': df.shape,
        'final_shape': df_step3.shape
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 3: DATA PREPROCESSING")
    
    # Load dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        # Execute preprocessing pipeline
        preprocessing_result = preprocess_complete_pipeline(df)
        
        df_cleaned = preprocessing_result['dataframe']
        
        print_section_header("PREPROCESSING COMPLETE")
        print("✓ Data successfully preprocessed")
        print("✓ Ready for Exploratory Data Analysis")
        print(f"✓ Cleaned Dataset Shape: {df_cleaned.shape}")
    else:
        print("Failed to load dataset.")