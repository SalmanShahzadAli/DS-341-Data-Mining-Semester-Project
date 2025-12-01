import pandas as pd
import numpy as np
from imports_setup import print_section_header, print_subsection_header

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(filepath='ecommerce_customer_data_custom_ratios.csv'):
    """
    Load the e-commerce dataset from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file (default: 'ecommerce_customer_data_custom_ratios.csv')
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

# ============================================================================
# DATASET OVERVIEW
# ============================================================================

def display_dataset_overview(df):
    """
    Display comprehensive overview of the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    """
    print_section_header("DATASET OVERVIEW")
    
    # Basic Information
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   • Total Records: {df.shape[0]}")
    print(f"   • Total Features: {df.shape[1]}")
    
    # First few rows
    print_subsection_header("First 5 Rows")
    print(df.head())
    
    # Last few rows
    print_subsection_header("Last 5 Rows")
    print(df.tail())
    
    # Data types
    print_subsection_header("Data Types")
    print(df.dtypes)
    
    # Dataset Info
    print_subsection_header("Detailed Dataset Info")
    print(df.info())
    
    return {
        'shape': df.shape,
        'records': df.shape[0],
        'features': df.shape[1]
    }

# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df):
    """
    Assess data quality issues in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    dict : Dictionary containing quality metrics
    """
    print_section_header("DATA QUALITY ASSESSMENT")
    
    quality_report = {}
    
    # Missing values
    print_subsection_header("Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percentage.values
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df)
    quality_report['missing_values'] = missing_df
    
    # Duplicates
    print_subsection_header("Duplicate Records")
    duplicate_count = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicate_count}")
    print(f"Duplicate Percentage: {(duplicate_count / len(df)) * 100:.2f}%")
    quality_report['duplicates'] = duplicate_count
    
    # Feature Statistics
    print_subsection_header("Numerical Features Statistics")
    numerical_stats = df.describe()
    print(numerical_stats)
    quality_report['numerical_stats'] = numerical_stats
    
    # Categorical Features Statistics
    print_subsection_header("Categorical Features Summary")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique Values: {df[col].nunique()}")
        print(f"  Top 5 Values:")
        print(df[col].value_counts().head())
    
    quality_report['categorical_cols'] = categorical_cols
    
    # Memory usage
    print_subsection_header("Memory Usage")
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total Memory Usage: {memory_usage:.2f} MB")
    quality_report['memory_usage'] = memory_usage
    
    return quality_report

# ============================================================================
# COLUMN INFORMATION
# ============================================================================

def get_column_info(df):
    """
    Get detailed information about all columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    dict : Dictionary with column information
    """
    print_subsection_header("Column Information")
    
    column_info = {
        'numerical': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    
    print(f"\nNumerical Columns ({len(column_info['numerical'])}):")
    print(f"  {column_info['numerical']}")
    
    print(f"\nCategorical Columns ({len(column_info['categorical'])}):")
    print(f"  {column_info['categorical']}")
    
    if column_info['datetime']:
        print(f"\nDatetime Columns ({len(column_info['datetime'])}):")
        print(f"  {column_info['datetime']}")
    
    return column_info

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 2: DATASET LOADING & EXPLORATION")
    
    # Load dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        # Display overview
        overview = display_dataset_overview(df)
        
        # Assess quality
        quality = assess_data_quality(df)
        
        # Get column info
        cols = get_column_info(df)
        
        print_section_header("DATASET LOADING COMPLETE")
        print("✓ Dataset successfully loaded and explored")
        print("✓ Ready for preprocessing stage")
    else:
        print("Failed to load dataset. Please check the file path.")