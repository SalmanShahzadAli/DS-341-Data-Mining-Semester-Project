import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, silhouette_score)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# MATPLOTLIB & SEABORN CONFIGURATION
# ============================================================================

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def print_subsection_header(title):
    """Print formatted subsection header"""
    print("\n" + "-"*80)
    print(title)
    print("-"*80)

# ============================================================================
# EXPORT CONFIGURATIONS
# ============================================================================

# Make all imports and configurations available when module is imported
__all__ = [
    'pd', 'np', 'plt', 'sns',
    'StandardScaler', 'LabelEncoder',
    'train_test_split',
    'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier',
    'KMeans', 'PCA',
    'classification_report', 'confusion_matrix', 'accuracy_score',
    'precision_score', 'recall_score', 'f1_score', 'silhouette_score',
    'apriori', 'association_rules', 'TransactionEncoder',
    'print_section_header', 'print_subsection_header'
]

print_section_header("DS341 DATA MINING PROJECT - IMPORTS INITIALIZED")
print("✓ All libraries loaded successfully")
print("✓ Plotting configuration applied")
print("✓ Utility functions ready")
print("="*80)