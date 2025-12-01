"""
DS341 - DATA MINING PROJECT
Section 5: ASSOCIATION RULE MINING
Extract co-purchase patterns and relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from imports_setup import print_section_header, print_subsection_header
from preprocessing import preprocess_complete_pipeline
from dataset_loading import load_dataset

# ============================================================================
# DATA PREPARATION FOR MARKET BASKET ANALYSIS
# ============================================================================

def prepare_binary_data(df):
    """
    Prepare binary data for Association Rule Mining — SAFE & MEANINGFUL VERSION
    Focus: Co-occurrence of customer behavioral patterns (not individual customers/products)
    """
    print_subsection_header("Preparing Binary Data for Association Rules (Behavioral Patterns)")

    df_encoded = df.copy()
    binary_data = pd.DataFrame(index=df_encoded.index)

    # ===================================================================
    # 1. HIGH-VALUE CUSTOMER SEGMENTS (Behavioral Bins)
    # ===================================================================
    print("Creating behavioral segments...")

    # Purchase Frequency Bins
    freq_bins = pd.qcut(df_encoded['Purchase Frequency'], q=3, duplicates='drop')
    binary_data['Freq_Low']  = (freq_bins == freq_bins.cat.categories[0]).astype(int)
    binary_data['Freq_Medium'] = (freq_bins == freq_bins.cat.categories[1] if len(freq_bins.cat.categories) > 1 else False).astype(int)
    binary_data['Freq_High'] = (freq_bins == freq_bins.cat.categories[-1]).astype(int)

    # Monetary Value Bins
    monetary_bins = pd.qcut(df_encoded['Monetary Value'], q=3, duplicates='drop')
    binary_data['Spend_Low']  = (monetary_bins == monetary_bins.cat.categories[0]).astype(int)
    binary_data['Spend_Medium'] = (monetary_bins == monetary_bins.cat.categories[1] if len(monetary_bins.cat.categories) > 1 else False).astype(int)
    binary_data['Spend_High'] = (monetary_bins == monetary_bins.cat.categories[-1]).astype(int)

    # Recency Bins
    recency_bins = pd.qcut(df_encoded['Recency'], q=3, duplicates='drop')
    binary_data['Recent_Active'] = (recency_bins == recency_bins.cat.categories[0]).astype(int)  # Low recency = recently active
    binary_data['Moderately_Active'] = (recency_bins == recency_bins.cat.categories[1] if len(recency_bins.cat.categories) > 1 else False).astype(int)
    binary_data['Inactive'] = (recency_bins == recency_bins.cat.categories[-1]).astype(int)

    # ===================================================================
    # 2. CATEGORICAL FEATURES (Only Low Cardinality!)
    # ===================================================================
    safe_categorical = ['Gender', 'Location', 'Product Category', 'Payment Method', 'Category Preference']
    
    print("\nEncoding safe categorical features...")
    for col in safe_categorical:
        if col not in df_encoded.columns:
            continue
        unique_count = df_encoded[col].nunique()
        if unique_count > 20:
            print(f"  Skipping {col}: too many categories ({unique_count})")
            continue
        
        dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)
        binary_data = pd.concat([binary_data, dummies], axis=1)
        print(f"  Added {col}: {unique_count} → {dummies.shape[1]} columns")

    # ===================================================================
    # 3. DEVICE & BROWSER (Common in e-commerce)
    # ===================================================================
    if 'Device' in df_encoded.columns:
        binary_data['Device_Mobile'] = (df_encoded['Device'] == 'Mobile').astype(int)
        binary_data['Device_Desktop'] = (df_encoded['Device'] == 'Desktop').astype(int)

    if 'Browser' in df_encoded.columns:
        top_browsers = df_encoded['Browser'].value_counts().head(5).index
        for browser in top_browsers:
            binary_data[f'Browser_{browser}'] = (df_encoded['Browser'] == browser).astype(int)

    # ===================================================================
    # 4. LOYALTY & CHURN RISK
    # ===================================================================
    binary_data['Loyal_Customer'] = (df_encoded['Customer Age'] > 365).astype(int)  # >1 year
    binary_data['High_Return_Rate'] = (df_encoded.get('Returns', 0) > df_encoded.get('Returns', 0).median()).astype(int)
    binary_data['Multiple_Purchases'] = (df_encoded['Purchase Frequency'] > 1).astype(int)

    # ===================================================================
    # FINAL RESULT
    # ===================================================================
    print(f"\nFinal Binary Matrix: {binary_data.shape[0]:,} rows × {binary_data.shape[1]} behavioral items")
    print("Sample items:", list(binary_data.columns[:15]), "...")

    return binary_data
# ============================================================================
# APPLY APRIORI ALGORITHM
# ============================================================================

def find_frequent_itemsets(binary_data, min_support=0.05):
    """
    Find frequent itemsets using Apriori algorithm
    
    Parameters:
    -----------
    binary_data : pd.DataFrame
        Binary encoded dataset
    min_support : float
        Minimum support threshold (default: 0.05)
    
    Returns:
    --------
    pd.DataFrame : Frequent itemsets
    """
    print_subsection_header("Applying Apriori Algorithm")
    print(f"Minimum Support: {min_support} ({min_support*100:.1f}%)")
    
    frequent_itemsets = apriori(binary_data, min_support=min_support, use_colnames=True)
    
    print(f"\nFrequent Itemsets Found: {len(frequent_itemsets)}")
    print("\nTop 10 Frequent Itemsets (by support):")
    print(frequent_itemsets.nlargest(10, 'support')[['support', 'itemsets']])
    
    # Visualize support distribution
    if len(frequent_itemsets) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(frequent_itemsets['support'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        plt.xlabel('Support')
        plt.ylabel('Frequency')
        plt.title('Distribution of Itemset Support', fontweight='bold', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return frequent_itemsets

# ============================================================================
# GENERATE ASSOCIATION RULES
# ============================================================================

def generate_association_rules(frequent_itemsets, min_confidence=0.3, min_lift=1.0):
    """
    Generate association rules from frequent itemsets
    
    Parameters:
    -----------
    frequent_itemsets : pd.DataFrame
        Frequent itemsets from Apriori
    min_confidence : float
        Minimum confidence threshold (default: 0.3)
    min_lift : float
        Minimum lift threshold (default: 1.0)
    
    Returns:
    --------
    pd.DataFrame : Association rules
    """
    print_subsection_header("Generating Association Rules")
    
    if len(frequent_itemsets) < 2:
        print("Not enough itemsets to generate rules")
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        print(f"No rules found with confidence >= {min_confidence}")
        return rules
    
    # Calculate additional metrics
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    
    # Round metrics for better display
    rules['support'] = rules['support'].round(4)
    rules['confidence'] = rules['confidence'].round(4)
    rules['lift'] = rules['lift'].round(4)
    
    print(f"Total Rules Generated: {len(rules)}")
    print(f"Minimum Confidence: {min_confidence}")
    print(f"Minimum Lift: {min_lift}")
    
    # Filter by lift if specified
    if min_lift > 1.0:
        rules = rules[rules['lift'] >= min_lift]
        print(f"Rules after Lift filter: {len(rules)}")
    
    return rules

# ============================================================================
# ANALYZE AND INTERPRET RULES
# ============================================================================

def analyze_rules(rules):
    """
    Analyze and interpret association rules
    
    Parameters:
    -----------
    rules : pd.DataFrame
        Association rules
    """
    print_subsection_header("Association Rules Analysis")
    
    if len(rules) == 0:
        print("No rules to analyze")
        return
    
    print(f"\nTotal Rules: {len(rules)}")
    
    # Statistics
    print(f"\nSupport Statistics:")
    print(f"  Mean: {rules['support'].mean():.4f}")
    print(f"  Min: {rules['support'].min():.4f}")
    print(f"  Max: {rules['support'].max():.4f}")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {rules['confidence'].mean():.4f}")
    print(f"  Min: {rules['confidence'].min():.4f}")
    print(f"  Max: {rules['confidence'].max():.4f}")
    
    print(f"\nLift Statistics:")
    print(f"  Mean: {rules['lift'].mean():.4f}")
    print(f"  Min: {rules['lift'].min():.4f}")
    print(f"  Max: {rules['lift'].max():.4f}")
    
    # Top rules by different metrics
    print_subsection_header("Top 5 Rules by Lift")
    top_lift = rules.nlargest(5, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    for idx, row in top_lift.iterrows():
        print(f"\n  {idx+1}. {row['antecedents']} → {row['consequents']}")
        print(f"     Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
    
    print_subsection_header("Top 5 Rules by Confidence")
    top_conf = rules.nlargest(5, 'confidence')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    for idx, row in top_conf.iterrows():
        print(f"\n  {idx+1}. {row['antecedents']} → {row['consequents']}")
        print(f"     Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
    
    print_subsection_header("Top 5 Rules by Support")
    top_sup = rules.nlargest(5, 'support')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    for idx, row in top_sup.iterrows():
        print(f"\n  {idx+1}. {row['antecedents']} → {row['consequents']}")
        print(f"     Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")

# ============================================================================
# VISUALIZE ASSOCIATION RULES
# ============================================================================

def visualize_rules(rules):
    """
    Visualize association rules
    
    Parameters:
    -----------
    rules : pd.DataFrame
        Association rules
    """
    print_subsection_header("Visualizing Association Rules")
    
    if len(rules) == 0:
        print("No rules to visualize")
        return
    
    # Scatter plot: Support vs Confidence colored by Lift
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Support vs Confidence
    scatter = axes[0].scatter(rules['support'], rules['confidence'], 
                             c=rules['lift'], s=100, alpha=0.6, cmap='viridis', edgecolors='black')
    axes[0].set_xlabel('Support', fontweight='bold')
    axes[0].set_ylabel('Confidence', fontweight='bold')
    axes[0].set_title('Association Rules: Support vs Confidence\n(colored by Lift)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=axes[0])
    cbar1.set_label('Lift')
    
    # Plot 2: Confidence vs Lift
    scatter2 = axes[1].scatter(rules['confidence'], rules['lift'], 
                              c=rules['support'], s=100, alpha=0.6, cmap='plasma', edgecolors='black')
    axes[1].set_xlabel('Confidence', fontweight='bold')
    axes[1].set_ylabel('Lift', fontweight='bold')
    axes[1].set_title('Association Rules: Confidence vs Lift\n(colored by Support)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Support')
    
    plt.tight_layout()
    plt.show()
    
    # Bar plot: Top rules by lift
    if len(rules) > 0:
        top_rules = rules.nlargest(10, 'lift').copy()
        top_rules['rule'] = top_rules.apply(
            lambda x: f"{str(x['antecedents'])[:20]}→{str(x['consequents'])[:20]}", axis=1
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top_rules)), top_rules['lift'], color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Rules', fontweight='bold')
        ax.set_ylabel('Lift', fontweight='bold')
        ax.set_title('Top 10 Association Rules by Lift', fontweight='bold')
        ax.set_xticks(range(len(top_rules)))
        ax.set_xticklabels([f"Rule {i+1}" for i in range(len(top_rules))], rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# COMPLETE ARM PIPELINE
# ============================================================================

def perform_association_rule_mining(df):
    """
    Execute complete Association Rule Mining pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_section_header("ASSOCIATION RULE MINING")
    
    # Step 1: Prepare binary data
    binary_data = prepare_binary_data(df)
    
    # Step 2: Find frequent itemsets
    frequent_itemsets = find_frequent_itemsets(binary_data, min_support=0.10)
    
    if len(frequent_itemsets) < 2:
        print("\n⚠ Not enough frequent itemsets for rules generation")
        return None, None
    
    # Step 3: Generate association rules
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.2)
    
    if len(rules) > 0:
        # Step 4: Analyze rules
        analyze_rules(rules)
        
        # Step 5: Visualize rules
        visualize_rules(rules)
    
    print_section_header("ASSOCIATION RULE MINING COMPLETE")
    print(f"✓ Frequent Itemsets: {len(frequent_itemsets)}")
    print(f"✓ Association Rules: {len(rules)}")
    print("✓ Ready for Classification Models")
    
    return frequent_itemsets, rules

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 5: ASSOCIATION RULE MINING")
    
    # Load and preprocess dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        preprocessing_result = preprocess_complete_pipeline(df)
        df_cleaned = preprocessing_result['dataframe']
        
        # Perform ARM
        frequent_itemsets, rules = perform_association_rule_mining(df_cleaned)
    else:
        print("Failed to load dataset.")