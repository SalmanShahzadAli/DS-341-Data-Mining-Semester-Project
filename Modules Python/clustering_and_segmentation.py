"""
DS341 - DATA MINING PROJECT
Section 7: CLUSTERING & CUSTOMER SEGMENTATION
Apply K-means clustering and customer segmentation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from imports_setup import print_section_header, print_subsection_header
from preprocessing import preprocess_complete_pipeline
from dataset_loading import load_dataset

# ============================================================================
# PREPARE DATA FOR CLUSTERING
# ============================================================================

def prepare_clustering_data(df):
    """
    Prepare features for clustering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    
    Returns:
    --------
    tuple : (X_cluster_scaled, scaler)
    """
    print_subsection_header("Preparing Data for Clustering")
    
    df_cluster = df.copy()
    numerical_cols = df_cluster.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cluster.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    print(f"Encoding Categorical Features: {len(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))
        print(f"  ✓ {col}: encoded")
    
    # Select all features for clustering
    X_cluster = df_cluster.select_dtypes(include=[np.number])
    
    print(f"\nFeatures for Clustering: {X_cluster.shape[1]}")
    print(f"Samples for Clustering: {X_cluster.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    print(f"\n✓ Features standardized using StandardScaler")
    
    return X_cluster_scaled, scaler, df_cluster

# ============================================================================
# ELBOW METHOD & SILHOUETTE ANALYSIS
# ============================================================================

def find_optimal_clusters(X_scaled, k_range=range(2, 10)):
    """
    Find optimal number of clusters using Elbow method and Silhouette score
    
    Parameters:
    -----------
    X_scaled : np.array
        Scaled feature matrix
    k_range : range
        Range of k values to test (default: 2-10)
    
    Returns:
    --------
    tuple : (optimal_k, inertias, silhouette_scores)
    """
    print_subsection_header("Finding Optimal Number of Clusters")
    
    inertias = []
    silhouette_scores = []
    
    print(f"Testing K values: {list(k_range)}\n")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        silhouette = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette)
        
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette:.4f}")
    
    # Find optimal k using silhouette score
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    
    print(f"\n✓ Optimal K (by Silhouette Score): {optimal_k}")
    print(f"  Silhouette Score: {silhouette_scores[optimal_k-min(k_range)]:.4f}")
    
    return optimal_k, inertias, silhouette_scores, list(k_range)

# ============================================================================
# VISUALIZE ELBOW & SILHOUETTE
# ============================================================================

def plot_elbow_and_silhouette(k_values, inertias, silhouette_scores):
    """
    Plot Elbow method and Silhouette analysis
    
    Parameters:
    -----------
    k_values : list
        List of k values tested
    inertias : list
        Inertia values
    silhouette_scores : list
        Silhouette scores
    """
    print_subsection_header("Visualizing Clustering Analysis")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontweight='bold')
    ax1.set_title('Elbow Method For Optimal k', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Silhouette scores
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    optimal_k = k_values[np.argmax(silhouette_scores)]
    ax2.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    ax2.set_title('Silhouette Score Analysis', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# APPLY K-MEANS CLUSTERING
# ============================================================================

def apply_kmeans(X_scaled, optimal_k):
    """
    Apply K-means clustering with optimal k
    
    Parameters:
    -----------
    X_scaled : np.array
        Scaled feature matrix
    optimal_k : int
        Optimal number of clusters
    
    Returns:
    --------
    KMeans : Fitted KMeans object
    """
    print_subsection_header("Applying K-Means Clustering")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"K-Means Clustering Complete")
    print(f"  Number of Clusters: {optimal_k}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score: {silhouette_score(X_scaled, clusters):.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_scaled, clusters):.4f}")
    
    return kmeans, clusters

# ============================================================================
# VISUALIZE CLUSTERS
# ============================================================================

def visualize_clusters_pca(X_scaled, clusters, kmeans):
    """
    Visualize clusters using PCA
    
    Parameters:
    -----------
    X_scaled : np.array
        Scaled feature matrix
    clusters : np.array
        Cluster assignments
    kmeans : KMeans
        Fitted KMeans object
    """
    print_subsection_header("Visualizing Clusters (PCA)")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Transform cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                         cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Plot cluster centers
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', 
               s=300, edgecolors='black', linewidth=2, label='Centroids', zorder=5)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontweight='bold')
    plt.title('Customer Segmentation - K-Means Clusters (PCA Visualization)', 
             fontweight='bold', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Explained Variance Ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  Total: {sum(pca.explained_variance_ratio_):.2%}")

# ============================================================================
# CLUSTER CHARACTERISTICS
# ============================================================================

def analyze_cluster_characteristics(df, clusters):
    """
    Analyze characteristics of each cluster
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset with cluster assignments
    clusters : np.array
        Cluster assignments
    """
    print_subsection_header("Cluster Characteristics Analysis")
    
    df['Cluster'] = clusters
    numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('Cluster', errors='ignore')
    
    print(f"\nCluster Distribution:")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = count / len(df) * 100
        print(f"  Cluster {cluster_id}: {count} customers ({pct:.1f}%)")
    
    # Detailed statistics per cluster
    print(f"\nDetailed Cluster Characteristics:")
    
    for cluster_id in sorted(df['Cluster'].unique()):
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id} (n={cluster_counts[cluster_id]})")
        print(f"{'='*60}")
        
        cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
        
        # Show statistics
        stats = cluster_data.describe().loc[['mean', 'std', 'min', 'max']]
        print("\nStatistics:")
        print(stats.round(2))

# ============================================================================
# VISUALIZE CLUSTER PROFILES
# ============================================================================

def visualize_cluster_profiles(df, clusters):
    """
    Visualize cluster profiles
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset
    clusters : np.array
        Cluster assignments
    """
    print_subsection_header("Visualizing Cluster Profiles")
    
    df_viz = df.copy()
    df_viz['Cluster'] = clusters
    
    numerical_cols = df_viz.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['Cluster']]
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for profiling")
        return
    
    # Plot cluster profiles
    n_features = min(len(numerical_cols), 4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols[:4]):
        ax = axes[idx]
        
        df_viz.boxplot(column=col, by='Cluster', ax=ax)
        ax.set_title(f'{col} by Cluster', fontweight='bold')
        ax.set_xlabel('Cluster', fontweight='bold')
        ax.set_ylabel(col, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()
    plt.show()

# ============================================================================
# COMPLETE CLUSTERING PIPELINE
# ============================================================================

def perform_clustering(df):
    """
    Execute complete clustering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    """
    print_section_header("CLUSTERING & CUSTOMER SEGMENTATION")
    
    # Step 1: Prepare data
    X_scaled, scaler, df_cluster = prepare_clustering_data(df)
    
    # Step 2: Find optimal clusters
    optimal_k, inertias, silhouette_scores, k_values = find_optimal_clusters(X_scaled, k_range=range(2, 10))
    
    # Step 3: Visualize elbow & silhouette
    plot_elbow_and_silhouette(k_values, inertias, silhouette_scores)
    
    # Step 4: Apply K-means
    kmeans, clusters = apply_kmeans(X_scaled, optimal_k)
    
    # Step 5: Visualize clusters
    visualize_clusters_pca(X_scaled, clusters, kmeans)
    
    # Step 6: Analyze cluster characteristics
    analyze_cluster_characteristics(df_cluster, clusters)
    
    # Step 7: Visualize cluster profiles
    visualize_cluster_profiles(df_cluster, clusters)
    
    print_section_header("CLUSTERING COMPLETE")
    print(f"✓ Optimal clusters found: {optimal_k}")
    print(f"✓ Silhouette Score: {silhouette_score(X_scaled, clusters):.4f}")
    print("✓ Ready for Business Insights")
    
    return {
        'kmeans': kmeans,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'X_scaled': X_scaled,
        'df_cluster': df_cluster
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_section_header("SECTION 7: CLUSTERING & CUSTOMER SEGMENTATION")
    
    # Load and preprocess dataset
    df = load_dataset('ecommerce_customer_data_custom_ratios.csv')
    
    if df is not None:
        preprocessing_result = preprocess_complete_pipeline(df)
        df_cleaned = preprocessing_result['dataframe']
        
        # Perform clustering
        clustering_result = perform_clustering(df_cleaned)
    else:
        print("Failed to load dataset.")