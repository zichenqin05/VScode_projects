from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['SimHei']  # Keep Chinese font setting for display compatibility

# Keep your original feature creation function
def create_user_features(df):
    """Create aggregated features for each user (for clustering)"""
    user_features = df.groupby('user_id').agg(
        # Play behavior features
        avg_play_time=('play_time_ms', 'mean'),
        total_play_time=('play_time_ms', 'sum'),
        video_count=('video_id', 'nunique'),
        
        # User attribute features
        is_live_streamer=('is_live_streamer', lambda x: x.iloc[0]),
        is_video_author=('is_video_author', lambda x: x.iloc[0]),
        follow_user_num=('follow_user_num', lambda x: x.iloc[0]),
        fans_user_num=('fans_user_num', lambda x: x.iloc[0]),
        friends_user_num=('friends_user_num', lambda x: x.iloc[0]),
        user_active_code=('user_active_code', lambda x: x.iloc[0]),
        
        # Content preference features
        avg_video_type=('video_type', 'mean'),
        dominant_music_type=('music_type', lambda x: x.mode()[0] if not x.mode().empty else 0),
        
        # Tag preference features
        tag1_freq=('tag1', lambda x: (x != 0).mean()),
        tag2_freq=('tag2', lambda x: (x != 0).mean()),
        tag3_freq=('tag3', lambda x: (x != 0).mean())
    ).reset_index()
    
    return user_features

# Add elbow method function
def find_optimal_clusters(user_features, max_clusters=10):
    """
    Find optimal number of clusters using elbow method
    
    Parameters:
    user_features: DataFrame containing user features
    max_clusters: Maximum number of clusters to try
    
    Returns:
    Recommended optimal number of clusters and plots the elbow graph
    """
    # Extract feature columns (exclude user_id)
    feature_cols = user_features.columns.drop('user_id')
    X = user_features[feature_cols].values
    
    # Feature standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate WCSS (Within-Cluster Sum of Squares) for different cluster numbers
    wcss = []
    cluster_range = range(1, max_clusters + 1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)  # Record WCSS value
    
    # Plot elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters (n_clusters)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method to Determine Optimal Number of Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mark possible optimal cluster number (elbow point)
    # Calculate slope changes to find the point with maximum change
    slopes = np.diff(wcss) / np.diff(list(cluster_range))
    optimal_idx = np.argmin(slopes) + 1  # +1 because length decreases by 1 after diff
    plt.scatter(cluster_range[optimal_idx], wcss[optimal_idx], 
                color='red', s=100, label=f'Recommended Clusters: {cluster_range[optimal_idx]}')
    plt.legend()
    plt.show()
    
    return cluster_range[optimal_idx]

# Keep clustering function, but use recommended number of clusters
def cluster_users(user_features, n_clusters=None):
    """Cluster users with optimal n_clusters from elbow method"""
    # Extract feature columns
    feature_cols = user_features.columns.drop('user_id')
    X = user_features[feature_cols].values
    
    # Feature standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Automatically calculate if number of clusters is not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(user_features)
        print(f"Number of clusters recommended by elbow method: {n_clusters}")
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score (evaluate clustering performance)
    if n_clusters > 1:
        sil_score = silhouette_score(X_scaled, user_features['cluster'])
        print(f"Clustering Silhouette Score: {sil_score:.4f}")
    
    return user_features, scaler, kmeans

# Keep tag recommendation function
def get_recommended_tags(target_user_id, df, user_clusters, top_n=5):
    """Recommend tags for target user"""
    target_cluster = user_clusters[user_clusters['user_id'] == target_user_id]['cluster'].iloc[0]
    cluster_users = user_clusters[user_clusters['cluster'] == target_cluster]['user_id'].tolist()
    cluster_data = df[df['user_id'].isin(cluster_users)]
    
    all_tags = []
    for _, row in cluster_data.iterrows():
        for tag_col in ['tag1', 'tag2', 'tag3']:
            tag = row[tag_col]
            if tag != 0 and tag != '0':
                all_tags.append(tag)
    
    tag_counts = defaultdict(int)
    for tag in all_tags:
        tag_counts[tag] += 1
    
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in sorted_tags[:top_n]]