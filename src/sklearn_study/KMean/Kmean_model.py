from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
import load_data as ld

def create_user_features(df):
    """为每个用户创建聚合特征（用于聚类）"""
    # 按用户ID分组聚合
    user_features = df.groupby('user_id').agg(
        # 播放行为特征
        avg_play_time=('play_time_ms', 'mean'),  # 平均播放时长
        total_play_time=('play_time_ms', 'sum'),  # 总播放时长
        video_count=('video_id', 'nunique'),  # 观看视频数量
        
        # 用户属性特征
        is_live_streamer=('is_live_streamer', lambda x: x.iloc[0]),  # 取分组内第一个值
        is_video_author=('is_video_author', lambda x: x.iloc[0]),
        follow_user_num=('follow_user_num', lambda x: x.iloc[0]),
        fans_user_num=('fans_user_num', lambda x: x.iloc[0]),
        friends_user_num=('friends_user_num', lambda x: x.iloc[0]),
        user_active_code=('user_active_code', lambda x: x.iloc[0]),
        
        # 内容偏好特征
        avg_video_type=('video_type', 'mean'),  # 视频类型偏好（1的比例）
        dominant_music_type=('music_type', lambda x: x.mode()[0] if not x.mode().empty else 0),  # 最常看的音乐类型
        
        # 标签偏好特征（统计每个标签出现的频率）
        tag1_freq=('tag1', lambda x: (x != 0).mean()),
        tag2_freq=('tag2', lambda x: (x != 0).mean()),
        tag3_freq=('tag3', lambda x: (x != 0).mean())
    ).reset_index()
    
    return user_features

def cluster_users(user_features, n_clusters=5):
    """对用户进行聚类"""
    # 提取特征列（排除user_id）
    feature_cols = user_features.columns.drop('user_id')
    X = user_features[feature_cols].values
    
    # 特征标准化（聚类算法对尺度敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 应用K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    return user_features, scaler, kmeans

def get_recommended_tags(target_user_id, df, user_clusters, top_n=5):
    """为目标用户推荐标签"""
    # 1. 找到目标用户所在的簇
    target_cluster = user_clusters[user_clusters['user_id'] == target_user_id]['cluster'].iloc[0]
    
    # 2. 找到同簇的所有用户
    cluster_users = user_clusters[user_clusters['cluster'] == target_cluster]['user_id'].tolist()
    
    # 3. 收集这些用户观看过的所有标签
    cluster_data = df[df['user_id'].isin(cluster_users)]
    all_tags = []
    
    # 收集所有非0标签
    for _, row in cluster_data.iterrows():
        for tag_col in ['tag1', 'tag2', 'tag3']:
            tag = row[tag_col]
            if tag != 0 and tag != '0':  # 排除填充的0值
                all_tags.append(tag)
    
    # 4. 统计标签频率并排序
    tag_counts = defaultdict(int)
    for tag in all_tags:
        tag_counts[tag] += 1
    
    # 按频率排序，取前N个
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    recommended_tags = [tag for tag, _ in sorted_tags[:top_n]]
    
    return recommended_tags
