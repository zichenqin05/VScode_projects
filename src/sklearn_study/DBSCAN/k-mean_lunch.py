import load_data as ld
import Kmean_model as km

# 1. 加载并预处理数据
print("加载数据中...")
data = ld.load_data()
    
# 2. 创建用户特征
print("创建用户特征...")
user_features = km.create_user_features(data)
    
# 3. 聚类用户
print("聚类用户中...")
user_clusters, scaler, kmeans = km.cluster_users(user_features, n_clusters=5)
    
# 4. 为示例用户推荐标签
target_user = 0  # 替换为实际要推荐的用户ID
if target_user in user_clusters['user_id'].values:
    recommended = km.get_recommended_tags(target_user, data, user_clusters, top_n=5)
    print(f"\n为用户 {target_user} 推荐的标签: {recommended}")
else:
    print(f"\n用户 {target_user} 不在数据集中")