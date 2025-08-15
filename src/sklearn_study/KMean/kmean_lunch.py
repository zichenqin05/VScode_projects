from . import load_data as ld
from . import Kmean_model as km

def lunch_recommendation(target_user):
    # 1. 加载并预处理数据
    print("="*40)
    print("【步骤1】加载数据中...")
    data = ld.load_data()
    
    # 2. 创建用户特征
    print("="*40)
    print("【步骤2】创建用户特征...")
    user_features = km.create_user_features(data)
    
    # 3. 聚类用户
    print("="*40)
    print("【步骤3】聚类用户中...")
    user_clusters, scaler, kmeans = km.cluster_users(user_features, n_clusters=5)
    
    # 4. 为示例用户推荐标签
    print("="*40)
    print("【步骤4】推荐结果")
    if target_user in user_clusters['user_id'].values:
        recommended = km.get_recommended_tags(target_user, data, user_clusters, top_n=5)
        print(f"\n为用户 {target_user} 推荐的标签:")
        for i, tag in enumerate(recommended, 1):
            print(f"  {i}. {tag}")
        print("="*40)
    else:
        print(f"\n用户 {target_user} 不在数据集中")
        print("="*40)

    return recommended

if __name__ == "__main__":
    target_user = 1
    lunch_recommendation(target_user)