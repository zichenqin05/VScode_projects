import load_data as ld
import Kmean_model as km

def lunch_recommendation(target_user):
    # 1. Load and preprocess data
    print("="*40)
    print("[Step 1] Loading data...")
    data = ld.load_data()
    
    # 2. Create user features
    print("="*40)
    print("[Step 2] Creating user features...")
    user_features = km.create_user_features(data)
    
    # 3. Cluster users
    print("="*40)
    print("[Step 3] Clustering users...")
    user_clusters, scaler, kmeans = km.cluster_users(user_features)
    
    # 4. Recommend tags for the sample user
    print("="*40)
    print("[Step 4] Recommendation results")
    if target_user in user_clusters['user_id'].values:
        recommended = km.get_recommended_tags(target_user, data, user_clusters, top_n=5)
        print(f"\nRecommended tags for user {target_user}:")
        for i, tag in enumerate(recommended, 1):
            print(f"  {i}. {tag}")
        print("="*40)
    else:
        print(f"\nUser {target_user} is not in the dataset")
        print("="*40)

    return recommended

if __name__ == "__main__":
    target_user = 10030
    lunch_recommendation(target_user)