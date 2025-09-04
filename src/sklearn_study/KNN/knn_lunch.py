import pandas as pd
import joblib  # 用于加载模型

def load_model_and_scaler():
    """Load the saved model, standardizer, and feature column names"""
    # Load model and standardizer
    knn = joblib.load(r'src\sklearn_study\KNN\models\knn_model.pkl')
    scaler = joblib.load(r'src\sklearn_study\KNN\models\scaler.pkl')
    # Load feature column names
    with open(r'src\sklearn_study\KNN\models\feature_columns.txt', 'r') as f:
        feature_columns = f.read().split(',')
    return knn, scaler, feature_columns

def predict_new_video(new_video_data):
    """Predict the quality level of a new video"""
    # Load model components
    knn, scaler, feature_columns = load_model_and_scaler()
    
    # Ensure the feature columns of the new video data match those of the training data
    new_video = pd.DataFrame(new_video_data).reindex(columns=feature_columns, fill_value=0)
    
    # Standardize the new data
    new_video_scaled = scaler.transform(new_video)
    
    # Predict
    predicted_level = knn.predict(new_video_scaled)
    return predicted_level[0]

if __name__ == "__main__":
    # 示例：新视频数据（键为特征列名，值为对应数据）
    new_video3 = {
        'video_duration': [200],
        'like':[600],  
        'complete_play': [1000],
        'share': [60],
        'comment_user_number': [50],
        'width': [1280],    
        'height': [720],
    }
    
    new_video2 = {
        'video_duration': [100],
        'like':[3],  
        'complete_play': [10],
        'share': [3],
        'comment_user_number': [8],
        'width': [1280],    
        'height': [720],
    }

    new_video1 = {
        'video_duration': [0],
        'like':[13],  
        'complete_play': [5.6],
        'share': [0.5],
        'comment_user_number': [0.03],
        'width': [1440],    
        'height': [720],
    }

    # 预测并输出结果
    level = predict_new_video(new_video2)
    print(f"视频预测质量等级：{level}级")


