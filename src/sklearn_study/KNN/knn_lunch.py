import pandas as pd
import joblib  # 用于加载模型

def load_model_and_scaler():
    """加载保存的模型、标准化器和特征列名"""
    # 加载模型和标准化器
    knn = joblib.load(r'src\sklearn_study\KNN\models\knn_model.pkl')
    scaler = joblib.load(r'src\sklearn_study\KNN\models\scaler.pkl')
    # 加载特征列名
    with open(r'src\sklearn_study\KNN\models\feature_columns.txt', 'r') as f:
        feature_columns = f.read().split(',')
    return knn, scaler, feature_columns

def predict_new_video(new_video_data):
    """预测新视频的质量等级"""
    # 加载模型组件
    knn, scaler, feature_columns = load_model_and_scaler()
    
    # 确保新视频数据的特征列与训练数据一致
    new_video = pd.DataFrame(new_video_data).reindex(columns=feature_columns, fill_value=0)
    
    # 标准化新数据
    new_video_scaled = scaler.transform(new_video)
    
    # 预测
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


