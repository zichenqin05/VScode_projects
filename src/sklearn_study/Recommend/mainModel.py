import sys
from pathlib import Path
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from sklearn_study.KMean import kmean_lunch as km
import sql
import pandas as pd
import joblib

def find_videos(tag):

    print("加载视频数据库中...")

    df = sql.do("""SELECT
            v.video_id,
            v.video_duration,
            v.server_width,
            v.server_height,
            v.tag,
            s.like_cnt,
            s.complete_play_cnt,
            s.share_cnt,
            s.comment_user_num
        FROM
            video_features_basic_pure v
        INNER JOIN
            video_features_statistic_pure s
        ON
            v.video_id = s.video_id""")

    df = pd.DataFrame(df, columns=['video_id','video_duration','width','height','tag','like','complete_play','share','commnet_user_number'])

    def normalize_resolution(row):
        min_dim = min(row['width'], row['height'])
        max_dim = max(row['width'], row['height'])
        return f"{min_dim}×{max_dim}"

    df['resolution'] = df.apply(normalize_resolution, axis=1)
    # 转换为秒
    df['video_duration'] = pd.to_numeric(df['video_duration'], errors='coerce')  
    df['video_duration'] = df['video_duration'] / 100 

    # 只对数值列做转换
    for col in ['like', 'complete_play', 'share', 'commnet_user_number', 'width', 'height']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.drop(columns=['height','width'])

    split_res = df['resolution'].str.split('×', expand=True)  # expand=True返回DataFrame

    # 转换为整数
    df['width'] = split_res[0].astype(float).astype(int)   # 宽度列
    df['height'] = split_res[1].astype(float).astype(int)  # 高度列
    df = df.drop(columns=['resolution'])  # 删除原分辨率列

    # 拆分tag列，并转为字符串
    df[['tag1', 'tag2', 'tag3']] = df['tag'].str.split(',', expand=True).fillna('0').astype(str)
    df = df.drop(columns=['tag'])

    # tag_list 是 [np.float64(39.0), ...]，先转成字符串
    tag = [str(int(t)) for t in tag]

    # 找出所有包含这些tag的行
    mask = (
        df['tag1'].isin(tag) |
        df['tag2'].isin(tag) |
        df['tag3'].isin(tag)
    )
    result = df[mask].copy()

    # 拆分，每个tag一行
    tag_cols = ['tag1', 'tag2', 'tag3']
    melted = result.melt(
        id_vars=[col for col in result.columns if col not in tag_cols],
        value_vars=tag_cols,
        var_name='tag_col',
        value_name='tag'
    )
    # 只保留在tag_list中的tag
    melted = melted[melted['tag'].isin(tag)]

    melted = melted.reset_index(drop=True)
    melted = melted.drop(columns=['tag_col'])
    return melted

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

def main(user_id):
    tag_list = km.lunch_recommendation(user_id)
    data = find_videos(tag_list)
    recommended = {}

    print("正在根据视频数据库和视频质量为用户推荐优质视频...")

    for tag in tag_list:
        tag_str = str(int(tag))
        tag_videos = data[data['tag'] == tag_str]
        found = False
        # 遍历该tag下所有视频
        for idx, row in tag_videos.iterrows():
            video_data = row.to_frame().T  # 转为DataFrame
            # 只保留模型需要的特征列
            video_features = video_data.drop(['video_id', 'tag'], axis=1, errors='ignore')
            level = predict_new_video(video_features)
            if level > 3:
                recommended[tag_str] = row['video_id']
                found = True
                break
        # 如果没有评分大于3的视频，可以选一个评分最高的
        if not found and not tag_videos.empty:
            max_score = -1
            best_video_id = None
            for idx, row in tag_videos.iterrows():
                video_data = row.to_frame().T
                video_features = video_data.drop(['video_id', 'tag'], axis=1, errors='ignore')
                level = predict_new_video(video_features)
                if level > max_score:
                    max_score = level
                    best_video_id = row['video_id']
            recommended[tag_str] = best_video_id

    print(f"推荐用户{user_id}的视频为：")
    for tag, vid in recommended.items():
        print(f"类别 {tag} 推荐视频ID: {vid}")
    return recommended


