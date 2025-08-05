import sql
import pandas as pd
import numpy as np

def data_luncher():
    df = sql.do("""SELECT
            v.video_id,
            v.video_duration,
            v.server_width,
            v.server_height,
            s.like_cnt,
            s.complete_play_cnt,
            s.share_cnt,
            s.comment_user_num,
            s.play_progress
        FROM
            video_features_basic_pure v
        INNER JOIN
            video_features_statistic_pure s
        ON
            v.video_id = s.video_id""")

    df = pd.DataFrame(df, columns=['video_id','video_duration','width','height','like','complete_play','share','commnet_user_number','play_progress'])

    def normalize_resolution(row):
        min_dim = min(row['width'], row['height'])
        max_dim = max(row['width'], row['height'])
        return f"{min_dim}×{max_dim}"

    df['resolution'] = df.apply(normalize_resolution, axis=1)

    # 转换为秒
    df['video_duration'] = pd.to_numeric(df['video_duration'], errors='coerce')  
    df['video_duration'] = df['video_duration'] / 100 

    df = df.drop(columns=['height','width'])
    return df

def main_Model():
    # 获取数据
    data = data_luncher()

    # 分数列
    data['score'] = 0

    # 2. 分辨率打分
    resolution_scores = {
    '1440.0×1080.0': 5,
    '1280.0×720.0': 4,
    '1282.0×720.0': 4,
    '960.0×720.0': 2,
    '720.0×720.0': 1
    }
    # 循环打分
    for resolution, add_score in resolution_scores.items():
        data.loc[data['resolution'] == resolution, 'score'] += add_score

    def give_score(data, bins, part):
        labels = [1, 2, 3, 4, 5]
        data[f'{part}'] = pd.to_numeric(data[f'{part}'], errors='coerce').fillna(0)
        data[f'{part}_score'] = pd.cut(data[f'{part}'], bins=bins, labels=labels, include_lowest=True).astype(int)

        data['score'] = data['score'] + data[f'{part}_score']
        data = data.drop(columns=[f'{part}_score'])
        return data

    # 3. 点赞数打分
    bins = [0, 100, 500, 1000, 3000, float('inf')]
    data = give_score(data, bins, 'like')

    # 4. 完播打分
    bins2 = [0, 500, 1000, 5000, 10000, float('inf')]
    data = give_score(data,bins2, 'complete_play')

    # 5. 分享打分
    bins3 = [0, 0.5, 3, 10, 100, float('inf')]
    data = give_score(data, bins3, 'share')

    # 6. 评论打分
    bins4 = [0, 0.5, 10, 50, 100, float('inf')]
    data = give_score(data, bins4, 'commnet_user_number')

    # 7. 播放率打分
    bins5 = [0, 0.10, 0.30, 0.50, 0.80, float('inf')]
    data = give_score(data, bins5, 'play_progress')

    return data

def grade():

    data = main_Model()

    # 定义等级区间和对应的标签
    bins = [-np.inf, 7, 14, 21, np.inf] 
    labels = [4, 3, 2, 1]  # 对应等级: 1级最好，4级最差
    
    # 划分等级
    data['quality_level'] = pd.cut(
        data['score'],
        bins=bins,
        labels=labels,
        include_lowest=True  # 包含左边界
    ).astype(int)
    
    data = data.drop(columns=['score'])

    return data

def video_quality():
    result = grade()
    video_quality = result[['video_id', 'quality_level']]
    return video_quality

if __name__ == "__main__":

    result = grade()
    print(result.head(20))
