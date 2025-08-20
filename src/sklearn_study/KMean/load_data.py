import pandas as pd
import sql

def load_data():
    df = sql.do("""SELECT
            l.user_id,
            l.video_id,
            l.play_time_ms,
            b.video_type,
            b.music_type,
            u.user_active_degree,
            u.is_live_streamer,
            u.is_video_author,
            u.follow_user_num,
            u.fans_user_num,
            u.friend_user_num,
            b.tag
        FROM
            (SELECT * FROM log_standard LIMIT 5000) l
        INNER JOIN
            user_features_pure u
        ON
            l.user_id = u.user_id
        INNER JOIN
            video_features_basic_pure b
        ON
            l.video_id = b.video_id
        ORDER BY
            l.user_id;""")

    df = pd.DataFrame(df, columns=['user_id', 'video_id', 'play_time_ms', 'video_type', 'music_type', 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num', 'fans_user_num', 'friends_user_num','tag'])

    # 处理一下文字
    df['video_type'] = df['video_type'].replace('NORMAL', 1)
    df['video_type'] = df['video_type'].replace('AD', 0)
    df = df.drop(df[df['video_type'] == 'unknown'].index)

    # 分离tag
    df[['tag1', 'tag2', 'tag3']] = df['tag'].str.split(',', expand=True).fillna(0)
    df = df.drop(columns=['tag'])

    #对用户活跃度进行编码
    df['user_active_code'] = df['user_active_degree'].factorize()[0]

    df = df.apply(pd.to_numeric, errors='coerce')  # 转换为数值类型
    return df

# ['full_active' '2_14_day_new' 'high_active' 'middle_active' 'low_active']
# [0, 1, 2, 3, 4]
