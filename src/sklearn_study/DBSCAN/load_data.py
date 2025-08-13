import pandas as pd
import sql

def load_data_DBSCAN():
    df = sql.do("""SELECT
            user_id,
            user_active_degree,
            is_live_streamer,
            is_video_author,
            follow_user_num,
            fans_user_num,
            friend_user_num,
            register_days
            FROM
            user_features_pure
        """)
    df = pd.DataFrame(df, columns=['user_id', 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df