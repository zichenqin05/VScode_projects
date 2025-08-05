import sql
import pandas as pd
import propotion as pp

df = sql.do("""SELECT
            v.video_id,
            v.video_duration,
            v.music_type,
            v.server_width,
            v.server_height,
            s.like_cnt,
            s.play_cnt,
            s.share_cnt
        FROM
            video_features_basic_pure v
        INNER JOIN
            video_features_statistic_pure s
        ON
            v.video_id = s.video_id""")

df = pd.DataFrame(df, columns=['video_id','video_duration','music_type','width','height','like','play','share'])

def normalize_resolution(row):
        min_dim = min(row['width'], row['height'])
        max_dim = max(row['width'], row['height'])
        return f"{min_dim}×{max_dim}"

df['resolution'] = df.apply(normalize_resolution, axis=1)

# 转换为秒
df['video_duration'] = pd.to_numeric(df['video_duration'], errors='coerce')  
df['video_duration'] = df['video_duration'] / 100 

df = df.drop(columns=['height','width'])

print(df.head(20))


