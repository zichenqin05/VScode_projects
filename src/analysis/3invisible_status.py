import sql
import pandas as pd

df = sql.search('visible_status', 'video_features_basic_pure')

print(df)