import sql
import pandas as pd

tables = [
        ('log_standard', 'l'),
        ('video_features_basic_pure', 'v'),
        ('user_features_pure', 'u')
    ]
    
join_conditions = [
        "l.video_id = v.video_id",
        "l.user_id = u.user_id"
    ]
    
select_columns = [
        "l.user_id",
        "l.video_id",
        "v.video_type",
        "v.author_id",
        "u.user_active_degree",
        "l.duration_ms",
        
    ]
    
# 执行查询
result_df = sql.join_query(tables, join_conditions, select_columns, limit=10)
print (result_df)