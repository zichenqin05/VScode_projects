import pandas as pd

def join_query(tables, join_conditions, select_columns, limit=10):
    """
    动态生成多表关联查询的SQL，并执行返回DataFrame
    
    参数:
        tables: 列表，格式为 [(表名, 别名), ...] 
                例: [('log_standard', 'l'), ('user_features', 'u')]
        join_conditions: 列表，关联条件字符串 
                例: ["l.user_id = u.user_id", "l.video_id = v.video_id"]
        select_columns: 列表，查询列字符串，支持AS别名 
                例: ["l.user_id", "u.name AS username", "v.type AS video_type"]
        limit: 整数，限制返回行数，默认10
    
    返回:
        DataFrame: 查询结果
    """
    # 生成SELECT部分
    select_clause = ", ".join(select_columns)
    
    # 生成FROM和JOIN部分
    from_clause = f"{tables[0][0]} {tables[0][1]}"  # 第一个表
    for i in range(1, len(tables)):
        table_name, alias = tables[i]
        condition = join_conditions[i-1]
        from_clause += f"\nINNER JOIN {table_name} {alias} ON {condition}"
    
    # 生成完整SQL
    sql = f"""
    SELECT {select_clause}
    FROM {from_clause}
    LIMIT {limit};
    """
    
    # 执行查询（使用已有的conn连接）
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        # 获取列名（用于DataFrame表头）
        columns = [desc[0] for desc in cursor.description]
        
        # 获取结果并转为DataFrame
        result = cursor.fetchall()
        df = pd.DataFrame(result, columns=columns)
        
        cursor.close()
        return df
        
    except Exception as e:
        print(f"查询出错: {str(e)}")
        print("执行的SQL语句:")
        print(sql)
        return None


import sql
import pandas as pd

df = sql.do("""SELECT
            l.user_id,
            u.user_active_degree,
			v.video_type,
            v.video_id
		FROM
			log_standard l
		INNER JOIN user_features_pure u ON l.user_id = u.user_id
		INNER JOIN video_features_basic_pure v ON l.video_id = v.video_id
		LIMIT 10;""")

df = pd.DataFrame(df, columns=['user_id', 'user_active', 'video_type', 'video_id'])
print (df)