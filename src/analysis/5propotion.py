import sql
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. 从数据库获取分辨率数据
    df = sql.do("""
    SELECT 
        server_width,
        server_height,
        COUNT(*) AS count
    FROM 
        video_features_basic_pure
    WHERE 
        server_width IS NOT NULL 
        AND server_height IS NOT NULL
    GROUP BY 
        server_width, server_height;
    """)
    df = pd.DataFrame(df, columns=['server_width', 'server_height', 'count'])

    # 2. 将宽和高转换为数值类型
    df['server_width'] = pd.to_numeric(df['server_width'])
    df['server_height'] = pd.to_numeric(df['server_height'])

    # 3. 核心步骤：按大小排序宽高后组合，确保相同像素的分辨率得到相同标识
    def normalize_resolution(row):
    # 取宽和高中的较小值作为第一个值，较大值作为第二个值
        min_dim = min(row['server_width'], row['server_height'])
        max_dim = max(row['server_width'], row['server_height'])
        return f"{min_dim}×{max_dim}"

    # 新增标准化分辨率列
    df['normalized_resolution'] = df.apply(normalize_resolution, axis=1)

    # 4. 按标准化分辨率合并并求和
    merged_df = df.groupby('normalized_resolution', as_index=False).agg({
        'count': 'sum'  # 合并数量
    })

    # 5. 计算占比
    total = merged_df['count'].sum()
    merged_df['占比(%)'] = (merged_df['count'] / total * 100).round(2)

    # 6. 按数量降序排序
    merged_df = merged_df.sort_values('count', ascending=False).reset_index(drop=True)

    # 7. 显示结果
    print(merged_df.head())
    
    return merged_df

if __name__=="__main__":

    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    #输出数据（保留前5）
    df2 = main()
    df2 = df2.iloc[:5]

    plt.figure(figsize=(12, 6))  # 设置画布大小
    bar_width = 0.6

    bars = plt.bar(df2['normalized_resolution'], df2['count'], width=bar_width, color='#4A90E2', label='数量')
    ax2 = plt.twinx()
    ax2.plot(df2['normalized_resolution'], df2['占比(%)'], color='#FF7A00', marker='o', linestyle='-', linewidth=2, label='占比(%)')

    # 设置图表标题和坐标轴标签
    plt.title('视频分辨率分布统计', fontsize=16, pad=20)
    plt.xlabel('分辨率', fontsize=12, labelpad=10)
    plt.ylabel('数量', fontsize=12, labelpad=10, color='#4A90E2')
    ax2.set_ylabel('占比(%)', fontsize=12, labelpad=10, color='#FF7A00')

    # 设置网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='upper left')

    # 调整x轴标签角度
    plt.xticks(rotation=45, ha='right')

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()