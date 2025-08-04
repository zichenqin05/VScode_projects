import sql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def time_interval(df):
    """划分时长区间并返回统计结果"""
    if 'duration_seconds' not in df.columns:
        raise ValueError("DataFrame必须包含'duration_seconds'列")
    
    # 定义区间
    intervals = []
    intervals.append((0, 0.01, '0秒（照片）'))  # 特殊区间
    
    # 短时长（0.01-600秒，每30秒一个区间）
    i = 0.01
    while i < 10 * 60:
        next_i = i + 30
        intervals.append((i, next_i, f'{i:.2f}-{next_i:.0f}秒'))
        i = next_i
    
    # 中等时长（600-3600秒，每5分钟一个区间）
    for i in range(10*60, 60*60, 5*60):
        intervals.append((i, i+5*60, f'{i}-{i+5*60}秒'))
    
    # 长时长（3600秒以上）
    intervals.append((3600, 18000, '3600-18000秒'))
    intervals.append((18000, float('inf'), '18000秒以上'))

    # 统计区间数量
    duration_stats = {interval[2]: 0 for interval in intervals}
    for duration in df['duration_seconds']:
        for low, high, label in intervals:
            if low <= duration < high:
                duration_stats[label] += 1
                break
    
    # 转换为DataFrame并计算占比
    result_df = pd.DataFrame(list(duration_stats.items()), columns=['时长区间', '数量'])
    total = result_df['数量'].sum()
    result_df['占比(%)'] = (result_df['数量'] / total * 100).round(2)
    
    # 排序
    result_df['区间排序'] = result_df['时长区间'].apply(
        lambda x: [i for i, interval in enumerate(intervals) if interval[2] == x][0]
    )

    result_df = result_df.iloc[:23] 

    return result_df.sort_values('区间排序').drop('区间排序', axis=1).reset_index(drop=True)

def calculate_stats(df):
    """计算关键统计量"""
    positive_durations = df[df['duration_seconds'] > 0]['duration_seconds']
    stats = {
        '总样本量': len(df),
        '有效样本量': len(positive_durations),
        '均值': positive_durations.mean() if not positive_durations.empty else None,
        '中位数': positive_durations.median() if not positive_durations.empty else None,
        '大于0的最小值': positive_durations.min() if not positive_durations.empty else None,
        '最大值': df['duration_seconds'].max()
    }
    return stats

def plot_duration_distribution(result_df, stats):
    """绘制区间分布柱状图并在右上角添加统计量"""
    plt.figure(figsize=(16, 8))
    
    # 绘制柱状图
    bars = plt.bar(result_df['时长区间'], result_df['数量'], color='#4A90E2', alpha=0.8)
    
    # 添加数量标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 只显示有数据的标签
            plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'{height}', ha='center', va='bottom', fontsize=9)
    
    # 设置图表标题和坐标轴
    plt.title('视频时长区间分布', fontsize=16, pad=20)
    plt.xlabel('时长区间', fontsize=12, labelpad=10)
    plt.ylabel('视频数量', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统计量文本框
    stats_text = "统计信息：\n"
    stats_text += f"总样本量：{stats['总样本量']}\n"
    stats_text += f"有效样本量：{stats['有效样本量']}\n"
    stats_text += f"均值：{stats['均值']:.1f}秒\n" if stats['均值'] else "均值：无数据\n"
    stats_text += f"中位数：{stats['中位数']:.1f}秒\n" if stats['中位数'] else "中位数：无数据\n"
    stats_text += f"最小正值：{stats['大于0的最小值']:.1f}秒\n" if stats['大于0的最小值'] else "最小正值：无数据\n"
    stats_text += f"最大值：{stats['最大值']:.1f}秒" if stats['最大值'] else "最大值：无数据"
    
    # 调整文本框位置到右上角:
    plt.gcf().text(0.87, 0.83, stats_text, fontsize=10, 
                  bbox=dict(facecolor='white', edgecolor='gray', pad=10),
                  verticalalignment='top',  
                  horizontalalignment='left')  
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, top=0.85)  # 预留右上角空间
    plt.show()

if __name__ == "__main__":
    # 1. 获取数据
    df = sql.do("""
        SELECT video_duration 
        FROM video_features_basic_pure
        WHERE video_duration IS NOT NULL;
    """)
    df = pd.DataFrame(df, columns=['video_duration'])
    df['video_duration'] = pd.to_numeric(df['video_duration'])
    df['duration_seconds'] = df['video_duration'] / 1000  # 毫秒转秒
    
    # 2. 计算区间分布和统计量
    interval_df = time_interval(df)
    stats = calculate_stats(df)
    
    # 3. 绘图
    plot_duration_distribution(interval_df, stats)