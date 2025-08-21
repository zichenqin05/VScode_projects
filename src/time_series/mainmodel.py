import sql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_data():
    """获取并处理数据，返回处理后的数据集"""
    # 查询数据
    df = sql.do("""SELECT 
                l.user_id,
                l.video_id,
                l.date,
                l.hourmin,
                l.time_ms, 
                v.like_cnt,
                v.comment_cnt,
                v.share_cnt
                FROM 
                log_standard l
                INNER JOIN
                video_features_statistic_pure v
                ON l.video_id = v.video_id""")
    
    df = pd.DataFrame(df, columns=['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 
                                   'like_cnt', 'comment_cnt', 'share_cnt'])
    
    # 时间转换
    df['time_ms'] = pd.to_numeric(df['time_ms'])
    df['readable_time'] = pd.to_datetime(df['time_ms'], unit='ms')
    
    # 以小时为时间段统计互动量
    df['hour'] = df['readable_time'].dt.strftime('%Y-%m-%d %H:00')
    df['day'] = df['readable_time'].dt.strftime('%Y-%m-%d')
    df['hour_of_day'] = df['readable_time'].dt.hour  # 提取小时数（0-23）
    
    # 转换互动量为数值类型并处理缺失值
    df[['like_cnt', 'comment_cnt', 'share_cnt']] = df[['like_cnt', 'comment_cnt', 'share_cnt']].apply(
        pd.to_numeric, errors='coerce').fillna(0)
    
    # 按小时分组统计
    hourly = df.groupby(['day', 'hour', 'hour_of_day'])[['like_cnt', 'comment_cnt', 'share_cnt']].sum().reset_index()
    
    # 找出每天的点赞峰值及其出现时间和小时
    peak_list = []
    for day, group in hourly.groupby('day'):
        idx = group['like_cnt'].idxmax()
        peak_row = group.loc[idx]
        peak_list.append({
            'day': day,
            'hour': peak_row['hour'],
            'hour_of_day': peak_row['hour_of_day'],
            'like_cnt': peak_row['like_cnt'],
            'comment_cnt': peak_row['comment_cnt'],  # 新增评论量
            'share_cnt': peak_row['share_cnt']       # 新增分享量
        })
    peaks = pd.DataFrame(peak_list)
    
    # 处理异常值：只使用Q1到Q3范围内的数据进行训练
    q1 = peaks['like_cnt'].quantile(0.25)
    q3 = peaks['like_cnt'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    non_outliers = peaks[(peaks['like_cnt'] >= lower_bound) & (peaks['like_cnt'] <= upper_bound)].copy()
    
    # 为时间序列添加索引
    non_outliers['day_idx'] = np.arange(len(non_outliers))
    peaks['day_idx'] = np.arange(len(peaks))
    
    return {
        'hourly': hourly,
        'peaks': peaks,
        'non_outliers': non_outliers,
        'q1': q1,
        'q3': q3,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def train_models(data, future_days=5):
    """训练预测模型并返回预测结果"""
    peaks = data['peaks']
    non_outliers = data['non_outliers']
    
    # 1. 训练点赞量预测模型（线性回归）
    model_like = LinearRegression()
    model_like.fit(non_outliers[['day_idx']], non_outliers['like_cnt'])
    
    # 2. 训练小时预测模型（随机森林）
    model_hour = RandomForestRegressor(n_estimators=50, random_state=42)
    model_hour.fit(non_outliers[['day_idx']], non_outliers['hour_of_day'])
    
    # 生成历史预测值
    historical_like_pred = model_like.predict(peaks[['day_idx']])
    historical_hour_pred = model_hour.predict(peaks[['day_idx']])
    
    # 预测未来指定天数
    last_day_idx = peaks['day_idx'].max()
    future_indices = np.arange(last_day_idx + 1, last_day_idx + 1 + future_days).reshape(-1, 1)
    
    # 预测未来的点赞量和小时
    future_like_preds = model_like.predict(future_indices)
    future_hour_preds = model_hour.predict(future_indices)
    
    # 处理小时预测结果（确保在0-23范围内，并取整）
    future_hour_preds = [int(round(np.clip(hour, 0, 23))) for hour in future_hour_preds]
    
    # 生成未来日期字符串
    last_date = pd.to_datetime(peaks['day'].iloc[-1])
    future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') 
                   for i in range(future_days)]
    
    return {
        'model_like': model_like,
        'model_hour': model_hour,
        'historical_like_pred': historical_like_pred,
        'historical_hour_pred': historical_hour_pred,
        'future_indices': future_indices.flatten(),
        'future_dates': future_dates,
        'future_like_preds': future_like_preds,
        'future_hour_preds': future_hour_preds
    }

def plot_time_and_interactive(data, predictions):
    """绘制时间（折线）与互动值（柱状）的组合图表"""
    peaks = data['peaks']
    future_dates = predictions['future_dates']
    future_like_preds = predictions['future_like_preds']
    future_hour_preds = predictions['future_hour_preds']
    
    # 整合历史与未来数据
    history_dates = list(peaks['day'])  # 历史日期
    all_dates = history_dates + future_dates  # 所有日期（历史+未来）
    n_history = len(history_dates)
    n_future = len(future_dates)
    
    # 互动值数据（柱状图用）：历史峰值点赞量 + 未来预测点赞量
    history_likes = peaks['like_cnt']
    all_likes = list(history_likes) + list(future_like_preds)
    
    # 时间特征数据（折线图用）：历史峰值小时 + 未来预测小时
    history_hours = peaks['hour_of_day']
    all_hours = list(history_hours) + list(future_hour_preds)
    
    # 创建画布和双轴（左侧：互动值，右侧：时间）
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 左侧y轴：柱状图展示互动值（点赞量）
    bars = ax1.bar(
        all_dates, 
        all_likes, 
        color=['#1f77b4']*n_history + ['#ff7f0e']*n_future,  # 历史蓝色，未来橙色
        alpha=0.7, 
        width=0.6,
        label='每日峰值点赞量'
    )
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('峰值点赞量', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 右侧y轴：折线图展示时间特征（峰值出现小时）
    ax2 = ax1.twinx()  # 共享x轴的第二个y轴
    # 绘制历史小时折线
    ax2.plot(
        history_dates, 
        history_hours, 
        marker='o', 
        color='#2ca02c', 
        linewidth=2, 
        markersize=6,
        label='历史峰值小时'
    )
    # 绘制未来小时折线（单独设置颜色区分）
    if n_future > 0:
        ax2.plot(
            future_dates, 
            future_hour_preds, 
            marker='s', 
            color='#d62728', 
            linewidth=2, 
            markersize=6,
            label='预测峰值小时'
        )
    ax2.set_ylabel('峰值出现小时（0-23）', color='#2ca02c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 23)  # 小时范围固定为0-23
    ax2.set_yticks(range(0, 24, 3))  # 每3小时显示一个刻度
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 设置x轴格式
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=12))  # 控制x轴显示的日期数量
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # 标题
    plt.title('每日峰值点赞量（柱状）与峰值出现小时（折线）趋势及预测', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 获取数据
    data = get_data()
    
    # 训练模型并获取预测（可指定预测未来天数，默认5天）
    predictions = train_models(data, future_days=5)
    
    # 绘制时间与互动值组合图表
    plot_time_and_interactive(data, predictions)
    
    # 输出预测结果
    print("历史每日峰值信息：")
    print(data['peaks'][['day', 'hour_of_day', 'like_cnt', 'comment_cnt', 'share_cnt']]
          .rename(columns={'hour_of_day': '峰值出现小时', 'like_cnt': '峰值点赞量', 
                          'comment_cnt': '峰值评论量', 'share_cnt': '峰值分享量'}))
    
    print(f"\n四分位数范围 (用于训练的点赞量范围): {data['q1']:.2f} - {data['q3']:.2f}")
    outliers = data['peaks'][(data['peaks']['like_cnt'] < data['lower_bound']) | 
                            (data['peaks']['like_cnt'] > data['upper_bound'])]
    print(f"异常值数量: {len(outliers)}")
    
    print("\n未来预测：")
    for i in range(len(predictions['future_dates'])):
        print(f"日期: {predictions['future_dates'][i]}, "
              f"预测峰值小时: {predictions['future_hour_preds'][i]}:00, "
              f"预测点赞量: {int(predictions['future_like_preds'][i])}")