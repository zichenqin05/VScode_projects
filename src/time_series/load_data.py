import sql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']

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
            'like_cnt': peak_row['like_cnt']
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

def plot_graph1(data, predictions):
    """绘制第一张图表：包含所有小时数据和峰值"""
    hourly = data['hourly']
    peaks = data['peaks']
    non_outliers = data['non_outliers']
    lower_bound = data['lower_bound']
    upper_bound = data['upper_bound']
    
    plt.figure(figsize=(14, 7))
    
    # 绘制所有小时数据（灰色背景）
    for day in hourly['day'].unique():
        day_data = hourly[hourly['day'] == day]
        plt.plot(day_data['hour'], day_data['like_cnt'], color='lightgray', alpha=0.6, linewidth=1, zorder=1)
    
    # 绘制每日峰值
    plt.plot(peaks['day'], peaks['like_cnt'], marker='o', color='blue', label='每日峰值点赞量', zorder=2)
    
    # 绘制回归线
    plt.plot(peaks['day'], predictions['historical_like_pred'], 
             linestyle='--', color='red', label='点赞量趋势预测', zorder=3)
    
    # 标记非异常值训练数据
    plt.scatter(non_outliers['day'], non_outliers['like_cnt'], 
                color='green', s=50, label='训练数据（非异常值）', zorder=4)
    
    # 标记异常值
    outliers = peaks[(peaks['like_cnt'] < lower_bound) | (peaks['like_cnt'] > upper_bound)]
    if not outliers.empty:
        plt.scatter(outliers['day'], outliers['like_cnt'], 
                   color='orange', s=50, label='异常值（未用于训练）', zorder=4)
    
    # 标记未来预测值
    plt.scatter(predictions['future_dates'], predictions['future_like_preds'], 
               color='magenta', marker='*', s=200, label='未来预测峰值', zorder=5)
    
    # 添加预测值文本标签
    for date, like, hour in zip(predictions['future_dates'], 
                               predictions['future_like_preds'],
                               predictions['future_hour_preds']):
        plt.text(date, like, f"{int(like)}\n{hour}:00", 
                color='magenta', fontsize=10, ha='center', va='bottom')
    
    # 设置x轴标签间隔
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    plt.xlabel('日期')
    plt.ylabel('点赞量')
    plt.title('每小时点赞量分布、每日峰值及未来趋势预测')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_graph2(data, predictions):
    """绘制第二张图表：展示峰值出现的小时分布及预测"""
    peaks = data['peaks']
    
    plt.figure(figsize=(14, 7))
    
    # 绘制历史峰值出现的小时
    plt.plot(peaks['day'], peaks['hour_of_day'], marker='s', color='purple', 
             label='历史峰值出现小时', zorder=2)
    
    # 绘制小时预测线
    plt.plot(peaks['day'], predictions['historical_hour_pred'], 
             linestyle='--', color='orange', label='小时趋势预测', zorder=3)
    
    # 标记未来小时预测
    plt.scatter(predictions['future_dates'], predictions['future_hour_preds'], 
               color='magenta', marker='*', s=200, label='未来预测峰值小时', zorder=5)
    
    # 添加小时预测文本标签
    for date, hour in zip(predictions['future_dates'], predictions['future_hour_preds']):
        plt.text(date, hour, f"{hour}:00", 
                color='magenta', fontsize=12, ha='center', va='bottom')
    
    # 设置y轴为0-23小时
    plt.ylim(0, 23)
    plt.yticks(range(0, 24, 2))  # 每2小时显示一个刻度
    
    # 设置x轴标签间隔
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    plt.xlabel('日期')
    plt.ylabel('小时 (0-23)')
    plt.title('每日峰值出现小时及未来预测')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 获取数据
    data = get_data()
    
    # 训练模型并获取预测（可指定预测未来天数，默认5天）
    predictions = train_models(data, future_days=5)
    
    # 选择要绘制的图表
    #plot_graph1(data, predictions)  # 绘制第一张图
    plot_graph2(data, predictions)  # 绘制第二张图（需要时取消注释）
    
    # 输出预测结果
    print("历史每日峰值信息：")
    print(data['peaks'][['day', 'hour_of_day', 'like_cnt']].rename(columns={'hour_of_day': '峰值出现小时'}))
    
    print(f"\n四分位数范围 (用于训练的点赞量范围): {data['q1']:.2f} - {data['q3']:.2f}")
    outliers = data['peaks'][(data['peaks']['like_cnt'] < data['lower_bound']) | 
                            (data['peaks']['like_cnt'] > data['upper_bound'])]
    print(f"异常值数量: {len(outliers)}")
    
    print("\n未来预测：")
    for i in range(len(predictions['future_dates'])):
        print(f"日期: {predictions['future_dates'][i]}, "
              f"预测峰值小时: {predictions['future_hour_preds'][i]}:00, "
              f"预测点赞量: {int(predictions['future_like_preds'][i])}")
    