import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')  # 忽略统计模型警告

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_data():
    """获取并处理数据，返回小时级点赞量时间序列及每日峰值"""
    # 模拟数据库查询（实际使用时替换为真实SQL查询）
    # 此处用随机数据模拟，方便测试代码逻辑
    def mock_data():
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='H')
        n = len(dates)
        np.random.seed(42)
        like_cnt = np.random.poisson(lam=50, size=n) + np.sin(dates.hour/24*2*np.pi)*20  # 加入小时周期性
        like_cnt = np.maximum(0, like_cnt.astype(int))  # 确保非负
        return pd.DataFrame({
            'full_hour': dates,
            'like_cnt': like_cnt
        })
    
    # 获取数据（实际场景替换为SQL查询）
    df = mock_data()
    
    # 处理小时级时间序列（确保索引为时间，按小时排序）
    hourly = df.set_index('full_hour').sort_index()['like_cnt']
    
    # 计算每日峰值（按天分组取最大值）
    daily_peaks = hourly.resample('D').max().reset_index()
    daily_peaks.columns = ['day', 'daily_peak_like']  # 每日最大点赞量
    
    return {
        'hourly_ts': hourly,  # 小时级点赞量时间序列
        'daily_peaks': daily_peaks  # 每日峰值数据
    }


def train_model(data, future_days=7):
    """训练SARIMA模型，预测未来小时级点赞量及每日峰值"""
    hourly_ts = data['hourly_ts']
    
    # 训练小时级SARIMA模型（考虑日周期24小时和周周期7天）
    # 模型参数可根据实际数据调整（通过AIC准则或网格搜索优化）
    sarima_order = (1, 1, 1)  # (p,d,q)
    seasonal_order = (1, 1, 1, 24*7)  # (P,D,Q,s)，s=24*7表示周周期（小时级）
    
    model = SARIMAX(
        hourly_ts.values,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # 预测未来小时级数据（未来7天共7*24=168小时）
    future_hours = future_days * 24
    hourly_preds = model_fit.forecast(steps=future_hours)
    # 生成预测时间索引
    last_time = hourly_ts.index[-1]
    pred_index = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=future_hours,
        freq='H'
    )
    hourly_preds = pd.Series(hourly_preds, index=pred_index).clip(lower=0)  # 确保非负
    
    # 计算未来每日峰值（按天分组取最大值）
    daily_pred_peaks = hourly_preds.resample('D').max().reset_index()
    daily_pred_peaks.columns = ['day', 'pred_peak_like']
    
    return {
        'model': model_fit,
        'hourly_preds': hourly_preds,  # 未来小时级点赞量预测
        'daily_pred_peaks': daily_pred_peaks  # 未来每日峰值预测
    }


def plot_results(data, predictions):
    """可视化历史数据与预测结果"""
    hourly_ts = data['hourly_ts']
    daily_peaks = data['daily_peaks']
    hourly_preds = predictions['hourly_preds']
    daily_pred_peaks = predictions['daily_pred_peaks']
    
    # 绘制小时级趋势与预测
    plt.figure(figsize=(18, 10))
    
    # 历史数据（只展示最近30天，避免图表过密）
    recent_days = 30
    plot_start = hourly_ts.index[-recent_days*24]
    plt.plot(
        hourly_ts.loc[plot_start:], 
        label='历史小时级点赞量', 
        color='steelblue', 
        alpha=0.8
    )
    
    # 预测数据
    plt.plot(
        hourly_preds, 
        label='预测小时级点赞量', 
        color='crimson', 
        linestyle='--', 
        alpha=0.8
    )
    
    # 标记历史每日峰值
    plt.scatter(
        daily_peaks['day'], 
        daily_peaks['daily_peak_like'], 
        color='darkorange', 
        s=50, 
        label='历史每日峰值', 
        zorder=5
    )
    
    # 标记预测每日峰值
    plt.scatter(
        daily_pred_peaks['day'], 
        daily_pred_peaks['pred_peak_like'], 
        color='green', 
        s=50, 
        marker='*', 
        label='预测每日峰值', 
        zorder=5
    )
    
    # 图表配置
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('点赞量', fontsize=12)
    plt.title('点赞量趋势及每日峰值预测', fontsize=15)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 获取并处理数据
    data = get_data()
    # 训练模型并预测（默认预测未来7天）
    predictions = train_model(data, future_days=7)
    # 可视化结果
    plot_results(data, predictions)