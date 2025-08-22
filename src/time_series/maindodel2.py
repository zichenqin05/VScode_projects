import sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')  # 忽略统计模型警告

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_data(train_cutoff='2022-05-08 00:00:00'):
    """
    获取并处理数据，按指定时间分割训练集和测试集
    - 训练集：截止到train_cutoff（含）的所有小时级数据
    - 测试集：train_cutoff之后24小时的真实数据（用于与预测对比）
    """
    # 查询数据
    df = sql.do("""SELECT 
                l.user_id,
                l.video_id,
                l.date,
                l.hourmin,
                l.time_ms, 
                v.like_cnt
                FROM 
                log_standard l
                INNER JOIN
                video_features_statistic_pure v
                ON l.video_id = v.video_id""")
    
    df = pd.DataFrame(df, columns=['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'like_cnt'])
    
    # 时间转换：将毫秒级时间戳转为datetime，并提取小时级时间
    df['time_ms'] = pd.to_numeric(df['time_ms'])
    df['readable_time'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['hour'] = df['readable_time'].dt.floor('H')  # 按小时向下取整
    
    # 转换点赞量为数值型并处理缺失值
    df['like_cnt'] = pd.to_numeric(df['like_cnt'], errors='coerce').fillna(0)
    
    # 按小时分组统计总点赞量，确保时间连续
    hourly = df.groupby('hour')['like_cnt'].sum().reset_index()
    hourly = hourly.set_index('hour').asfreq('H', fill_value=0)  # 补全缺失小时
    hourly = hourly.sort_index()  # 按时间排序
    hourly_ts = hourly['like_cnt']
    
    # 分割训练集和测试集
    train_cutoff = pd.to_datetime(train_cutoff)
    # 训练集：截止到指定时间（含）的所有数据
    train_data = hourly_ts[hourly_ts.index <= train_cutoff]
    # 测试集：指定时间之后的24小时真实数据（用于对比）
    test_start = train_cutoff + pd.Timedelta(hours=1)
    test_end = test_start + pd.Timedelta(hours=23)
    test_data = hourly_ts[(hourly_ts.index >= test_start) & (hourly_ts.index <= test_end)]
    
    # 检查数据完整性
    if len(train_data) == 0:
        raise ValueError("训练集数据为空，请检查时间范围是否正确")
    if len(test_data) < 24:
        raise ValueError(f"测试集数据不足24小时（实际有{len(test_data)}小时），请检查数据完整性")
    
    return {
        'train_data': train_data,  # 训练数据（截止到指定时间）
        'test_data': test_data     # 测试数据（用于对比的24小时真实值）
    }


def train_model(data, extend_hours=24):
    """
    训练SARIMA模型，预测两部分数据：
    1. 与测试集对比的24小时预测
    2. 后续延伸的24小时预测
    """
    train_data = data['train_data']
    test_data = data['test_data']
    total_pred_hours = len(test_data) + extend_hours  # 总预测48小时（24+24）
    
    # 配置SARIMA模型（日周期24小时）
    sarima_order = (1, 1, 1)
    seasonal_order = (2, 1, 1, 24)  # 保持季节性参数
    
    model = SARIMAX(
        train_data.values,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # 预测总时长（48小时）
    pred_values = model_fit.forecast(steps=total_pred_hours)
    
    # 分割预测结果：前24小时与测试集对比，后24小时为延伸预测
    test_pred_values = pred_values[:len(test_data)]  # 对比用预测
    extend_pred_values = pred_values[len(test_data):]  # 延伸预测
    
    # 生成时间索引（确保连续）
    test_pred_index = test_data.index  # 与测试集时间一致
    extend_pred_start = test_pred_index[-1] + pd.Timedelta(hours=1)
    extend_pred_index = pd.date_range(
        start=extend_pred_start,
        periods=extend_hours,
        freq='H'
    )
    
    # 转为Series并确保非负
    test_pred_series = pd.Series(test_pred_values, index=test_pred_index).clip(lower=0)
    extend_pred_series = pd.Series(extend_pred_values, index=extend_pred_index).clip(lower=0)
    
    # 提取各部分峰值
    pred_peak = {'time': test_pred_series.idxmax(), 'value': test_pred_series.max()}  # 对比部分预测峰值
    true_peak = {'time': test_data.idxmax(), 'value': test_data.max()}  # 真实峰值
    extend_peak = {'time': extend_pred_series.idxmax(), 'value': extend_pred_series.max()}  # 延伸预测峰值
    
    return {
        'train_data': train_data,
        'test_pred_series': test_pred_series,  # 对比用预测（24h）
        'test_data': test_data,                # 真实值（24h）
        'extend_pred_series': extend_pred_series,  # 延伸预测（24h）
        'pred_peak': pred_peak,                # 对比部分预测峰值
        'true_peak': true_peak,                # 真实峰值
        'extend_peak': extend_peak             # 延伸预测峰值
    }


def plot_results(predictions):
    """可视化训练数据、对比预测、真实值及延伸预测"""
    train_data = predictions['train_data']
    test_pred_series = predictions['test_pred_series']
    test_data = predictions['test_data']
    extend_pred_series = predictions['extend_pred_series']
    pred_peak = predictions['pred_peak']
    true_peak = predictions['true_peak']
    extend_peak = predictions['extend_peak']
    
    plt.figure(figsize=(18, 10))
    
    # 1. 绘制训练数据（最后7天）
    if len(train_data) >= 7*24:
        plt.plot(
            train_data.iloc[-168:],  # 最后7天
            label='训练数据（最后7天）',
            color='steelblue',
            alpha=0.7
        )
    else:
        plt.plot(
            train_data,
            label='训练数据',
            color='steelblue',
            alpha=0.7
        )
    
    # 2. 绘制对比部分（预测vs真实）
    plt.plot(
        test_pred_series,
        label='对比用预测（24h）',
        color='crimson',
        linestyle='-',
        marker='o',
        markersize=5,
        alpha=0.8
    )
    plt.plot(
        test_data,
        label='真实值（24h）',
        color='forestgreen',
        linestyle='-',
        marker='s',
        markersize=5,
        alpha=0.8
    )
    
    # 3. 绘制延伸预测（24h）
    plt.plot(
        extend_pred_series,
        label='延伸预测（24h）',
        color='purple',
        linestyle='--',
        marker='x',
        markersize=5,
        alpha=0.8
    )
    
    # 4. 标记所有峰值
    plt.scatter(
        pred_peak['time'],
        pred_peak['value'],
        color='crimson',
        s=120,
        marker='*',
        label=f'对比预测峰值：{pred_peak["value"]:.0f}（{pred_peak["time"].strftime("%H:%M")}）'
    )
    plt.scatter(
        true_peak['time'],
        true_peak['value'],
        color='forestgreen',
        s=120,
        marker='*',
        label=f'真实峰值：{true_peak["value"]:.0f}（{true_peak["time"].strftime("%H:%M")}）'
    )
    plt.scatter(
        extend_peak['time'],
        extend_peak['value'],
        color='purple',
        s=120,
        marker='*',
        label=f'延伸预测峰值：{extend_peak["value"]:.0f}（{extend_peak["time"].strftime("%H:%M")}）'
    )
    
    # 5. 图表配置
    train_end = train_data.index[-1]
    plt.axvline(x=train_end, color='gray', linestyle='--', label='训练截止点')
    # 用竖线区分对比预测和延伸预测
    extend_start = test_pred_series.index[-1]
    plt.axvline(x=extend_start, color='orange', linestyle='--', label='延伸预测起点')
    
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('每小时点赞量', fontsize=12)
    plt.title('点赞量预测对比及延伸预测（共48小时）', fontsize=15)
    plt.legend(fontsize=10, loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    try:
        # 训练截止到2022-05-07 0:00，测试对比2022-05-07 1:00至2022-05-08 0:00，延伸预测至2022-05-09 0:00
        data = get_data(train_cutoff='2022-05-07 00:00:00')
        predictions = train_model(data)
        plot_results(predictions)
        
        # 打印峰值信息
        print("=== 对比部分 ===")
        print(f"预测峰值：{predictions['pred_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['pred_peak']['value']:.0f}")
        print(f"真实峰值：{predictions['true_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['true_peak']['value']:.0f}")
        print("\n=== 延伸预测部分 ===")
        print(f"延伸预测峰值：{predictions['extend_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['extend_peak']['value']:.0f}")
    except Exception as e:
        print(f"运行出错：{str(e)}")