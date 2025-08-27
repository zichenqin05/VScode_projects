import sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')  # 忽略统计模型警告

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_data(train_cutoff='2022-05-07 00:00:00'):
    """获取并处理数据，分割训练集和测试集"""
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
    
    # 时间转换与处理
    df['time_ms'] = pd.to_numeric(df['time_ms'])
    df['readable_time'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['hour'] = df['readable_time'].dt.floor('H')  # 按小时取整
    
    # 处理点赞量
    df['like_cnt'] = pd.to_numeric(df['like_cnt'], errors='coerce').fillna(0)
    
    # 按小时聚合并补全缺失时间点
    hourly = df.groupby('hour')['like_cnt'].sum().reset_index()
    hourly = hourly.set_index('hour').asfreq('H', fill_value=0)
    hourly = hourly.sort_index()
    hourly_ts = hourly['like_cnt']
    
    # 分割训练集和测试集
    train_cutoff = pd.to_datetime(train_cutoff)
    # 计算2周前的时间点（2周 = 2*7*24 = 336小时）
    train_start = train_cutoff - pd.Timedelta(weeks=2)
    # 训练集：仅包含 [train_start, train_cutoff] 区间的2周数据
    train_data = hourly_ts[(hourly_ts.index >= train_start) & (hourly_ts.index <= train_cutoff)]

    # 测试集：train_cutoff之后24小时
    test_start = train_cutoff + pd.Timedelta(hours=1)
    test_end = test_start + pd.Timedelta(hours=23)
    test_data = hourly_ts[(hourly_ts.index >= test_start) & (hourly_ts.index <= test_end)]
    
    # 数据完整性检查
    if len(train_data) == 0:
        raise ValueError("训练集数据为空，请检查时间范围")
    if len(test_data) < 24:
        raise ValueError(f"测试集数据不足24小时（实际有{len(test_data)}小时）")
    
    return {
        'train_data': train_data,
        'test_data': test_data
    }
                                                                                                                                                                                                                                                                                  

def train_hybrid_model(data, extend_hours=24):
    """混合模型：SARIMA学习周期+趋势 + 随机森林修正残差"""
    train_data = data['train_data'].copy()
    test_data = data['test_data'].copy()
    total_pred_hours = len(test_data) + extend_hours  # 总预测48小时

    period = 24  # 日周期
    if len(train_data) < 2 * period:
        raise ValueError(f"训练数据长度太短，至少需要 {2*period}，实际为 {len(train_data)}")

    # 避免0值导致乘法分解出错
    train_data_adj = train_data + 1

    # 尝试加法和乘法分解
    decomposition_add = seasonal_decompose(train_data_adj, model='additive', period=period, extrapolate_trend='freq')
    decomposition_mul = seasonal_decompose(train_data_adj, model='multiplicative', period=period, extrapolate_trend='freq')

    resid_add_var = decomposition_add.resid.dropna().var()
    resid_mul_var = decomposition_mul.resid.dropna().var()

    # 选择残差方差较大的分解方式（更能反映波动）
    if resid_add_var > resid_mul_var:
        decomposition = decomposition_add
        decompose_model = 'additive'
        print("选择加法模型进行时间序列分解")
    else:
        decomposition = decomposition_mul
        decompose_model = 'multiplicative'
        print("选择乘法模型进行时间序列分解")

    # 用原始index对齐
    trend = decomposition.trend.reindex(train_data.index)
    seasonal = decomposition.seasonal.reindex(train_data.index)
    residual = decomposition.resid.reindex(train_data.index)

    # 还原残差（减去1）
    if decompose_model == 'multiplicative':
        trend_seasonal = trend * seasonal
        residual = residual - 1
    else:
        trend_seasonal = trend + seasonal
        residual = residual

    # 检查残差
    print("trend 非零样本数：", (trend.fillna(0) != 0).sum())
    print("seasonal 非零样本数：", (seasonal.fillna(0) != 0).sum())
    print("residual 非零样本数：", (residual.fillna(0) != 0).sum())
    print("residual 样本预览：", residual.head(10).to_list())

    # 用dropna()去除NaN
    valid_idx = residual.dropna().index
    residual = residual.loc[valid_idx]

    if len(residual) == 0:
        raise ValueError("分解后残差全为NaN，无法训练随机森林。请检查数据或调整分解周期。")

    # 组合“趋势+周期”成分（根据分解类型）
    if decompose_model == 'additive':
        trend_seasonal = trend + seasonal
    else:
        trend_seasonal = trend * seasonal  # 乘法模型下的组合方式
    
    # ----------------------
    # 2. 训练SARIMA模型
    # ----------------------
    sarima_order = (1, 1, 1)
    seasonal_order = (2, 1, 1, period)
    model_sarima = SARIMAX(
        trend_seasonal.values,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fit = model_sarima.fit(disp=False)
    
    # ----------------------
    # 3. 训练随机森林（残差修正）
    # ----------------------
    residual_features = pd.DataFrame({
        'hour': residual.index.hour,
        'is_weekend': residual.index.weekday.isin([5, 6]).astype(int),
        'day_of_week': residual.index.weekday
    }, index=residual.index)
    
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        random_state=42
    )
    model_rf.fit(residual_features, residual.values)
    
    # ----------------------
    # 4. 混合预测（根据分解类型组合结果）
    # ----------------------
    all_pred_index = pd.date_range(
        start=test_data.index[0],
        periods=total_pred_hours,
        freq='H'
    )
    
    # SARIMA预测趋势+周期
    sarima_pred = sarima_fit.forecast(steps=total_pred_hours)
    
    # 随机森林预测残差
    rf_features = pd.DataFrame({
        'hour': all_pred_index.hour,
        'is_weekend': all_pred_index.weekday.isin([5, 6]).astype(int),
        'day_of_week': all_pred_index.weekday
    })
    residual_pred = model_rf.predict(rf_features)
    
    # 组合预测结果（根据分解类型处理，关键修复）
    if decompose_model == 'additive':
        hybrid_pred = pd.Series(
            (sarima_pred + residual_pred).clip(min=0),
            index=all_pred_index
        )
    else:
        # 乘法模型下残差修正（避免原始值为0导致的问题）
        hybrid_pred = pd.Series(
            (sarima_pred * (1 + residual_pred)).clip(min=0),
            index=all_pred_index
        )
    
    # 分割预测结果
    test_pred_series = hybrid_pred.iloc[:len(test_data)]
    extend_pred_series = hybrid_pred.iloc[len(test_data):]
    
    # 提取峰值
    pred_peak = {'time': test_pred_series.idxmax(), 'value': test_pred_series.max()}
    true_peak = {'time': test_data.idxmax(), 'value': test_data.max()}
    extend_peak = {'time': extend_pred_series.idxmax(), 'value': extend_pred_series.max()}
    
    return {
        'train_data': train_data,
        'decomposition': decomposition,
        'decompose_model': decompose_model,  # 返回分解类型用于绘图
        'test_pred_series': test_pred_series,
        'test_data': test_data,
        'extend_pred_series': extend_pred_series,
        'pred_peak': pred_peak,
        'true_peak': true_peak,
        'extend_peak': extend_peak
    }


def plot_results(predictions):
    """可视化分解结果和预测对比"""
    train_data = predictions['train_data']
    decomposition = predictions['decomposition']
    decompose_model = predictions['decompose_model']  # 使用手动记录的分解类型
    test_pred_series = predictions['test_pred_series']
    test_data = predictions['test_data']
    extend_pred_series = predictions['extend_pred_series']
    pred_peak = predictions['pred_peak']
    true_peak = predictions['true_peak']
    extend_peak = predictions['extend_peak']
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [1, 1.5]})
    
    # ----------------------
    # 子图1：时间序列分解（最后7天）
    # ----------------------
    plot_length = min(7 * 24, len(train_data))
    last_train_data = train_data.iloc[-plot_length:]
    trend = decomposition.trend.dropna().iloc[-plot_length:]
    seasonal = decomposition.seasonal.dropna().iloc[-plot_length:]
    residual = decomposition.resid.dropna().iloc[-plot_length:]
    
    # 根据分解类型计算趋势+周期（关键修复）
    if decompose_model == 'additive':
        trend_seasonal_plot = trend + seasonal
    else:
        trend_seasonal_plot = trend * seasonal
    
    # 绘制分解图
    ax1.plot(last_train_data.index, last_train_data.values, label='原始数据', color='black', alpha=0.5)
    ax1.plot(trend_seasonal_plot.index, trend_seasonal_plot.values, label='趋势+周期', color='blue', linewidth=2)
    ax1.plot(residual.index, residual.values, label='残差', color='orange', alpha=0.7)
    ax1.set_title(f'时间序列分解（{decompose_model}模型，最后7天）', fontsize=14)
    ax1.set_ylabel('点赞量')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(last_train_data.index.min(), last_train_data.index.max())
    
    # ----------------------
    # 子图2：预测对比
    # ----------------------
    ax2.plot(train_data.iloc[-plot_length:], label='训练数据（最后7天）', color='steelblue', alpha=0.7)
    ax2.plot(test_pred_series, label='混合模型预测（24h）', color='crimson', marker='o', markersize=5, alpha=0.8)
    ax2.plot(test_data, label='真实值（24h）', color='forestgreen', marker='s', markersize=5, alpha=0.8)
    ax2.plot(extend_pred_series, label='延伸预测（24h）', color='purple', linestyle='--', marker='x', alpha=0.8)
    
    # 标记峰值和分隔线
    ax2.scatter(pred_peak['time'], pred_peak['value'], color='crimson', s=120, marker='*', label=f'预测峰值：{pred_peak["value"]:.0f}')
    ax2.scatter(true_peak['time'], true_peak['value'], color='forestgreen', s=120, marker='*', label=f'真实峰值：{true_peak["value"]:.0f}')
    ax2.scatter(extend_peak['time'], extend_peak['value'], color='purple', s=120, marker='*', label=f'延伸预测峰值：{extend_peak["value"]:.0f}')
    ax2.axvline(x=train_data.index[-1], color='gray', linestyle='--', label='训练截止点')
    ax2.axvline(x=test_pred_series.index[-1], color='orange', linestyle='--', label='延伸预测起点')
    
    ax2.set_title('混合模型预测对比', fontsize=14)
    ax2.set_xlabel('时间')
    ax2.set_ylabel('每小时点赞量')
    ax2.legend(loc='upper left')
    plt.xticks(rotation=45)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.set_xlim(train_data.iloc[-plot_length:].index.min(), extend_pred_series.index.max())
    
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    try:
        data = get_data(train_cutoff='2022-05-07 00:00:00')
        predictions = train_hybrid_model(data)
        plot_results(predictions)
        
        print("=== 对比部分 ===")
        print(f"预测峰值：{predictions['pred_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['pred_peak']['value']:.0f}")
        print(f"真实峰值：{predictions['true_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['true_peak']['value']:.0f}")
        print("\n=== 延伸预测部分 ===")
        print(f"延伸预测峰值：{predictions['extend_peak']['time'].strftime('%Y-%m-%d %H:%M')}，点赞量：{predictions['extend_peak']['value']:.0f}")
    except Exception as e:
        print(f"运行出错：{str(e)}")