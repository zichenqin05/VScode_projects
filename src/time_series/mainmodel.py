import sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # 忽略统计模型的警告信息

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_data():
    """获取并处理数据，返回处理后的数据集，新增周末特征"""
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
        
        # 判断是否为周末
        day_date = pd.to_datetime(day)
        is_weekend = 1 if day_date.weekday() >= 5 else 0  # 5是周六，6是周日
        
        peak_list.append({
            'day': day,
            'hour': peak_row['hour'],
            'hour_of_day': peak_row['hour_of_day'],
            'like_cnt': peak_row['like_cnt'],
            'comment_cnt': peak_row['comment_cnt'],
            'share_cnt': peak_row['share_cnt'],
            'is_weekend': is_weekend,
            'day_of_week': day_date.weekday()  # 0-6，周一到周日
        })
    peaks = pd.DataFrame(peak_list)
    
    # 处理异常值
    q1 = peaks['like_cnt'].quantile(0.25)
    q3 = peaks['like_cnt'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    non_outliers = peaks[(peaks['like_cnt'] >= lower_bound) & (peaks['like_cnt'] <= upper_bound)].copy()
    
    # 为时间序列添加索引作为特征
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

def train_models(data, future_days=7):
    """训练预测模型（使用SARIMA）并返回预测结果，增加滞后特征"""
    peaks = data['peaks'].copy()
    non_outliers = data['non_outliers'].copy()
    
    # ----------------------
    # 1. 添加滞后特征
    # ----------------------
    def add_lag_features(df):
        df = df.copy()
        # 滞后1天的点赞量（前一天的峰值点赞量）
        df['lag1_like'] = df['like_cnt'].shift(1).fillna(df['like_cnt'].mean())
        # 滞后7天的点赞量（上周同一天的峰值点赞量，捕捉周周期）
        df['lag7_like'] = df['like_cnt'].shift(7).fillna(df['like_cnt'].mean())
        # 滞后1天的小时（前一天的峰值小时）
        df['lag1_hour'] = df['hour_of_day'].shift(1).fillna(df['hour_of_day'].mean())
        return df
    
    # 为训练数据添加滞后特征
    non_outliers = add_lag_features(non_outliers)
    # 为所有峰值数据添加滞后特征（用于历史预测）
    peaks = add_lag_features(peaks)
    
    # ----------------------
    # 2. 训练点赞量预测模型（SARIMA + 外部特征）
    # ----------------------
    # 准备时间序列和外部特征（滞后特征+周末特征）
    ts_like = non_outliers.set_index('day')['like_cnt'].sort_index()
    exog_features = non_outliers[['is_weekend', 'lag1_like', 'lag7_like']]  # 外部回归因子
    
    # SARIMA模型参数：(p,d,q)为非季节性参数，(P,D,Q,s)为季节性参数（s=7表示周周期）
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    
    # 拟合SARIMA模型（加入外部特征）
    model_like = SARIMAX(
        ts_like.values,
        exog=exog_features.values,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_like_fit = model_like.fit(disp=False)
    
    # 生成历史点赞量预测
    historical_like_pred = model_like_fit.predict(
        start=0, 
        end=len(ts_like)-1, 
        exog=exog_features.values
    )
    
    # ----------------------
    # 3. 训练小时预测模型（随机森林 + 滞后特征）
    # ----------------------
    # 特征包括：滞后特征、周末、星期几
    X_hour_train = non_outliers[['is_weekend', 'day_of_week', 'lag1_hour', 'lag1_like']]
    y_hour_train = non_outliers['hour_of_day']
    
    model_hour = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model_hour.fit(X_hour_train, y_hour_train)
    
    # 生成历史小时预测
    historical_hour_pred = model_hour.predict(
        peaks[['is_weekend', 'day_of_week', 'lag1_hour', 'lag1_like']]
    )
    
    # ----------------------
    # 4. 预测未来数据
    # ----------------------
    last_day_idx = peaks['day_idx'].max()
    last_date = pd.to_datetime(peaks['day'].iloc[-1])
    
    # 初始化未来特征存储
    future_dates = []
    future_weekends = []
    future_like_preds = []
    future_hour_preds = []
    
    # 初始化滞后特征（用最后已知的值）
    last_like = peaks['like_cnt'].iloc[-1]
    last_like7 = peaks['like_cnt'].iloc[-7] if len(peaks)>=7 else peaks['like_cnt'].mean()
    last_hour = peaks['hour_of_day'].iloc[-1]
    
    for i in range(future_days):
        # 计算未来日期
        future_date = last_date + pd.Timedelta(days=i+1)
        future_dates.append(future_date.strftime('%Y-%m-%d'))
        is_weekend = 1 if future_date.weekday() >=5 else 0
        future_weekends.append(is_weekend)
        
        # 构建未来外部特征（包含滞后特征）
        future_exog = np.array([[
            is_weekend,
            last_like,  # 滞后1天（用上一步预测的点赞量）
            last_like7  # 滞后7天（用7天前的实际值）
        ]])
        
        # 预测未来点赞量
        like_pred = model_like_fit.forecast(steps=1, exog=future_exog)[0]
        like_pred = max(0, like_pred)  # 确保非负
        future_like_preds.append(like_pred)
        
        # 构建小时预测特征（包含滞后特征）
        hour_features = pd.DataFrame([[
            is_weekend,
            future_date.weekday(),
            last_hour,  # 滞后1天的小时
            last_like   # 滞后1天的点赞量
        ]], columns=X_hour_train.columns)
        
        # 预测未来小时
        hour_pred = model_hour.predict(hour_features)[0]
        hour_pred_rounded = int(round(np.clip(hour_pred, 0, 23)))
        future_hour_preds.append(hour_pred_rounded)
        
        # 更新滞后特征（用于下一次预测）
        last_like7 = last_like if (i+1) %7 ==0 else last_like7  # 每7天更新一次lag7
        last_like = like_pred
        last_hour = hour_pred_rounded
    
    return {
        'model_like': model_like_fit,
        'model_hour': model_hour,
        'historical_like_pred': historical_like_pred,
        'historical_hour_pred': historical_hour_pred,
        'future_indices': np.arange(last_day_idx + 1, last_day_idx + 1 + future_days),
        'future_dates': future_dates,
        'future_weekends': future_weekends,
        'future_like_preds': np.array(future_like_preds),
        'future_hour_preds': future_hour_preds,
        'ts_like': ts_like
    }
    