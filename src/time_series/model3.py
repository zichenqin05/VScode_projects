import sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')  # Ignore statistical model warnings

# Set font for Chinese display (can be changed to a default font if not needed)
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

def get_data(train_cutoff='2022-05-07 00:00:00'):
    """Fetch and process data, split into train and test sets"""
    # Query data
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
    
    # Time conversion and processing
    df['time_ms'] = pd.to_numeric(df['time_ms'])
    df['readable_time'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['hour'] = df['readable_time'].dt.floor('H')  # Round to hour
    
    # Process like count
    df['like_cnt'] = pd.to_numeric(df['like_cnt'], errors='coerce').fillna(0)
    
    # Aggregate by hour and fill missing time points
    hourly = df.groupby('hour')['like_cnt'].sum().reset_index()
    hourly = hourly.set_index('hour').asfreq('H', fill_value=0)
    hourly = hourly.sort_index()
    hourly_ts = hourly['like_cnt']
    
    # Split train and test sets
    train_cutoff = pd.to_datetime(train_cutoff)
    # Calculate the time point 2 weeks before (2 weeks = 2*7*24 = 336 hours)
    train_start = train_cutoff - pd.Timedelta(weeks=2)
    # Train set: only data in [train_start, train_cutoff] (2 weeks)
    train_data = hourly_ts[(hourly_ts.index >= train_start) & (hourly_ts.index <= train_cutoff)]

    # Test set: 24 hours after train_cutoff
    test_start = train_cutoff + pd.Timedelta(hours=1)
    test_end = test_start + pd.Timedelta(hours=23)
    test_data = hourly_ts[(hourly_ts.index >= test_start) & (hourly_ts.index <= test_end)]
    
    # Data integrity check
    if len(train_data) == 0:
        raise ValueError("Train set is empty, please check the time range.")
    if len(test_data) < 24:
        raise ValueError(f"Test set has less than 24 hours (actual: {len(test_data)} hours).")
    
    return {
        'train_data': train_data,
        'test_data': test_data
    }

def train_hybrid_model(data, extend_hours=24):
    """Hybrid model: SARIMA for trend+seasonality, Random Forest for residual correction"""
    train_data = data['train_data'].copy()
    test_data = data['test_data'].copy()
    total_pred_hours = len(test_data) + extend_hours  # Total prediction: 48 hours

    period = 24  # Daily period
    if len(train_data) < 2 * period:
        raise ValueError(f"Train set too short, at least {2*period} needed, got {len(train_data)}.")

    # Avoid zeros for multiplicative decomposition
    train_data_adj = train_data + 1

    # Try additive and multiplicative decomposition
    decomposition_add = seasonal_decompose(train_data_adj, model='additive', period=period, extrapolate_trend='freq')
    decomposition_mul = seasonal_decompose(train_data_adj, model='multiplicative', period=period, extrapolate_trend='freq')

    resid_add_var = decomposition_add.resid.dropna().var()
    resid_mul_var = decomposition_mul.resid.dropna().var()

    # Choose the decomposition with larger residual variance (to better reflect fluctuation)
    if resid_add_var > resid_mul_var:
        decomposition = decomposition_add
        decompose_model = 'additive'
        print("Additive model selected for time series decomposition")
    else:
        decomposition = decomposition_mul
        decompose_model = 'multiplicative'
        print("Multiplicative model selected for time series decomposition")

    # Align with original index
    trend = decomposition.trend.reindex(train_data.index)
    seasonal = decomposition.seasonal.reindex(train_data.index)
    residual = decomposition.resid.reindex(train_data.index)

    # Restore residual (subtract 1)
    if decompose_model == 'multiplicative':
        trend_seasonal = trend * seasonal
        residual = residual - 1
    else:
        trend_seasonal = trend + seasonal
        residual = residual

    # Check residual
    print("Nonzero trend samples:", (trend.fillna(0) != 0).sum())
    print("Nonzero seasonal samples:", (seasonal.fillna(0) != 0).sum())
    print("Nonzero residual samples:", (residual.fillna(0) != 0).sum())
    print("Residual preview:", residual.head(10).to_list())

    # Remove NaN
    valid_idx = residual.dropna().index
    residual = residual.loc[valid_idx]

    if len(residual) == 0:
        raise ValueError("All residuals are NaN after decomposition, cannot train Random Forest. Please check data or adjust period.")

    # Combine trend+seasonality
    if decompose_model == 'additive':
        trend_seasonal = trend + seasonal
    else:
        trend_seasonal = trend * seasonal
    
    # 2. Train SARIMA
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
    
    # 3. Train Random Forest (residual correction)
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
    
    # 4. Hybrid prediction (combine results according to decomposition type)
    all_pred_index = pd.date_range(
        start=test_data.index[0],
        periods=total_pred_hours,
        freq='H'
    )
    
    # SARIMA prediction for trend+seasonality
    sarima_pred = sarima_fit.forecast(steps=total_pred_hours)
    
    # Random Forest prediction for residual
    rf_features = pd.DataFrame({
        'hour': all_pred_index.hour,
        'is_weekend': all_pred_index.weekday.isin([5, 6]).astype(int),
        'day_of_week': all_pred_index.weekday
    })
    residual_pred = model_rf.predict(rf_features)
    
    # Combine predictions
    if decompose_model == 'additive':
        hybrid_pred = pd.Series(
            (sarima_pred + residual_pred).clip(min=0),
            index=all_pred_index
        )
    else:
        hybrid_pred = pd.Series(
            (sarima_pred * (1 + residual_pred)).clip(min=0),
            index=all_pred_index
        )
    
    # Split prediction results
    test_pred_series = hybrid_pred.iloc[:len(test_data)]
    extend_pred_series = hybrid_pred.iloc[len(test_data):]
    
    # Extract peaks
    pred_peak = {'time': test_pred_series.idxmax(), 'value': test_pred_series.max()}
    true_peak = {'time': test_data.idxmax(), 'value': test_data.max()}
    extend_peak = {'time': extend_pred_series.idxmax(), 'value': extend_pred_series.max()}
    
    return {
        'train_data': train_data,
        'decomposition': decomposition,
        'decompose_model': decompose_model,
        'test_pred_series': test_pred_series,
        'test_data': test_data,
        'extend_pred_series': extend_pred_series,
        'pred_peak': pred_peak,
        'true_peak': true_peak,
        'extend_peak': extend_peak
    }

def plot_results(predictions):
    """Visualize decomposition and prediction comparison"""
    train_data = predictions['train_data']
    decomposition = predictions['decomposition']
    decompose_model = predictions['decompose_model']
    test_pred_series = predictions['test_pred_series']
    test_data = predictions['test_data']
    extend_pred_series = predictions['extend_pred_series']
    pred_peak = predictions['pred_peak']
    true_peak = predictions['true_peak']
    extend_peak = predictions['extend_peak']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [1, 1.5]})
    
    # Subplot 1: Decomposition (last 7 days)
    plot_length = min(7 * 24, len(train_data))
    last_train_data = train_data.iloc[-plot_length:]
    trend = decomposition.trend.dropna().iloc[-plot_length:]
    seasonal = decomposition.seasonal.dropna().iloc[-plot_length:]
    residual = decomposition.resid.dropna().iloc[-plot_length:]
    
    if decompose_model == 'additive':
        trend_seasonal_plot = trend + seasonal
    else:
        trend_seasonal_plot = trend * seasonal
    
    ax1.plot(last_train_data.index, last_train_data.values, label='Original Data', color='black', alpha=0.5)
    ax1.plot(trend_seasonal_plot.index, trend_seasonal_plot.values, label='Trend+Seasonality', color='blue', linewidth=2)
    ax1.plot(residual.index, residual.values, label='Residual', color='orange', alpha=0.7)
    ax1.set_title(f'Time Series Decomposition ({decompose_model} model, last 7 days)', fontsize=14)
    ax1.set_ylabel('Likes')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(last_train_data.index.min(), last_train_data.index.max())
    
    # Subplot 2: Prediction comparison
    ax2.plot(train_data.iloc[-plot_length:], label='Train Data (last 7 days)', color='steelblue', alpha=0.7)
    ax2.plot(test_pred_series, label='Hybrid Model Prediction (24h)', color='crimson', marker='o', markersize=5, alpha=0.8)
    ax2.plot(test_data, label='Ground Truth (24h)', color='forestgreen', marker='s', markersize=5, alpha=0.8)
    ax2.plot(extend_pred_series, label='Extended Prediction (24h)', color='purple', linestyle='--', marker='x', alpha=0.8)
    
    # Mark peaks and split lines
    ax2.scatter(pred_peak['time'], pred_peak['value'], color='crimson', s=120, marker='*', label=f'Predicted Peak: {pred_peak["value"]:.0f}')
    ax2.scatter(true_peak['time'], true_peak['value'], color='forestgreen', s=120, marker='*', label=f'Ground Truth Peak: {true_peak["value"]:.0f}')
    ax2.scatter(extend_peak['time'], extend_peak['value'], color='purple', s=120, marker='*', label=f'Extended Predicted Peak: {extend_peak["value"]:.0f}')
    ax2.axvline(x=train_data.index[-1], color='gray', linestyle='--', label='Train End')
    ax2.axvline(x=test_pred_series.index[-1], color='orange', linestyle='--', label='Extended Prediction Start')
    
    ax2.set_title('Hybrid Model Prediction Comparison', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Likes per Hour')
    ax2.legend(loc='upper left')
    plt.xticks(rotation=45)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.set_xlim(train_data.iloc[-plot_length:].index.min(), extend_pred_series.index.max())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        data = get_data(train_cutoff='2022-05-07 00:00:00')
        predictions = train_hybrid_model(data)
        plot_results(predictions)
        
        print("=== Comparison ===")
        print(f"Predicted Peak: {predictions['pred_peak']['time'].strftime('%Y-%m-%d %H:%M')}, Likes: {predictions['pred_peak']['value']:.0f}")
        print(f"Ground Truth Peak: {predictions['true_peak']['time'].strftime('%Y-%m-%d %H:%M')}, Likes: {predictions['true_peak']['value']:.0f}")
        print("\n=== Extended Prediction ===")
        print(f"Extended Predicted Peak: {predictions['extend_peak']['time'].strftime('%Y-%m-%d %H:%M')}, Likes: {predictions['extend_peak']['value']:.0f}")
    except Exception as e:
        print(f"Error: {str(e)}")