import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mainmodel as mm

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
    
    # 标记周末
    for i, date in enumerate(all_dates):
        is_weekend = False
        if i < n_history:
            # 历史数据的周末标记
            is_weekend = peaks.iloc[i]['is_weekend'] == 1
        else:
            # 未来数据的周末标记
            is_weekend = predictions['future_weekends'][i - n_history] == 1
            
        if is_weekend:
            # 在周末日期下方添加灰色横线标记
            ax1.axvline(x=i, color='#f0f0f0', linewidth=10, alpha=0.5, zorder=0)
    
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
    # 添加周末标记说明
    lines1.append(plt.Line2D([0], [0], color='#f0f0f0', lw=4, label='周末'))
    labels1.append('周末')
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
    data = mm.get_data()
    
    # 训练模型并获取预测（可指定预测未来天数，默认5天）
    predictions = mm.train_models(data)
    
    # 绘制时间与互动值组合图表
    plot_time_and_interactive(data, predictions)
    
    # 输出预测结果
    print("历史每日峰值信息：")
    print(data['peaks'][['day', 'hour_of_day', 'is_weekend', 'like_cnt', 'comment_cnt', 'share_cnt']]
          .rename(columns={'hour_of_day': '峰值出现小时', 'is_weekend': '是否为周末',
                          'like_cnt': '峰值点赞量', 'comment_cnt': '峰值评论量', 
                          'share_cnt': '峰值分享量'}))
    
    print(f"\n四分位数范围 (用于训练的点赞量范围): {data['q1']:.2f} - {data['q3']:.2f}")
    outliers = data['peaks'][(data['peaks']['like_cnt'] < data['lower_bound']) | 
                            (data['peaks']['like_cnt'] > data['upper_bound'])]
    print(f"异常值数量: {len(outliers)}")
    
    print("\n未来预测：")
    for i in range(len(predictions['future_dates'])):
        weekend_flag = "是" if predictions['future_weekends'][i] == 1 else "否"
        print(f"日期: {predictions['future_dates'][i]}, "
              f"是否周末: {weekend_flag}, "
              f"预测峰值小时: {predictions['future_hour_preds'][i]}:00, "
              f"预测点赞量: {int(predictions['future_like_preds'][i])}")