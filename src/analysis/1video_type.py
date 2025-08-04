import sql
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取每种类型的数量
df = sql.search('video_type','video_features_basic_pure')

# 计算总数量
total = df['count'].sum()

# 计算占比（保留两位小数）
df['percent'] = (df['count'] / total * 100).round(2)

print(df)

plt.figure(figsize=(12, 6))  # 设置画布大小
bar_width = 0.6

bars = plt.bar(df['video_type'], df['count'], width=bar_width, color='#4A90E2', label='数量')

# 绘制右侧占比坐标轴
ax2 = plt.twinx()
ax2.plot(df['video_type'], df['percent'], color='#FF7A00', marker='o', linestyle='-', linewidth=2, label='占比(%)')

# 设置图表标题和坐标轴标签
plt.title('视频类型分布统计', fontsize=16, pad=20)
plt.xlabel('视频类型', fontsize=12, labelpad=10)
plt.ylabel('数量', fontsize=12, labelpad=10, color='#4A90E2')
ax2.set_ylabel('占比(%)', fontsize=12, labelpad=10, color='#FF7A00')

# 设置网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
plt.legend(loc='upper left')
ax2.legend(loc='upper right')

# 调整x轴标签角度
plt.xticks(rotation=45, ha='right')

# 调整布局并显示图表
plt.tight_layout()
plt.show()