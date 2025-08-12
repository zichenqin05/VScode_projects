import sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 获取数据
df = sql.do("""SELECT
            register_days,
            register_days_range,
            is_video_author,
            user_active_degree,
            fans_user_num,
            friend_user_num
            FROM
            user_features_pure""")

df = pd.DataFrame(df, columns=['days', 'days_range', 'is_author', 'active_degree', 'fans', 'friend'])

# 处理days_range列，划分用户阶段
def get_days_category(range_str):
    if '+' in range_str:
        num = int(range_str.replace('+', ''))
        return '老用户' if num > 365 else '中期用户'
    elif '-' in range_str:
        start, end = map(int, range_str.split('-'))
        if end < 30:
            return '新用户'
        elif start >= 30 and end <= 365:
            return '中期用户'
        else:  # end > 365
            return '老用户'
    else:
        return '未知'

# 创建用户阶段列
df['user_tenure'] = df['days_range'].apply(get_days_category)

df['is_author'] = df['is_author'].astype(int)  # 确保is_author为整数类型
# 1. 不同阶段用户中视频作者的比例
plt.figure(figsize=(10, 6))
author_ratio = df.groupby('user_tenure')['is_author'].mean() * 100
sns.barplot(x=author_ratio.index, y=author_ratio.values, palette='Set2')
plt.title('不同注册阶段用户中视频作者的比例(%)')
plt.ylabel('视频作者比例(%)')
plt.xlabel('用户注册阶段')
# 添加数值标签
for i, v in enumerate(author_ratio.values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.show()

# 2. 不同阶段用户的活跃度分布
plt.figure(figsize=(12, 7))
active_dist = pd.crosstab(df['user_tenure'], df['active_degree'], normalize='index') * 100
active_dist.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 7))
plt.title('不同注册阶段用户的活跃度分布(%)')
plt.ylabel('比例(%)')
plt.xlabel('用户注册阶段')
plt.legend(title='活跃度', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. 不同阶段用户的粉丝数和朋友数比较
plt.figure(figsize=(15, 6))
# 粉丝数比较
plt.subplot(1, 2, 1)
sns.boxplot(x='user_tenure', y='fans', data=df, palette='Set2')
plt.title('不同注册阶段用户的粉丝数分布')
plt.ylabel('粉丝数')
plt.xlabel('用户注册阶段')
plt.yscale('log')  # 使用对数刻度，处理可能的极端值

# 朋友数比较
plt.subplot(1, 2, 2)
sns.boxplot(x='user_tenure', y='friend', data=df, palette='Set2')
plt.title('不同注册阶段用户的朋友数分布')
plt.ylabel('朋友数')
plt.xlabel('用户注册阶段')
plt.yscale('log')  # 使用对数刻度，处理可能的极端值

plt.tight_layout()
plt.show()

# 5. 各阶段用户的平均粉丝数和朋友数

df['fans'] = df['fans'].astype(int)
df['friend'] = df['friend'].astype(int)
plt.figure(figsize=(12, 7))
stats = df.groupby('user_tenure')[['fans', 'friend']].mean().reset_index()
stats_melt = pd.melt(stats, id_vars=['user_tenure'], var_name='社交指标', value_name='平均值')

sns.barplot(x='user_tenure', y='平均值', hue='社交指标', data=stats_melt, palette='Set3')
plt.title('不同注册阶段用户的平均粉丝数和朋友数')
plt.ylabel('平均值')
plt.xlabel('用户注册阶段')
plt.yscale('log')  # 使用对数刻度
plt.legend(title='社交指标')
plt.tight_layout()
plt.show()