import sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据获取与预处理
df = sql.do("""SELECT
            user_id,
            follow_user_num,
            fans_user_num,
            friend_user_num,
            user_active_degree,
            is_video_author,
            is_live_streamer
            FROM
            user_features_pure""")

df = pd.DataFrame(
    df, 
    columns=['user_id','follow', 'fans', 'friend', 'active_degree', 'is_author', 'is_live']
)

# 数据类型转换
df[['follow', 'fans', 'friend']] = df[['follow', 'fans', 'friend']].apply(
    pd.to_numeric, errors='coerce'
)
df['is_author'] = df['is_author'].astype('category')
df['is_live'] = df['is_live'].astype('category')
df['active_degree'] = df['active_degree'].astype('category')  # 保持原始分组类别

# 处理缺失值
df = df.dropna(subset=['follow', 'fans', 'friend', 'active_degree'])


# 2. 先统计活跃分组的类别及数量（核心步骤：明确有哪些分组）
active_categories = df['active_degree'].value_counts().reset_index()
active_categories.columns = ['活跃分组', '用户数量']
active_categories['占比'] = active_categories['用户数量'] / len(df)
active_categories['占比'] = active_categories['占比'].apply(lambda x: f'{x:.2%}')

print("活跃分组类别及数量统计：")
print(active_categories.sort_values('用户数量', ascending=False))  # 按数量排序


# 3. 分析1：不同活跃分组的社交指标差异（直接用原始分组）
## 3.1 各活跃分组的社交指标均值对比
active_social_stats = df.groupby('active_degree')[['follow', 'fans', 'friend']].mean().reset_index()
active_social_stats = active_social_stats.rename(columns={'active_degree': '活跃分组'})
print("\n各活跃分组的社交指标均值：")
print(active_social_stats.round(2))

# 可视化：柱状图（按原始分组展示）
plt.figure(figsize=(12, 8))
# 关注数与活跃分组的关系
plt.subplot(3, 1, 1)
sns.barplot(x='活跃分组', y='follow', data=active_social_stats, palette='Set2')
plt.title('不同活跃分组的关注数均值')
plt.ylabel('关注数均值')
plt.xticks(rotation=45)  # 旋转标签避免重叠

# 粉丝数与活跃分组的关系
plt.subplot(3, 1, 2)
sns.barplot(x='活跃分组', y='fans', data=active_social_stats, palette='Set2')
plt.title('不同活跃分组的粉丝数均值')
plt.ylabel('粉丝数均值')
plt.xticks(rotation=45)

# 朋友数与活跃分组的关系
plt.subplot(3, 1, 3)
sns.barplot(x='活跃分组', y='friend', data=active_social_stats, palette='Set2')
plt.title('不同活跃分组的朋友数均值')
plt.ylabel('朋友数均值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 3.2 社交指标在活跃分组间的差异显著性（ANOVA检验）
# 提取原始活跃分组的类别列表
active_groups = df['active_degree'].cat.categories.tolist()

for col in ['follow', 'fans', 'friend']:
    # 按原始分组提取数据
    data_groups = [df[df['active_degree'] == g][col].dropna() for g in active_groups]
    f_val, p_val = f_oneway(*data_groups)
    significance = "存在显著差异" if p_val < 0.05 else "无显著差异"
    print(f"\n{col}数在各活跃分组间的ANOVA结果：F值={f_val.round(3)}，P值={p_val.round(4)}（{significance}）")


# 4. 分析2：直播主播的粉丝数与活跃分组的关系
live_users = df[df['is_live'] == '1']
if len(live_users) > 0:
    live_fans_stats = live_users.groupby('active_degree')['fans'].mean().reset_index()
    live_fans_stats = live_fans_stats.rename(columns={'active_degree': '活跃分组'})
    print("\n直播主播在各活跃分组的粉丝数均值：")
    print(live_fans_stats.round(2))
    
    # 可视化：折线图（展示趋势）
    plt.figure(figsize=(10, 5))
    sns.pointplot(x='活跃分组', y='fans', data=live_fans_stats, color='red', markers='o')
    plt.title('直播主播：粉丝数与活跃分组的关系')
    plt.xlabel('活跃分组')
    plt.ylabel('粉丝数均值')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("\n无直播主播样本，无法分析粉丝数与活跃分组的关系")


# 5. 分析3：朋友数与低活跃分组的关系（假设低活跃分组可从原始类别中识别，如'2_14_day_new'）
# 先从活跃分组中确认低活跃标签（例如你认为'2_14_day_new'是低活跃）
low_active_labels = ['2_14_day_new']  # 根据你的实际低活跃分组修改
df['is_lowactive'] = df['active_degree'].isin(low_active_labels)

# 按朋友数分组，计算低活跃比例
friend_bins = pd.qcut(df['friend'], q=2, labels=['低朋友数', '高朋友数'])
lowactive_by_friend = df.groupby(friend_bins)['is_lowactive'].mean().reset_index()
lowactive_by_friend = lowactive_by_friend.rename(columns={'friend': '朋友数组'})
lowactive_by_friend['低活跃比例'] = lowactive_by_friend['is_lowactive'].apply(lambda x: f'{x:.2%}')

print("\n不同朋友数组的低活跃比例：")
print(lowactive_by_friend[['朋友数组', '低活跃比例']])

# 可视化：柱状图
plt.figure(figsize=(8, 5))
sns.barplot(x='朋友数组', y='is_lowactive', data=lowactive_by_friend, palette='Set3')
plt.title('朋友数与低活跃分组的关系')
plt.ylabel('低活跃比例')
for i, v in enumerate(lowactive_by_friend['is_lowactive']):
    plt.text(i, v+0.02, f'{v:.2%}', ha='center')
plt.show()