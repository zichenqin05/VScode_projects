import sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def statistic():
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # 从数据库获取数据
    df = sql.do("""SELECT
            user_id,
            is_live_streamer,
            is_video_author,
            follow_user_num,
            fans_user_num,
            friend_user_num
            FROM
            user_features_pure""")

    # 转换为DataFrame并指定列名
    df = pd.DataFrame(df, columns=['user_id', 'live', 'author', 'followers', 'fans', 'friend'])

    # 转换数值列的数据类型
    numeric_cols = ['followers', 'fans', 'friend']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 转换身份标识为布尔值
    df['live'] = df['live'].map({'1': True, '0': False, True: True, False: False}).fillna(False)
    df['author'] = df['author'].map({'1': True, '0': False, True: True, False: False}).fillna(False)
    
    # 定义用户类型：普通用户、仅直播、仅视频创作者、交叉身份（既是直播又是视频创作者）
    df['user_type'] = '普通用户'
    df.loc[df['live'] & ~df['author'], 'user_type'] = '仅直播主播'
    df.loc[~df['live'] & df['author'], 'user_type'] = '仅视频创作者'
    df.loc[df['live'] & df['author'], 'user_type'] = '交叉身份用户'
    
    # 按用户类型分组
    groups = {
        '全部用户': df,
        '普通用户': df[df['user_type'] == '普通用户'],
        '仅直播主播': df[df['user_type'] == '仅直播主播'],
        '仅视频创作者': df[df['user_type'] == '仅视频创作者'],
        '交叉身份用户': df[df['user_type'] == '交叉身份用户']
    }
    
    # 查看基本统计量
    for name, group_df in groups.items():
        print(f"\n{name}：")
        print(group_df[numeric_cols].describe())

    return df

def graph(df):
    # 同时支持英文和中文（优先用中文字体，英文会自动适配）
    plt.rcParams["font.family"] = ["SimHei", "Arial", "sans-serif"]  # 中文优先，Arial作为英文后备

    # 设置颜色方案
    palette = sns.color_palette("Set2", 4)
    type_order = ['普通用户', '仅直播主播', '仅视频创作者', '交叉身份用户']
    
    # 1. 社交指标均值比较
    plt.figure(figsize=(15, 8))
    
    # 关注数均值
    plt.subplot(1, 3, 1)
    sns.barplot(x='user_type', y='followers', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('不同用户类型的关注数均值比较')
    plt.ylabel('关注数均值')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    # 粉丝数均值
    plt.subplot(1, 3, 2)
    sns.barplot(x='user_type', y='fans', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('不同用户类型的粉丝数均值比较')
    plt.ylabel('粉丝数均值')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    # 好友数均值
    plt.subplot(1, 3, 3)
    sns.barplot(x='user_type', y='friend', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('不同用户类型的好友数均值比较')
    plt.ylabel('好友数均值')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.show()
    
    # 2. 社交指标中位数比较（更能反映普通水平）
    plt.figure(figsize=(15, 8))
    
    # 关注数中位数
    plt.subplot(1, 3, 1)
    sns.barplot(x='user_type', y='followers', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('不同用户类型的关注数中位数比较')
    plt.ylabel('关注数中位数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    # 粉丝数中位数
    plt.subplot(1, 3, 2)
    sns.barplot(x='user_type', y='fans', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('不同用户类型的粉丝数中位数比较')
    plt.ylabel('粉丝数中位数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    # 好友数中位数
    plt.subplot(1, 3, 3)
    sns.barplot(x='user_type', y='friend', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('不同用户类型的好友数中位数比较')
    plt.ylabel('好友数中位数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.show()
        
    # 3. 小提琴图 - 展示完整数据分布
    plt.figure(figsize=(18, 6))
    
    # 关注数分布
    plt.subplot(1, 3, 1)
    sns.violinplot(x='user_type', y='followers', data=df, palette=palette, order=type_order)
    plt.title('不同用户类型的关注数分布')
    plt.ylabel('关注数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    plt.yscale('log')  # 使用对数刻度处理极端值
    
    # 粉丝数分布
    plt.subplot(1, 3, 2)
    sns.violinplot(x='user_type', y='fans', data=df, palette=palette, order=type_order)
    plt.title('不同用户类型的粉丝数分布')
    plt.ylabel('粉丝数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    plt.yscale('log')  # 使用对数刻度处理极端值
    
    # 好友数分布
    plt.subplot(1, 3, 3)
    sns.violinplot(x='user_type', y='friend', data=df, palette=palette, order=type_order)
    plt.title('不同用户类型的好友数分布')
    plt.ylabel('好友数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    plt.yscale('log')  # 使用对数刻度处理极端值
    
    plt.tight_layout()
    plt.show()
    
    # 4. 用户类型数量分布
    type_counts = df['user_type'].value_counts().reindex(type_order)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette=palette,
                hue=type_counts.index, legend=False)
    plt.title('不同用户类型的数量分布')
    plt.ylabel('用户数量')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    
    # 添加数量标签
    for i, v in enumerate(type_counts.values):
        plt.text(i, v + 50, f'{v}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # 5. 各类型用户的社交指标比率（粉丝数/关注数）
    df['fans_follow_ratio'] = df['fans'] / (df['followers'] + 1)  # +1避免除零
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='user_type', y='fans_follow_ratio', data=df, palette=palette,
                order=type_order, hue='user_type', legend=False)
    plt.title('不同用户类型的粉丝数/关注数比率')
    plt.ylabel('粉丝数/关注数')
    plt.xlabel('用户类型')
    plt.xticks(rotation=15)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# 执行分析和可视化
if __name__ == "__main__":
    df = statistic()
    graph(df)
