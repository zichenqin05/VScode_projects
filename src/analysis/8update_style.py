import sql
import quality
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sql.do("""SELECT 
                video_id,
                upload_type
            FROM
                video_features_basic_pure""")

df = pd.DataFrame(df, columns=['video_id', 'upload_type'])
df2 = quality.video_quality()
data = pd.merge(df, df2, on='video_id')

# 按upload_type分组，计算统计量
stats = data.groupby('upload_type')['quality_level'].agg(
    样本量='count',
    均值='mean',
    标准差='std',
    最优值='min',
    最差值='max'
).reset_index()

# 处理标准差为NaN的情况（当某组只有1个样本时，标准差无意义）
stats['标准差'] = stats['标准差'].fillna(0)

print(stats)

def graph():
    # 设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))

    # 绘制箱线图
    sns.boxplot(
        data=data,
        x='upload_type',
        y='quality_level',
        palette='Set3'
    )

    # 添加散点显示原始数据分布
    sns.stripplot(
        data=data,
        x='upload_type',
        y='quality_level',
        color='black',
        size=5,
        alpha=0.5
    )

    plt.title('不同上传类型的用户体验质量分布', fontsize=14)
    plt.xlabel('上传类型', fontsize=12)
    plt.ylabel('质量水平', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def graph2():
    # 设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))

    # 绘制箱线图
    sns.boxplot(
        data=data,
        x='upload_type',
        y='quality_level',
        palette='Set3'
    )

    # 添加散点显示原始数据分布
    sns.stripplot(
        data=data,
        x='upload_type',
        y='quality_level',
        color='black',
        size=5,
        alpha=0.5
    )

    plt.title('不同上传类型的用户体验质量分布', fontsize=14)
    plt.xlabel('上传类型', fontsize=12)
    plt.ylabel('质量水平', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graph()
    graph2()