import sql
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = sql.search('tag', 'video_features_basic_pure')
df = df.sort_values(by='count', ascending=False)
df = df.iloc[:20]

df2= sql.search('upload_dt', 'video_features_basic_pure')

print (df)
print (df2)

def graph1():
    plt.figure(figsize=(12, 6))  # 设置画布大小
    bar_width = 0.6

    bars = plt.bar(df['tag'], df['count'], width=bar_width, color='#4A90E2', label='count')

    # 设置图表标题和坐标轴标签
    plt.title('The distribution of video tag(tag 20)', fontsize=16, pad=20)
    plt.xlabel('tag', fontsize=12, labelpad=10)
    plt.ylabel('count', fontsize=12, labelpad=10, color='#4A90E2')


    # 设置网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='upper left')

    # 调整x轴标签角度
    plt.xticks(rotation=45, ha='right')

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

def graph2(df2):
    # 1. 将upload_dt转换为日期类型（确保正确排序）
    df2['upload_dt'] = pd.to_datetime(df2['upload_dt'])

    df2 = df2.sort_values(by='upload_dt')

    plt.figure(figsize=(12, 6))  # 设置画布大小
    bar_width = 0.6

    bars = plt.bar(df2['upload_dt'], df2['count'], width=bar_width, color='#4A90E2', label='count')

    # 设置图表标题和坐标轴标签
    plt.title('The distribution of video release times', fontsize=16, pad=20)
    plt.xlabel('date', fontsize=12, labelpad=10)
    plt.ylabel('count', fontsize=12, labelpad=10, color='#4A90E2')


    # 设置网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='upper left')

    # 调整x轴标签角度
    plt.xticks(rotation=45, ha='right')

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graph1()
    graph2(df2)
