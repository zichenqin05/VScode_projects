import sql
import pandas as pd
import matplotlib.pyplot as plt
import random
import colorsys

def graph1(df):
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(12, 6))  # 设置画布大小
    bar_width = 0.6

    bars = plt.bar(df['music_id'], df['count'], width=bar_width, color='#4A90E2', label='次数')

    # 设置图表标题和坐标轴标签
    plt.title('音乐使用频率图', fontsize=16, pad=20)
    plt.xlabel('音乐id', fontsize=12, labelpad=10)
    plt.ylabel('次数', fontsize=12, labelpad=10, color='#4A90E2')

    # 设置网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(loc='upper left')

    # 调整x轴标签角度
    plt.xticks(rotation=45, ha='right')

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

def get_harmonious_colors(n):

    colors = []
    for i in range(n):
        # 均匀分布色相，确保颜色多样性
        hue = i / n + random.uniform(-0.1, 0.1)  # 增加一点随机性
        hue = hue % 1.0  # 确保在0-1范围内
        
        # 固定柔和的饱和度和明度
        saturation = random.uniform(0.4, 0.6)
        value = random.uniform(0.7, 0.9)
        
        # 转换为RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
    return colors

def graph2(df, value_col, label_col, title="音乐类型分布图", 
           explode=None, autopct='%1.1f%%', shadow=False, figsize=(8, 6)):
  
    # 设置中文字体
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    sizes = df[value_col].values
    labels = df[label_col].values
    
    # 生成和谐的随机颜色
    colors = get_harmonious_colors(len(sizes))
    
    # 绘制饼图
    plt.figure(figsize=figsize)
    wedges, texts, autotexts = plt.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct=autopct, shadow=shadow, startangle=90,
        textprops=dict(fontsize=10)  # 设置标签字体大小
    )
    
    # 美化百分比文本
    plt.setp(autotexts, size=9, weight="bold", color="white")
    
    # 确保饼图为正圆形
    plt.axis('equal')
    
    # 设置标题
    plt.title(title, fontsize=15, pad=20) 
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

def music_type_compare():
    df3 = sql.do("""SELECT 
            music_type as mt,
            COUNT(*) as count  
        FROM
            video_features_basic_pure
        WHERE 
            video_type = 'NORMAL'  
        GROUP BY
            mt;""")
    df3 = pd.DataFrame(df3, columns=['music_type','count'])

    df4 = sql.do("""SELECT 
            music_type as mt,
            COUNT(*) as count  
        FROM
            video_features_basic_pure
        WHERE 
            video_type = 'AD'  
        GROUP BY
            mt;""")
    df4 = pd.DataFrame(df4, columns=['music_type','count'])
    df_combined = pd.concat([df3, df4], axis=1)

    print(df_combined)

if __name__ == "__main__":
    df = sql.search('music_id', 'video_features_basic_pure')
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    df = df.iloc[1:10]
    print(df.head(10))
    graph1(df)

    df2 = sql.search('music_type', 'video_features_basic_pure')
    print(df2)
    graph2(df2,'count','music_type')

    music_type_compare()