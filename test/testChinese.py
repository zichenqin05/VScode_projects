import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.rcParams["font.family"] = ["SimHei", "Arial", "sans-serif"]  # 中文优先，Arial作为英文后备
# 3. 测试绘图（确保中文正常显示）
def test_plot():
    df = pd.DataFrame({
        "用户类型": ["普通用户", "直播主播", "视频创作者"],
        "粉丝数": [1200, 5000, 3500]
    })
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x="用户类型", y="粉丝数", data=df)
    plt.title("不同用户类型的粉丝数对比")  # 中文标题
    plt.xlabel("用户类型")  # 中文标签
    plt.ylabel("粉丝数")
    plt.show()

test_plot()