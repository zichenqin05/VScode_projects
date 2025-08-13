from sklearn.preprocessing import StandardScaler
import pandas as pd
import load_data as ld
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 加载数据
data = ld.load_data_KNN()

def describe_data(df):
    """数据可视化与统计描述"""
    # 绘制特征间关系图
    sns.pairplot(df, hue="quality_level", palette="Set1")
    plt.show()
    # 输出统计信息
    print("\n数据统计描述：")
    print(df.describe())

def correlation_heatmap(df):
    """绘制特征相关性热力图"""
    # 排除非特征列计算相关性
    plt.rcParams['font.family'] = ['SimHei']
    correlation_matrix = df.drop(columns=['quality_level', 'video_id']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.show()    

def prepare_data(df):
    """准备数据：分离特征和标签，不包含标准化"""
    # 保留所有必要列用于后续处理
    prepared_df = df.copy()
    return prepared_df

def main():
    # 数据准备（不包含标准化）
    prepared_data = prepare_data(data)
    
    # 划分特征(X)和标签(y)
    X = prepared_data.drop(columns=['video_id', 'quality_level','play_progress'])
    y = prepared_data['quality_level']
    
    # 数据分割（训练集和测试集）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化（仅在分割后进行一次）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合
    X_test_scaled = scaler.transform(X_test)        # 用相同的scaler转换测试集
    
    # 构建并训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    
    # 预测与评估
    y_pred = knn.predict(X_test_scaled)
    print(f"\n模型准确率：{accuracy_score(y_test, y_pred):.2f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    # 保存模型和标准化器（保存到当前目录）
    joblib.dump(knn, r'src\sklearn_study\KNN\models\knn_model.pkl')  # 保存模型
    joblib.dump(scaler, r'src\sklearn_study\KNN\models\scaler.pkl')   # 保存标准化器
    # 保存特征列名（确保预测时特征顺序一致）
    with open(r'src\sklearn_study\KNN\models\feature_columns.txt', 'w') as f:
        f.write(','.join(X.columns))
    
    print("\n模型、标准化器和特征列已保存")

if __name__ == "__main__":
    main()
