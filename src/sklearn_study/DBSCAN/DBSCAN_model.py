from sklearn.datasets import StandardScaler
import pandas as pd
import sklearn_study.KNN.load_data as ld
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
df = ld.load_data_DBSCAN()

# 查看前几行数据
print(df.head())

def describe_data(df):
    sns.pairplot(df, hue="user_active_degree", palette="Set1")
    plt.show()
    print(df.describe())

def correlation_heatmap(df):
    correlation_matrix = df.drop(columns=['user_active_degree','user_id']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()    

def standardize_data(df):
    features = df.drop(columns=['user_id','user_active_degree'])
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    standardized_df = pd.DataFrame(standardized_features, columns=features.columns)
    standardized_df['user_id'] = df['user_id'].values
    return standardized_df


if __name__ == "__main__":
    correlation_heatmap(df)