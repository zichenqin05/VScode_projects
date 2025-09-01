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

# Group by upload_type and calculate statistics
stats = data.groupby('upload_type')['quality_level'].agg(
    sample_count='count',
    mean_value='mean',
    std_deviation='std',
    best_value='min',
    worst_value='max'
).reset_index()

# Handle NaN in std_deviation (meaningless when a group has only 1 sample)
stats['std_deviation'] = stats['std_deviation'].fillna(0)

print(stats)

def graph():
    # Set Chinese font (retained for potential Chinese display needs)
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))

    # Plot boxplot
    sns.boxplot(
        data=data,
        x='upload_type',
        y='quality_level',
        palette='Set3'
    )

    # Add scatter plot to show raw data distribution
    sns.stripplot(
        data=data,
        x='upload_type',
        y='quality_level',
        color='black',
        size=5,
        alpha=0.5
    )

    plt.title('User Experience Quality Distribution by Upload Type', fontsize=14)
    plt.xlabel('Upload Type', fontsize=12)
    plt.ylabel('Quality Level', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def graph2():
    # Set Chinese font (retained for potential Chinese display needs)
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))

    # Plot bar chart (x-axis: upload type, y-axis: mean value, error bars: standard deviation)
    sns.barplot(
        data=data,
        x='upload_type',
        y='quality_level',
        errorbar='sd',  # Error bars show standard deviation
        capsize=0.1,    # Length of horizontal lines on error bars
        palette='Set2'
    )

    # Add title and labels
    plt.title('User Experience Quality Level by Upload Type', fontsize=14)
    plt.xlabel('Upload Type', fontsize=12)
    plt.ylabel('Quality Level (Mean ± Std Dev)', fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlap
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Auto-adjust layout
    plt.show()

if __name__ == "__main__":
    graph()
    graph2()