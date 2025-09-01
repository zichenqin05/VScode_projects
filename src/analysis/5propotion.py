import sql
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Retrieve resolution data from database
    df = sql.do("""
    SELECT 
        server_width,
        server_height,
        COUNT(*) AS count
    FROM 
        video_features_basic_pure
    WHERE 
        server_width IS NOT NULL 
        AND server_height IS NOT NULL
    GROUP BY 
        server_width, server_height;
    """)
    df = pd.DataFrame(df, columns=['server_width', 'server_height', 'count'])

    # 2. Convert width and height to numeric types
    df['server_width'] = pd.to_numeric(df['server_width'])
    df['server_height'] = pd.to_numeric(df['server_height'])

    # 3. Core step: Normalize resolution by sorting dimensions to ensure consistent identification
    def normalize_resolution(row):
        # Take the smaller value as first, larger as second
        min_dim = min(row['server_width'], row['server_height'])
        max_dim = max(row['server_width'], row['server_height'])
        return f"{min_dim}×{max_dim}"

    # Add normalized resolution column
    df['normalized_resolution'] = df.apply(normalize_resolution, axis=1)

    # 4. Merge and sum by normalized resolution
    merged_df = df.groupby('normalized_resolution', as_index=False).agg({
        'count': 'sum'  # Merge counts
    })

    # 5. Calculate percentage
    total = merged_df['count'].sum()
    merged_df['percentage(%)'] = (merged_df['count'] / total * 100).round(2)

    # 6. Sort by count in descending order
    merged_df = merged_df.sort_values('count', ascending=False).reset_index(drop=True)

    # 7. Display results
    print(merged_df.head())
    
    return merged_df

if __name__=="__main__":

    plt.rcParams['font.family'] = ['SimHei']  # Keep Chinese font support for potential use
    plt.rcParams['axes.unicode_minus'] = False

    # Output data (keep top 5)
    df2 = main()
    df2 = df2.iloc[:5]

    plt.figure(figsize=(12, 6))  # Set figure size
    bar_width = 0.6

    bars = plt.bar(df2['normalized_resolution'], df2['count'], width=bar_width, color='#4A90E2', label='Count')
    ax2 = plt.twinx()
    ax2.plot(df2['normalized_resolution'], df2['percentage(%)'], color='#FF7A00', marker='o', linestyle='-', linewidth=2, label='Percentage(%)')

    # Set chart title and axis labels
    plt.title('Video Resolution Distribution Statistics', fontsize=16, pad=20)
    plt.xlabel('Resolution', fontsize=12, labelpad=10)
    plt.ylabel('Count', fontsize=12, labelpad=10, color='#4A90E2')
    ax2.set_ylabel('Percentage(%)', fontsize=12, labelpad=10, color='#FF7A00')

    # Set grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper left')

    # Adjust x-axis label angle
    plt.xticks(rotation=45, ha='right')

    # Adjust layout and display chart
    plt.tight_layout()
    plt.show()
