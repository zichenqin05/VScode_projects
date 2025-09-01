import sql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set font (can be changed to default if not needed)
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def time_interval(df):
    """Divide duration into intervals and return statistics"""
    if 'duration_seconds' not in df.columns:
        raise ValueError("DataFrame must contain 'duration_seconds' column")
    
    # Define intervals
    intervals = []
    intervals.append((0, 0.01, '0s (photo)'))  # Special interval
    
    # Short duration (0.01-600s, every 30s an interval)
    i = 0.01
    while i < 10 * 60:
        next_i = i + 30
        intervals.append((i, next_i, f'{i:.2f}-{next_i:.0f}s'))
        i = next_i
    
    # Medium duration (600-3600s, every 5min an interval)
    for i in range(10*60, 60*60, 5*60):
        intervals.append((i, i+5*60, f'{i}-{i+5*60}s'))
    
    # Long duration (above 3600s)
    intervals.append((3600, 18000, '3600-18000s'))
    intervals.append((18000, float('inf'), 'above 18000s'))

    # Count number in each interval
    duration_stats = {interval[2]: 0 for interval in intervals}
    for duration in df['duration_seconds']:
        for low, high, label in intervals:
            if low <= duration < high:
                duration_stats[label] += 1
                break
    
    # Convert to DataFrame and calculate percentage
    result_df = pd.DataFrame(list(duration_stats.items()), columns=['Duration Interval', 'Count'])
    total = result_df['Count'].sum()
    result_df['Percent(%)'] = (result_df['Count'] / total * 100).round(2)
    
    # Sort
    result_df['Interval Order'] = result_df['Duration Interval'].apply(
        lambda x: [i for i, interval in enumerate(intervals) if interval[2] == x][0]
    )

    result_df = result_df.iloc[:23] 

    return result_df.sort_values('Interval Order').drop('Interval Order', axis=1).reset_index(drop=True)

def calculate_stats(df):
    """Calculate key statistics"""
    positive_durations = df[df['duration_seconds'] > 0]['duration_seconds']
    stats = {
        'Total Samples': len(df),
        'Valid Samples': len(positive_durations),
        'Mean': positive_durations.mean() if not positive_durations.empty else None,
        'Median': positive_durations.median() if not positive_durations.empty else None,
        'Min Positive': positive_durations.min() if not positive_durations.empty else None,
        'Max': df['duration_seconds'].max()
    }
    return stats

def plot_duration_distribution(result_df, stats):
    """Plot interval distribution bar chart and show statistics in the top right"""
    plt.figure(figsize=(16, 8))
    
    # Plot bar chart
    bars = plt.bar(result_df['Duration Interval'], result_df['Count'], color='#4A90E2', alpha=0.8)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'{height}', ha='center', va='bottom', fontsize=9)
    
    # Set chart title and axes
    plt.title('Video Duration Interval Distribution', fontsize=16, pad=20)
    plt.xlabel('Duration Interval', fontsize=12, labelpad=10)
    plt.ylabel('Video Count', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add statistics textbox
    stats_text = "Statistics:\n"
    stats_text += f"Total Samples: {stats['Total Samples']}\n"
    stats_text += f"Valid Samples: {stats['Valid Samples']}\n"
    stats_text += f"Mean: {stats['Mean']:.1f}s\n" if stats['Mean'] else "Mean: N/A\n"
    stats_text += f"Median: {stats['Median']:.1f}s\n" if stats['Median'] else "Median: N/A\n"
    stats_text += f"Min Positive: {stats['Min Positive']:.1f}s\n" if stats['Min Positive'] else "Min Positive: N/A\n"
    stats_text += f"Max: {stats['Max']:.1f}s" if stats['Max'] else "Max: N/A"
    
    # Place textbox at top right
    plt.gcf().text(0.87, 0.83, stats_text, fontsize=10, 
                  bbox=dict(facecolor='white', edgecolor='gray', pad=10),
                  verticalalignment='top',  
                  horizontalalignment='left')  
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, top=0.85)
    plt.show()

if __name__ == "__main__":
    # 1. Get data
    df = sql.do("""
        SELECT video_duration 
        FROM video_features_basic_pure
        WHERE video_duration IS NOT NULL;
    """)
    df = pd.DataFrame(df, columns=['video_duration'])
    df['video_duration'] = pd.to_numeric(df['video_duration'])
    df['duration_seconds'] = df['video_duration'] / 1000  # ms to s
    
    # 2. Calculate interval distribution and statistics
    interval_df = time_interval(df)
    stats = calculate_stats(df)
    
    # 3. Plot
    plot_duration_distribution(interval_df, stats)