import sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Chinese font display
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # Display negative signs correctly

def statistic():
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Retrieve data from database
    df = sql.do("""SELECT
            user_id,
            is_live_streamer,
            is_video_author,
            follow_user_num,
            fans_user_num,
            friend_user_num
            FROM
            user_features_pure""")

    # Convert to DataFrame and specify column names
    df = pd.DataFrame(df, columns=['user_id', 'live', 'author', 'followers', 'fans', 'friend'])

    # Convert data types of numeric columns
    numeric_cols = ['followers', 'fans', 'friend']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Convert identity indicators to boolean values
    df['live'] = df['live'].map({'1': True, '0': False, True: True, False: False}).fillna(False)
    df['author'] = df['author'].map({'1': True, '0': False, True: True, False: False}).fillna(False)
    
    # Define user types: Regular User, Live Streamer Only, Video Creator Only, Cross-Identity User (both streamer and creator)
    df['user_type'] = 'Regular User'
    df.loc[df['live'] & ~df['author'], 'user_type'] = 'Live Streamer Only'
    df.loc[~df['live'] & df['author'], 'user_type'] = 'Video Creator Only'
    df.loc[df['live'] & df['author'], 'user_type'] = 'Cross-Identity User'
    
    # Group by user type
    groups = {
        'All Users': df,
        'Regular User': df[df['user_type'] == 'Regular User'],
        'Live Streamer Only': df[df['user_type'] == 'Live Streamer Only'],
        'Video Creator Only': df[df['user_type'] == 'Video Creator Only'],
        'Cross-Identity User': df[df['user_type'] == 'Cross-Identity User']
    }
    
    # View basic statistics
    for name, group_df in groups.items():
        print(f"\n{name}:")
        print(group_df[numeric_cols].describe())

    return df

def graph(df):
    # Support both English and Chinese (prioritize Chinese font, English adapts automatically)
    plt.rcParams["font.family"] = ["SimHei", "Arial", "sans-serif"]  # Chinese first, Arial as English fallback

    # Set color scheme
    palette = sns.color_palette("Set2", 4)
    type_order = ['Regular User', 'Live Streamer Only', 'Video Creator Only', 'Cross-Identity User']
    
    # 1. Comparison of average social metrics
    plt.figure(figsize=(15, 8))
    
    # Average number of followers
    plt.subplot(1, 3, 1)
    sns.barplot(x='user_type', y='followers', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('Average Number of Followers by User Type')
    plt.ylabel('Average Number of Followers')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    # Average number of fans
    plt.subplot(1, 3, 2)
    sns.barplot(x='user_type', y='fans', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('Average Number of Fans by User Type')
    plt.ylabel('Average Number of Fans')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    # Average number of friends
    plt.subplot(1, 3, 3)
    sns.barplot(x='user_type', y='friend', data=df, palette=palette, order=type_order,
                hue='user_type', legend=False)
    plt.title('Average Number of Friends by User Type')
    plt.ylabel('Average Number of Friends')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Comparison of median social metrics (better reflects average level)
    plt.figure(figsize=(15, 8))
    
    # Median number of followers
    plt.subplot(1, 3, 1)
    sns.barplot(x='user_type', y='followers', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('Median Number of Followers by User Type')
    plt.ylabel('Median Number of Followers')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    # Median number of fans
    plt.subplot(1, 3, 2)
    sns.barplot(x='user_type', y='fans', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('Median Number of Fans by User Type')
    plt.ylabel('Median Number of Fans')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    # Median number of friends
    plt.subplot(1, 3, 3)
    sns.barplot(x='user_type', y='friend', data=df, palette=palette, order=type_order,
                estimator=np.median, hue='user_type', legend=False)
    plt.title('Median Number of Friends by User Type')
    plt.ylabel('Median Number of Friends')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.show()
        
    # 3. Violin plot - show complete data distribution
    plt.figure(figsize=(18, 6))
    
    # Followers distribution
    plt.subplot(1, 3, 1)
    sns.violinplot(x='user_type', y='followers', data=df, palette=palette, order=type_order)
    plt.title('Followers Distribution by User Type')
    plt.ylabel('Number of Followers')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    plt.yscale('log')  # Use log scale to handle outliers
    
    # Fans distribution
    plt.subplot(1, 3, 2)
    sns.violinplot(x='user_type', y='fans', data=df, palette=palette, order=type_order)
    plt.title('Fans Distribution by User Type')
    plt.ylabel('Number of Fans')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    plt.yscale('log')  # Use log scale to handle outliers
    
    # Friends distribution
    plt.subplot(1, 3, 3)
    sns.violinplot(x='user_type', y='friend', data=df, palette=palette, order=type_order)
    plt.title('Friends Distribution by User Type')
    plt.ylabel('Number of Friends')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    plt.yscale('log')  # Use log scale to handle outliers
    
    plt.tight_layout()
    plt.show()
    
    # 4. User count distribution by user type
    type_counts = df['user_type'].value_counts().reindex(type_order)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette=palette,
                hue=type_counts.index, legend=False)
    plt.title('User Count Distribution by User Type')
    plt.ylabel('Number of Users')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    
    # Add count labels
    for i, v in enumerate(type_counts.values):
        plt.text(i, v + 50, f'{v}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Social metric ratio (fans/followers) for each user type
    df['fans_follow_ratio'] = df['fans'] / (df['followers'] + 1)  # +1 to avoid division by zero
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='user_type', y='fans_follow_ratio', data=df, palette=palette,
                order=type_order, hue='user_type', legend=False)
    plt.title('Fans-to-Followers Ratio by User Type')
    plt.ylabel('Fans-to-Followers Ratio')
    plt.xlabel('User Type')
    plt.xticks(rotation=15)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Execute analysis and visualization
if __name__ == "__main__":
    df = statistic()
    graph(df)