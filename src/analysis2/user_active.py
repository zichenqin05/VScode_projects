import sql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Set Chinese font for display
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. Data Acquisition and Preprocessing
df = sql.do("""SELECT
            user_id,
            follow_user_num,
            fans_user_num,
            friend_user_num,
            user_active_degree,
            is_video_author,
            is_live_streamer
            FROM
            user_features_pure""")

df = pd.DataFrame(
    df, 
    columns=['user_id','follow', 'fans', 'friend', 'active_degree', 'is_author', 'is_live']
)

# Data type conversion
df[['follow', 'fans', 'friend']] = df[['follow', 'fans', 'friend']].apply(
    pd.to_numeric, errors='coerce'
)
df['is_author'] = df['is_author'].astype('category')
df['is_live'] = df['is_live'].astype('category')
df['active_degree'] = df['active_degree'].astype('category')  # Keep original grouping categories

# Handle missing values
df = df.dropna(subset=['follow', 'fans', 'friend', 'active_degree'])


# 2. First, count categories and quantities of active groups (Core step: Identify all groups)
active_categories = df['active_degree'].value_counts().reset_index()
active_categories.columns = ['Active Group', 'User Count']
active_categories['Proportion'] = active_categories['User Count'] / len(df)
active_categories['Proportion'] = active_categories['Proportion'].apply(lambda x: f'{x:.2%}')

print("Statistics of Active Group Categories and Quantities:")
print(active_categories.sort_values('User Count', ascending=False))  # Sort by quantity


# 3. Analysis 1: Differences in social metrics across active groups (Using original groups directly)
## 3.1 Comparison of average social metrics across active groups
active_social_stats = df.groupby('active_degree')[['follow', 'fans', 'friend']].mean().reset_index()
active_social_stats = active_social_stats.rename(columns={'active_degree': 'Active Group'})
print("\nAverage Social Metrics by Active Group:")
print(active_social_stats.round(2))

# Visualization: Bar charts (Display by original groups)
plt.figure(figsize=(12, 8))
# Relationship between follow count and active groups
plt.subplot(3, 1, 1)
sns.barplot(x='Active Group', y='follow', data=active_social_stats, palette='Set2')
plt.title('Average Follow Count by Active Group')
plt.ylabel('Average Follow Count')
plt.xticks(rotation=45)  # Rotate labels to avoid overlap

# Relationship between fan count and active groups
plt.subplot(3, 1, 2)
sns.barplot(x='Active Group', y='fans', data=active_social_stats, palette='Set2')
plt.title('Average Fan Count by Active Group')
plt.ylabel('Average Fan Count')
plt.xticks(rotation=45)

# Relationship between friend count and active groups
plt.subplot(3, 1, 3)
sns.barplot(x='Active Group', y='friend', data=active_social_stats, palette='Set2')
plt.title('Average Friend Count by Active Group')
plt.ylabel('Average Friend Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 3.2 Significance of differences in social metrics across active groups (ANOVA test)
# Extract list of original active group categories
active_groups = df['active_degree'].cat.categories.tolist()

for col in ['follow', 'fans', 'friend']:
    # Extract data by original groups
    data_groups = [df[df['active_degree'] == g][col].dropna() for g in active_groups]
    f_val, p_val = f_oneway(*data_groups)
    significance = "Significant difference exists" if p_val < 0.05 else "No significant difference"
    print(f"\nANOVA Result for {col} Count Across Active Groups: F-value={f_val.round(3)}, P-value={p_val.round(4)} ({significance})")


# 4. Analysis 2: Relationship between fan count of live streamers and active groups
live_users = df[df['is_live'] == '1']
if len(live_users) > 0:
    live_fans_stats = live_users.groupby('active_degree')['fans'].mean().reset_index()
    live_fans_stats = live_fans_stats.rename(columns={'active_degree': 'Active Group'})
    print("\nAverage Fan Count of Live Streamers by Active Group:")
    print(live_fans_stats.round(2))
    
    # Visualization: Line chart (Show trend)
    plt.figure(figsize=(10, 5))
    sns.pointplot(x='Active Group', y='fans', data=live_fans_stats, color='red', markers='o')
    plt.title('Live Streamers: Relationship Between Fan Count and Active Group')
    plt.xlabel('Active Group')
    plt.ylabel('Average Fan Count')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("\nNo live streamer samples available; cannot analyze relationship between fan count and active groups")


# 5. Analysis 3: Relationship between friend count and low-active groups (Assume low-active groups can be identified from original categories, e.g., '2_14_day_new')
# First, confirm low-active labels from active groups (e.g., '2_14_day_new' is considered low-active in your case)
low_active_labels = ['2_14_day_new']  # Modify based on your actual low-active groups
df['is_lowactive'] = df['active_degree'].isin(low_active_labels)

# Group by friend count and calculate low-active proportion
friend_bins = pd.qcut(df['friend'], q=2, labels=['Low Friend Count', 'High Friend Count'])
lowactive_by_friend = df.groupby(friend_bins)['is_lowactive'].mean().reset_index()
lowactive_by_friend = lowactive_by_friend.rename(columns={'friend': 'Friend Count Group'})
lowactive_by_friend['Low-Active Proportion'] = lowactive_by_friend['is_lowactive'].apply(lambda x: f'{x:.2%}')

print("\nLow-Active Proportion by Friend Count Group:")
print(lowactive_by_friend[['Friend Count Group', 'Low-Active Proportion']])

# Visualization: Bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='Friend Count Group', y='is_lowactive', data=lowactive_by_friend, palette='Set3')
plt.title('Relationship Between Friend Count and Low-Active Groups')
plt.ylabel('Low-Active Proportion')
for i, v in enumerate(lowactive_by_friend['is_lowactive']):
    plt.text(i, v+0.02, f'{v:.2%}', ha='center')
plt.show()