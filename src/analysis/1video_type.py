import sql
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Get the count of each type
df = sql.search('video_type','video_features_basic_pure')

# Calculate total count
total = df['count'].sum()

# Calculate percentage (keep two decimals)
df['percent'] = (df['count'] / total * 100).round(2)

print(df)

plt.figure(figsize=(12, 6))  # Set canvas size
bar_width = 0.6

bars = plt.bar(df['video_type'], df['count'], width=bar_width, color='#4A90E2', label='Count')

# Draw right-side percentage axis
ax2 = plt.twinx()
ax2.plot(df['video_type'], df['percent'], color='#FF7A00', marker='o', linestyle='-', linewidth=2, label='Percent(%)')

# Set chart title and axis labels
plt.title('The Distribution of Video Types', fontsize=16, pad=20)
plt.xlabel('Video Type', fontsize=12, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad=10, color='#4A90E2')
ax2.set_ylabel('Percent(%)', fontsize=12, labelpad=10, color='#FF7A00')

# Set grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legends
plt.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjust x-axis label angle
plt.xticks(rotation=45, ha='right')

# Adjust layout and show chart
plt.tight_layout()
plt.show()