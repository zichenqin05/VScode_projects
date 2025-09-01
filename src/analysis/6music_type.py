import sql
import pandas as pd
import matplotlib.pyplot as plt
import random
import colorsys

def graph1(df):
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(12, 6))  # Set figure size
    bar_width = 0.6

    bars = plt.bar(df['music_id'], df['count'], width=bar_width, color='#4A90E2', label='Count')

    # Set chart title and axis labels
    plt.title('Music Usage Frequency', fontsize=16, pad=20)
    plt.xlabel('Music ID', fontsize=12, labelpad=10)
    plt.ylabel('Count', fontsize=12, labelpad=10, color='#4A90E2')

    # Set grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper left')

    # Adjust x-axis label angle
    plt.xticks(rotation=45, ha='right')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def get_harmonious_colors(n):

    colors = []
    for i in range(n):
        # Distribute hue evenly for color diversity
        hue = i / n + random.uniform(-0.1, 0.1)  # Add some randomness
        hue = hue % 1.0  # Ensure within 0-1 range
        
        # Fixed soft saturation and brightness
        saturation = random.uniform(0.4, 0.6)
        value = random.uniform(0.7, 0.9)
        
        # Convert to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
    return colors

def graph2(df, value_col, label_col, title="Music Type Distribution", 
           explode=None, autopct='%1.1f%%', shadow=False, figsize=(8, 6)):
  
    # Set Chinese font
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Extract data
    sizes = df[value_col].values
    labels = df[label_col].values
    
    # Generate harmonious random colors
    colors = get_harmonious_colors(len(sizes))
    
    # Draw pie chart
    plt.figure(figsize=figsize)
    wedges, texts, autotexts = plt.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct=autopct, shadow=shadow, startangle=90,
        textprops=dict(fontsize=10)  # Set label font size
    )
    
    # Beautify percentage text
    plt.setp(autotexts, size=9, weight="bold", color="white")
    
    # Ensure pie chart is circular
    plt.axis('equal')
    
    # Set title
    plt.title(title, fontsize=15, pad=20) 
    
    # Adjust layout and display
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
