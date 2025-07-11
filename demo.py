import pymysql
import pandas as pd
import matplotlib.pyplot as plt

conn = pymysql.connect(host =  'localhost',
                      user = 'root',
                      password = '20051108',
                      db = 'demo',
                          port = 3306,
                          charset = 'utf8'
                      )

def sql_do(query):
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

data = sql_do("SELECT DATE(readable_time1) AS date, COUNT(*) AS record_count FROM sampled_data GROUP BY DATE(readable_time1) ORDER BY date")
print(data);

df = pd.DataFrame(data, columns = ['times', 'counts'])
print(df);

df = df.drop(index=0) 

# 将 times 列转换为字符串（如果不是的话）
df['times'] = df['times'].astype(str)

plt.figure(figsize=(10, 6))
plt.bar(df['times'], df['counts'])
plt.xlabel('Date')
plt.ylabel('Record Count')
plt.title('Record Count by Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()