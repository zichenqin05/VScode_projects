import sql
import pandas as pd

df = sql.do("""SELECT
            user_id,
            user_active_degree,
            is_lowactive_period
            FROM
            user_features_pure
            WHERE
            user_active_degree = 'middle_active' """)

df = pd.DataFrame(df, columns=['id', 'degree', 'is_low'])

all_ratios = df['is_low'].value_counts(normalize=True)
print(all_ratios)
print (df.head())

## 算不出来啊全是0