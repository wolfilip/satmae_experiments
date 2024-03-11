import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/filip/baseline/data/input/train_data/train_62classes_new.csv')

data_arr = {}

for index, row in df.iterrows():
    prefix = row['image_path'][:-8].rsplit('_', 1)
    if prefix[0] not in data_arr:
        data_arr[prefix[0]] = 1
    else:
        data_arr[prefix[0]] += 1

data_keys = list(data_arr.keys())

print(len(data_keys))
_, new_arr = train_test_split(data_keys, test_size=0.1)
print(len(new_arr))
new_arr_set = set(new_arr)

df_new = pd.DataFrame(columns=['category', 'image_path', 'timestamp'])

for index, row in df.iterrows():
    prefix = row['image_path'][:-8].rsplit('_', 1)
    if prefix[0] in new_arr_set:
        df_new = df_new.append(row)

print(len(df_new))

df_new.to_csv('train_62classes_new_10pc.csv', index=False)


# prefix = single_image_name_1[:-8].rsplit('_', 1)