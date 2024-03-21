import os
from sklearn.model_selection import train_test_split

path = '/home/filip/resics-45/'

# open("train-resisc-metadata.txt", "w") as metadata_file:
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             metadata_file.write(root + '/' + file + '\n')


# print(len(data_keys))
# _, new_arr = train_test_split(data_keys, test_size=0.1)
# print(len(new_arr))
# new_arr_set = set(new_arr)

# df_new = pd.DataFrame(columns=['category', 'image_path', 'timestamp'])

# for index, row in df.iterrows():
#     prefix = row['image_path'][:-8].rsplit('_', 1)
#     if prefix[0] in new_arr_set:
#         df_new = df_new.append(row)

# print(len(df_new))

# df_new.to_csv('train_62classes_new_10pc.csv', index=False)

new_array = []

f = open(path + "train-resisc-metadata.txt", "r")
new_array = f.readlines()

print(len(new_array))

_, arr_small = train_test_split(new_array, test_size=0.1)

print(len(arr_small))

f = open(path + "train-resisc-metadata-10pc.txt", "w")

f.writelines(arr_small)
