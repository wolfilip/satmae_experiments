import os

# from sklearn.model_selection import train_test_split

path_dataset = "/home/filip/resics-45/"
data_arr = {}
cnt = 0

dir_list = os.listdir(path_dataset)


# with open(path_dataset + "train-resisc.txt") as train_file:
#     for line in train_file:
#         dir_name = line.split('/')[1].rstrip()

#         if dir_name not in data_arr:
#             data_arr[dir_name] = cnt
#             cnt += 1

# print(data_arr)

f = open(path_dataset + "val-resisc-metadata.txt", "w")

with open(
    "/home/filip/resics-45-train-val/val-resisc-metadata.txt", "w"
) as metadata_file:
    with open("/home/filip/resics-45-train-val/val-resisc.txt") as train_file:
        for line in train_file:
            dir_name = line.split("/")[1].rstrip()
            file_name = line.split("/")[2].rstrip()
            for dir_list_name in dir_list:
                if dir_name == dir_list_name:
                    metadata_file.write(
                        str(dir_list.index(dir_list_name))
                        + ","
                        + path_dataset
                        + dir_name
                        + "/"
                        + file_name
                        + "\n"
                    )
