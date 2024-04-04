import os


path_dataset = "/home/filip/EuroSAT/EuroSAT_RGB/"
backup_path = "/home/filip/resics-45-backup-2/"

with open(path_dataset + "val-eurosat.txt") as val_file:
    for line in val_file:
        dir_name = line.split("/")[0].rstrip()
        file_name = line.split("/")[1].rstrip()
        if not os.path.exists(path_dataset + "val/" + dir_name):
            os.mkdir(path_dataset + "val/" + dir_name)
        os.rename(
            path_dataset + dir_name + "/" + file_name,
            path_dataset + "val/" + dir_name + "/" + file_name,
        )
