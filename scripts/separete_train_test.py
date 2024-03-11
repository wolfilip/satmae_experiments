import os


path_dataset = '/home/filip/resics-45/'
backup_path = '/home/filip/resics-45-backup-2/'

with open(path_dataset + "val-resisc.txt") as val_file:
    for line in val_file:
        dir_name = line.split('/')[1].rstrip()
        file_name = line.split('/')[2].rstrip()
        if not os.path.exists(path_dataset + 'val/' + dir_name):
            os.mkdir(path_dataset + 'val/' + dir_name)
        os.rename(backup_path  + file_name, path_dataset +  'val/' + dir_name + '/' + file_name)
