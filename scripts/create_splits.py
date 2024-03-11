import os

path = '/home/filip/scale-mae/mae/data/train/'

with open("train-fmow-all.txt", "w") as metadata_file:
    for root, dirs, files in os.walk(path):
        for file in files:
            metadata_file.write(root + '/' + file + '\n')
