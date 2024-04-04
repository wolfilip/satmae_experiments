import os
import json
import numpy as np

path_dataset = '/home/filip/fMoW-Temporal/train/'

classes = {}

dir_list = os.listdir(path_dataset)

for dir_name in dir_list:
    classes[dir_name] = []
    
    for root, dirs, files in os.walk(path_dataset + dir_name):

        for file in files:
            if file.endswith('_rgb.json'):
                f = open(root + '/' + file)
                json_file = json.load(f)
                classes[dir_name].append(json_file['gsd'])

    print(dir_name + ': ' + str(np.mean(classes[dir_name])) + " +- " + str(np.std(classes[dir_name])))