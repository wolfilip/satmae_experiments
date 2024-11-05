import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

path_dataset = "/home/filip/fMoW-Temporal/train/"

classes_all = {}
classes_curr = {}

dir_list = os.listdir(path_dataset)

for i in range(0, 20000, 100):
    classes_all[int(i)] = 0

for dir_name in dir_list:
    for i in classes_curr:
        if classes_curr[i] > 0:
            print(str(i) + ": " + str(classes_curr[i]))
    print(dir_name)

    for i in range(0, 20000, 100):
        classes_curr[int(i)] = 0

    for root, dirs, files in os.walk(path_dataset + dir_name):

        for file in files:
            if file.endswith("_rgb.jpg"):
                image = Image.open(root + "/" + file)
                # print(image.size[0] - image.size[0]%100)
                # for i in classes:
                #     if image.size[0] > i:
                num = image.size[0] - image.size[0] % 100
                classes_all[num] += 1
                classes_curr[num] += 1
                # print(file)
                # classes[]
    #             f = open(root + '/' + file)
    #             json_file = json.load(f)
    #             classes[dir_name].append(json_file['gsd'])

    # print(dir_name + ': ' + str(np.mean(classes[dir_name])) + " +- " + str(np.std(classes[dir_name])))

print(classes_all)
