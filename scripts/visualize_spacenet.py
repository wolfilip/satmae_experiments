import os
import random

import matplotlib.pyplot as plt
import rasterio as rio

DataFolder = "/home/filip/SpaceNetV1/"
Raster = DataFolder + "3band/"
Vector = DataFolder + "geojson/"
Mask = DataFolder + "mask/"

raster_list = os.listdir(Raster)
raster_list.sort()
mask_list = os.listdir(Mask)
mask_list.sort()

fig, axes = plt.subplots(2, 10, figsize=(24, 7))
for i in range(10):
    j = random.randint(0, len(raster_list) - 1)
    img = rio.open(Raster + raster_list[j]).read()
    axes[0][i].imshow(img.transpose(1, 2, 0))
    mask = rio.open(Mask + mask_list[j]).read()
    axes[1][i].imshow(mask.squeeze())
