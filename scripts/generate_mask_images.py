import os
import random
from tqdm import tqdm
from PIL import Image
from osgeo import gdal
from osgeo import ogr
import rasterio as rio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt


DataFolder = "/home/filip/SpaceNetV1/"
Raster = DataFolder + "3band/"
Vector = DataFolder + "geojson/"
Mask = DataFolder + "mask/"


def create_poly_mask(rasterSrc, vectorSrc, npDistFileName=""):
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize
    dstPath = npDistFileName
    memdrv = gdal.GetDriverByName("GTiff")
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(
        dst_ds, [1], source_layer, burn_values=[1], options=["COMPRESS=LZW"]
    )
    dst_ds = 0
    mask_image = Image.open(dstPath)
    mask_image = np.array(mask_image)
    return mask_image


def build_labels(src_raster_dir, src_vector_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    file_count = len(
        [f for f in os.walk(src_vector_dir).__next__()[2] if f[-8:] == ".geojson"]
    )
    print(f"Found {file_count} geojson files. Preparing building mask images...")
    for idx in tqdm(range(1, file_count + 1)):
        src_raster_filename = f"3band_AOI_1_RIO_img{idx}.tif"
        src_vector_filename = f"Geo_AOI_1_RIO_img{idx}.geojson"
        dst_filename = f"AOI_1_RIO_img{idx}.tif"
        src_raster_path = os.path.join(src_raster_dir, src_raster_filename)
        src_vector_path = os.path.join(src_vector_dir, src_vector_filename)
        dst_path = os.path.join(dst_dir, dst_filename)
        create_poly_mask(src_raster_path, src_vector_path, npDistFileName=dst_path)


build_labels(Raster, Vector, Mask)
