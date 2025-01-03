import os
import random
import warnings
from glob import glob
from typing import Any, List, Optional

import kornia.augmentation as K
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import rasterio as rio
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from rasterio import logging
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

log = logging.getLogger()
log.setLevel(logging.ERROR)

warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def rgb2mask(img):

    color2index = {
        (255, 255, 255): 0,
        (0, 0, 255): 1,
        (0, 255, 255): 2,
        (0, 255, 0): 3,
        (255, 255, 0): 4,
        (255, 0, 0): 5,
        (0, 0, 0): 6,
    }

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0], [1], [2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id == c] = color2index[tuple(img[img_id == c][0])]
        except:
            pass
    return mask.astype(int)


CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                )
            )
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(
                    input_size, scale=(0.2, 1.0), interpolation=interpol_mode
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(
            transforms.Compose(
                [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            )
        )
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class SpaceNetDataset(SatelliteDataset):
    def __init__(self, raster, mask, raster_list, mask_list, is_train):
        super().__init__(in_c=3)
        self.raster = raster
        self.mask = mask
        self.raster_list = raster_list
        self.mask_list = mask_list
        self.s = 224

        self.is_train = is_train
        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.s, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32),
                    ]
                ),
            ]
        )

        self.transforms_test = transforms.Compose(
            [
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32),
                    ]
                ),
                # transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_val = transforms.Compose(
            [
                # transforms.Resize(self.s),
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32),
                    ]
                ),
            ]
        )

    def __len__(self):
        return len(self.raster_list)

    def __getitem__(self, index):
        img = (
            rio.open(self.raster + self.raster_list[index]).read(
                out_shape=(self.s, self.s), resampling=Resampling.bilinear
            )
            / 255
        )
        mask = (
            rio.open(self.mask + self.mask_list[index])
            .read(out_shape=(self.s, self.s), resampling=Resampling.bilinear)
            .squeeze()
        )
        img = torch.from_numpy(img.astype("float32"))
        mask = torch.from_numpy(mask.astype("int64"))
        if self.is_train:
            image, mask = self.transforms_train(img, mask)
            image = self.transforms_distort(image)
            # image, mask = self.transforms_test(image, mask)
        else:
            image, mask = self.transforms_val(img, mask)
        # mask = F.one_hot(mask, num_classes=2).permute(2, 0, 1)
        # if self.is_train:
        #     image_and_mask = torch.cat([img, mask], dim=0)
        #     image_and_mask = self.transforms_train(image_and_mask)
        #     img, mask = torch.split(image_and_mask, [3, 2])
        #     img = self.transforms_distort(img)
        #     mask = mask.to(torch.int64)
        # else:
        #     image_and_mask = torch.cat([img, mask], dim=0)
        #     image_and_mask = self.transforms_val(image_and_mask)
        #     img, mask = torch.split(image_and_mask, [3, 2])
        #     mask = mask.to(torch.int64)
        # img, mask = torch.split(image_and_mask, [3, 1])
        # image_and_mask = self.transforms_val(image_and_mask)
        return img, mask


class VaihingenPotsdamDataset(SatelliteDataset):
    def __init__(self, img_path, mask_path, is_train, args):
        super().__init__(in_c=3)

        self.img_path = img_path
        self.mask_path = mask_path
        self.s = 512
        self.is_train = is_train

        self.image_filenames = []  # List to store image file names
        self.mask_filenames = []  # List to store mask file names

        self.image_filenames_temp = sorted(os.listdir(self.img_path))
        self.mask_filenames_temp = sorted(os.listdir(self.mask_path))

        self.image_filenames = [
            os.path.join(self.img_path, file_name)
            for file_name in self.image_filenames_temp
        ]
        self.mask_filenames = [
            os.path.join(self.mask_path, file_name)
            for file_name in self.mask_filenames_temp
        ]

        if args.dataset_split == "20" and is_train:
            self.image_filenames = self.image_filenames[:68]
            self.mask_filenames = self.mask_filenames[:68]

        if args.dataset_split == "10" and is_train:
            self.image_filenames = self.image_filenames[:345]
            self.mask_filenames = self.mask_filenames[:345]

        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.s, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
                # transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_test = transforms.Compose(
            [
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
                # transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_val = transforms.Compose(
            [
                transforms.Resize(self.s),
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        mask_path = self.mask_filenames[index]
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        # mask_array = np.array(mask)
        # # mask_array = rgb2mask(mask_array)

        # print(np.unique(mask_array, return_counts=True))
        # color_list = ["white", "red", "yellow", "blue", "violet", "green", "black"]
        # print([color_list[i] for i in np.unique(mask_array)])
        # cmap = matplotlib.colors.ListedColormap(
        #     [color_list[i] for i in np.unique(mask_array)]
        # )
        # f, axarr = plt.subplots(2)
        # axarr[0].imshow(mask_array, cmap=cmap, interpolation="none")
        # axarr[1].imshow(image, interpolation="none")
        # # mask = self.transforms_test(mask)

        # plt.savefig("img.png", dpi=1200)
        # plt.close()

        if self.is_train:
            image, mask = self.transforms_train(image, mask)
            image = self.transforms_distort(image)
            # image, mask = self.transforms_test(image, mask)
        else:
            image, mask = self.transforms_val(image, mask)
            # image, mask = self.transforms_test(image, mask)

        mask = (mask * 256).to(torch.int64)

        # print(mask.unique())
        return image, mask


class LoveDADataset(SatelliteDataset):
    def __init__(self, data_paths, is_train):
        super().__init__(in_c=3)
        self.data_paths = data_paths
        self.is_train = is_train
        self.lookup_table = {
            0.0039: 0,
            0.0078: 1,
            0.0118: 2,
            0.0157: 3,
            0.0196: 4,
            0.0235: 5,
            0.0275: 6,
        }
        self.image_filenames = []  # List to store image file names
        if self.is_train:
            self.mask_filenames = []  # List to store mask file names

        # Load image and mask file names
        self.images_dir_1 = os.path.join(self.data_paths[0], "images_png")
        if self.is_train:
            self.masks_dir_1 = os.path.join(self.data_paths[0], "masks_png")

        self.image_filenames_temp = sorted(os.listdir(self.images_dir_1))
        if self.is_train:
            self.mask_filenames_temp = sorted(os.listdir(self.masks_dir_1))

        self.image_filenames = [
            os.path.join(self.images_dir_1, file_name)
            for file_name in self.image_filenames_temp
        ]
        if self.is_train:
            self.mask_filenames = [
                os.path.join(self.masks_dir_1, file_name)
                for file_name in self.mask_filenames_temp
            ]

        self.images_dir_2 = os.path.join(self.data_paths[1], "images_png")
        if self.is_train:
            self.masks_dir_2 = os.path.join(self.data_paths[1], "masks_png")

        self.image_filenames_temp = sorted(os.listdir(self.images_dir_2))
        if self.is_train:
            self.mask_filenames_temp = sorted(os.listdir(self.masks_dir_2))

        self.image_filenames += [
            os.path.join(self.images_dir_2, file_name)
            for file_name in self.image_filenames_temp
        ]
        if self.is_train:
            self.mask_filenames += [
                os.path.join(self.masks_dir_2, file_name)
                for file_name in self.mask_filenames_temp
            ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        mask_path = self.mask_filenames[idx]
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.is_train:
            self.data_transforms_all = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(15),  # type: ignore
                    transforms.Resize((224, 224)),
                    transforms.Compose(
                        [
                            transforms.ToImage(),
                            transforms.ToDtype(torch.float32, scale=True),
                        ]
                    ),
                ]
            )
            self.data_transforms_img = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
            # f, axarr = plt.subplots(3)
            # axarr[0].imshow(mask)
            # axarr[1].imshow(image)
            image, mask = self.data_transforms_all(image, mask)
            # axarr[2].imshow(mask.squeeze(0))
            # print(mask.unique())
            # plt.savefig(
            #     "satmae_experiments/loveda_results/images/dinov2_b_linear/img.png"
            # )
            # plt.close()
            mask = (mask * 256).to(torch.int64) - 1
            mask[mask == -1] = 0
            # print(mask.unique())
            image = self.data_transforms_img(image)
        else:
            self.data_transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Compose(
                        [
                            transforms.ToImage(),
                            transforms.ToDtype(torch.float32, scale=True),
                        ]
                    ),
                ]
            )
            image, mask = self.data_transforms(image, mask)
            mask = (mask * 256).to(torch.int64) - 1
            mask[mask == -1] = 0
        # print(mask.unique())
        # image = np.array(image)
        # mask = torch.from_numpy(mask.astype("int64"))
        # image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        # return (torch.from_numpy(image), torch.from_numpy(mask))
        return image, mask


class CustomDatasetFromImages(SatelliteDataset):
    # resics
    # mean = [0.368, 0.381, 0.3436]
    # std = [0.2035, 0.1854, 0.1849]
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        torch.set_num_threads(1)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        # self.totensor = transforms.ToTensor()
        # self.scale = transforms.Resize((224, 224))

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)  # type: ignore
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        # single_image_label = self.label_arr[index]
        # cv2.imwrite(
        #     os.path.join("/home/filip/satmae_experiments", "object_result.png"),
        #     (255 * self.scale(self.totensor(img_as_img)))
        #     .to(torch.uint8)
        #     .cpu()
        #     .detach()
        #     .numpy()
        #     .transpose(),
        # )

        return (
            img_as_tensor,
            0,
            # self.scale(self.totensor(img_as_img)),
        )

    def __len__(self):
        return self.data_len


class FMoWTemporalStacked(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path: str, transform: Any):
        """
        Creates Dataset for temporal RGB image classification. Stacks images along temporal dim.
        Usually used for fMoW-RGB-temporal dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion
        """
        super().__init__(in_c=9)
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

        self.min_year = 2002

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]

        splt = single_image_name_1.rsplit("/", 1)  # type: ignore
        base_path = splt[0]
        fname = splt[1]
        suffix = fname[-15:]
        prefix = fname[:-15].rsplit("_", 1)
        regexp = "{}/{}_*{}".format(base_path, prefix[0], suffix)
        temporal_files = glob(regexp)
        temporal_files.remove(single_image_name_1)  # type: ignore
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)  # type: ignore
        img_as_tensor_1 = self.transforms(img_as_img_1)  # (3, h, w)

        img_as_img_2 = Image.open(single_image_name_2)  # type: ignore
        img_as_tensor_2 = self.transforms(img_as_img_2)  # (3, h, w)

        img_as_img_3 = Image.open(single_image_name_3)  # type: ignore
        img_as_tensor_3 = self.transforms(img_as_img_3)  # (3, h, w)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        img = torch.cat(
            (img_as_tensor_1, img_as_tensor_2, img_as_tensor_3), dim=0
        )  # (9, h, w)
        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str, base_resolution=1.0):
        """
        Creates temporal dataset for fMoW RGB
        :param csv_path: Path to csv file containing paths to images
        :param meta_csv_path: Path to csv metadata file for each image
        """
        super().__init__(in_c=3)

        # Transforms
        # self.transforms = transforms
        self.base_resolution = base_resolution
        self.transforms_1 = transforms.Compose(
            [
                # transforms.Scale(224),
                transforms.RandomCrop(224),
            ]
        )
        self.transforms_2 = transforms.Compose(
            [
                # transforms.Scale(224),
                transforms.RandomCrop(160),
            ]
        )
        self.transforms_3 = transforms.Compose(
            [
                # transforms.Scale(224),
                transforms.RandomCrop(112),
            ]
        )

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # temp_list = []
        # for i, row in enumerate(self.image_arr):
        #     if i%10000 == 0:
        #         print(i)
        #     row_list = row.split('/')
        #     name_split = re.sub(r'[0-9]', '*', row_list[2])
        #     row_list.insert(3, name_split)
        #     row2 = '/'.join(row_list)
        #     row2 = os.path.join(os.path.dirname(csv_path), row2)
        #     nekaj = glob(row2)
        #     temp_list.append(nekaj[0])
        # self.image_arr = np.array(temp_list)
        # print(self.image_arr)
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        self.name2index = dict(
            zip([x for x in self.image_arr], np.arange(self.data_len))
        )
        # print(self.name2index)

        self.min_year = 2002  # hard-coded for fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.min_res = 200
        self.max_res = 16300
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        self.scale = transforms.Resize((224, 224))
        self.scale_1 = transforms.Resize((64, 64))
        self.scale_2 = transforms.Resize((224, 224))
        self.scale_3 = transforms.Resize((224, 224))
        # self.scale_2 = transforms.Resize(160)
        # self.scale_3 = transforms.Resize(112)
        self.transforms_train_0 = K.Resize((224, 224))
        self.transforms_train_1 = K.Resize((160, 160))
        self.transforms_train_2 = K.Resize((112, 112))
        # self.transforms_train_1 = K.Resize((160, 160))
        # self.transforms_train_2 = K.Resize((112, 112))

    def __getitem__(self, index):

        # combine this with all else
        ########################################################################
        # imgs = torch.stack(list(zip(*samples))[0])
        # imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        # res = ratios * self.base_resolution
        # imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        ########################################################################

        # Get image name from the pandas df
        single_image_name_1 = self.image_arr[index]
        # print(single_image_name_1)
        # print(self.image_arr)

        suffix = single_image_name_1[-8:]
        prefix = single_image_name_1[:-8].rsplit("_", 1)  # type: ignore
        # print(prefix, suffix)
        regexp = "{}_*{}".format(prefix[0], suffix)
        # regexp = os.path.join(self.dataset_root_path, regexp)
        # print(regexp)
        # single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)
        temporal_files = glob(regexp)
        # print(temporal_files)
        # print('image ' + str(index) +  single_image_name_1)
        # print(regexp)
        # print(len(temporal_files))
        # print(single_image_name_1)
        temporal_files.remove(single_image_name_1)  # type: ignore
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break
        # print(single_image_name_1, single_image_name_2, single_image_name_3)
        img_as_img_1 = Image.open(single_image_name_1)  # type: ignore
        img_as_img_2 = Image.open(single_image_name_2)  # type: ignore
        # img_as_img_3 = Image.open(single_image_name_3)
        img_scale_1 = (img_as_img_1.size[0] + img_as_img_1.size[1]) / 1800
        img_scale_2 = (img_as_img_2.size[0] + img_as_img_2.size[1]) / 1800
        # img_scale_3 = (img_as_img_3.size[0] + img_as_img_3.size[1]) / 1800
        # if img_as_img_1.size[0] > 2000:
        #     print(single_image_name_1)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        # img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        # del img_as_img_3

        img_as_tensor_1 = self.scale_1(img_as_tensor_1)
        img_as_tensor_2 = self.scale_3(img_as_tensor_2)
        # img_as_tensor_3 = self.scale_3(img_as_tensor_3)

        img_as_tensor_1 = self.scale_3(img_as_tensor_1)
        # img_as_tensor_3 = self.scale_1(img_as_tensor_3)
        # if img_as_tensor_3.size(dim=2) > 224:
        #     print(single_image_name_3)
        # if img_as_tensor_2.size(dim=2) > 224:
        #     print(single_image_name_2)
        # if img_as_tensor_1.size(dim=2) > 224:
        #     print(single_image_name_1)

        # img_as_tensor = torch.cat(
        #     [img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=-3
        # )
        img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_2], dim=-3)
        # try:
        #     if (
        #         img_as_tensor_1.shape[2] > 224
        #         and img_as_tensor_2.shape[2] > 224
        #         and img_as_tensor_3.shape[2] > 224
        #     ):
        #         min_w = min(
        #             img_as_tensor_1.shape[2],
        #             min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]),
        #         )
        #         img_as_tensor = torch.cat(
        #             [
        #                 img_as_tensor_1[..., :min_w],
        #                 img_as_tensor_2[..., :min_w],
        #                 img_as_tensor_3[..., :min_w],
        #             ],
        #             dim=-3,
        #         )
        #     elif (
        #         img_as_tensor_1.shape[1] > 224
        #         and img_as_tensor_2.shape[1] > 224
        #         and img_as_tensor_3.shape[1] > 224
        #     ):
        #         min_w = min(
        #             img_as_tensor_1.shape[1],
        #             min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]),
        #         )
        #         img_as_tensor = torch.cat(
        #             [
        #                 img_as_tensor_1[..., :min_w, :],
        #                 img_as_tensor_2[..., :min_w, :],
        #                 img_as_tensor_3[..., :min_w, :],
        #             ],
        #             dim=-3,
        #         )
        #     else:
        #         img_as_img_1 = Image.open(single_image_name_1)
        #         img_as_tensor_1 = self.totensor(img_as_img_1)
        #         img_as_tensor_1 = self.scale(img_as_tensor_1)
        #         img_as_tensor = torch.cat(
        #             [img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3
        #         )
        # except:
        #     print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
        #     assert False

        del img_as_tensor_1
        del img_as_tensor_2
        # del img_as_tensor_3

        # img_as_tensor, imgs_src, ratios, _, _ = self.transforms(img_as_tensor)
        # res = ratios * self.base_resolution
        img_as_tensor = self.transforms_1(img_as_tensor)
        # img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(
        #     img_as_tensor, 3, dim=-3
        # )
        img_as_tensor_1, img_as_tensor_2 = torch.chunk(img_as_tensor, 2, dim=-3)
        del img_as_tensor

        # img_as_tensor_1 = self.transforms_1(img_as_tensor_1)
        # img_as_tensor_2 = self.transforms_2(img_as_tensor_2)
        # img_as_tensor_3 = self.transforms_3(img_as_tensor_3)

        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        # img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        # ts3 = self.parse_timestamp(single_image_name_3)

        # ts = np.stack([ts1, ts2, ts3], axis=0)
        ts = np.stack([ts1, ts2], axis=0)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        # imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        # del img_as_tensor_1
        # del img_as_tensor_2
        # del img_as_tensor_3

        return (
            (img_as_tensor_1, img_as_tensor_2),
            (img_scale_1, img_scale_2),
            ts,
            single_image_label,
        )
        # return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        # print(name)
        # name2 = name.split('/')
        # del name2[8]
        # name = '/'.join(name2)
        # print(name)
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):

        return self.data_len


#########################################################
# SENTINEL DEFINITIONS
#########################################################


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ["value", "one-hot"]
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(
        self,
        csv_path: str,
        transform: Any,
        years: Optional[List[int]] = [*range(2000, 2021)],
        categories: Optional[List[str]] = None,
        label_type: str = "value",
        masked_bands: Optional[List[int]] = None,
        dropped_bands: Optional[List[int]] = None,
    ):
        """
        Creates dataset for multi-spectral single image classification.
        Usually used for fMoW-Sentinel dataset.
        :param csv_path: path to csv file.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param years: List of years to take images from, None to not filter
        :param categories: List of categories to take images from, None to not filter
        :param label_type: 'values' for single label, 'one-hot' for one hot labels
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path).sort_values(
            ["category", "location_id", "timestamp"]
        )

        # Filter by category
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Filter by year
        if years is not None:
            self.df["year"] = [
                int(timestamp.split("-")[0]) for timestamp in self.df["timestamp"]
            ]
            self.df = self.df[self.df["year"].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(
                f"FMOWDataset label_type {label_type} not allowed. Label_type must be one of the following:",
                ", ".join(self.label_types),
            )
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)  # type: ignore

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            # img = data.read(
            #     out_shape=(data.count, self.resize, self.resize),
            #     resampling=Resampling.bilinear
            # )
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        """
        Gets image (x,y) pair given index in dataset.
        :param idx: Index of (image, label) pair in dataset dataframe. (c, h, w)
        :return: Torch Tensor image, and integer label as a tuple.
        """
        selection = self.df.iloc[idx]

        # images = [torch.FloatTensor(rasterio.open(img_path).read()) for img_path in image_paths]
        images = self.open_image(selection["image_path"])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection["category"])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [
                i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands
            ]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            "images": images,
            "labels": labels,
            "image_ids": selection["image_id"],
            "timestamps": selection["timestamp"],
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(
                SentinelNormalize(mean, std)
            )  # use specific Sentinel normalization to avoid NaN
            t.append(
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                )
            )
            t.append(
                transforms.RandomResizedCrop(
                    input_size, scale=(0.2, 1.0), interpolation=interpol_mode
                ),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(
            transforms.Compose(
                [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
            )
        )
        t.append(
            transforms.Resize(
                size, interpolation=interpol_mode
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class EuroSat(SatelliteDataset):
    mean = [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        14.77112979,
        1732.16362238,
        1247.91870117,
    ]
    std = [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        14.3114637,
        1310.36996126,
        1087.6020813,
    ]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Creates dataset for multi-spectral single image classification for EuroSAT.
        :param file_path: path to txt file containing paths to image data for EuroSAT.
        :param transform: pytorch Transform for transforms and tensor conversion
        :param masked_bands: List of indices corresponding to which bands to mask out
        :param dropped_bands:  List of indices corresponding to which bands to drop from input image tensor
        """
        super().__init__(13)
        with open(file_path, "r") as f:
            data = f.read().splitlines()
        self.img_paths = [row.split(",")[1] for row in data]
        self.labels = [int(row.split(",")[0]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)  # type: ignore

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        with rasterio.open(img_path) as data:
            img = data.read()  # (c, h, w)

        return img.transpose(1, 2, 0).astype(np.float32)  # (h, w, c)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [
                i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands
            ]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == "rgb":
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = CustomDatasetFromImages(csv_path, transform)
    elif args.dataset_type == "rgb_scale":
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transforms_init = transforms.Compose(
            [
                transforms.RandomCrop(args.input_size * 2, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.Compose(
                    [
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                    ]
                ),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = CustomDatasetFromImages(csv_path, transforms_init)
    elif args.dataset_type == "temporal":
        dataset = CustomDatasetFromImagesTemporal(csv_path, args.base_resolution)
    elif args.dataset_type == "sentinel":
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = SentinelIndividualImageDataset(
            csv_path,
            transform,
            masked_bands=args.masked_bands,
            dropped_bands=args.dropped_bands,
        )
    elif args.dataset_type == "rgb_temporal_stacked":
        mean = FMoWTemporalStacked.mean
        std = FMoWTemporalStacked.std
        transform = FMoWTemporalStacked.build_transform(
            is_train, args.input_size, mean, std
        )
        dataset = FMoWTemporalStacked(csv_path, transform)
    elif args.dataset_type == "euro_sat":
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(
            csv_path,
            transform,
            masked_bands=args.masked_bands,
            dropped_bands=args.dropped_bands,
        )
    elif args.dataset_type == "spacenet":
        # DataFolder = "/storage/local/ssd/filipwolf-workspace/SpaceNetV1/"
        DataFolder = "/home/filip/SpaceNetV1/"
        raster = DataFolder + "3band/"
        mask = DataFolder + "mask/"

        raster_list = os.listdir(raster)
        raster_list.sort()
        mask_list = os.listdir(mask)
        mask_list.sort()
        r = 0.7

        if is_train:
            # train_raster_list = raster_list[: int(0.1 * len(raster_list))]
            # train_mask_list = mask_list[: int(0.1 * len(mask_list))]
            # train_raster_list = raster_list[: int(r * len(raster_list))]
            # train_mask_list = mask_list[: int(r * len(mask_list))]
            if args.dataset_split == "100":
                train_raster_list = raster_list[:4999]
                train_mask_list = mask_list[:4999]
                dataset = SpaceNetDataset(
                    raster, mask, train_raster_list, train_mask_list, is_train
                )
            elif args.dataset_split == "10":
                train_raster_list = raster_list[:499]
                train_mask_list = mask_list[:499]
                dataset = SpaceNetDataset(
                    raster, mask, train_raster_list, train_mask_list, is_train
                )
        else:
            val_raster_list = raster_list[4999:]
            val_mask_list = mask_list[4999:]
            dataset = SpaceNetDataset(
                raster, mask, val_raster_list, val_mask_list, is_train
            )
    elif args.dataset_type == "loveda":
        if is_train:
            data_paths_rural = "/home/filip/LoveDA/Train/Train/Rural"
            data_paths_urban = "/home/filip/LoveDA/Train/Train/Urban"
            dataset = LoveDADataset((data_paths_rural, data_paths_urban), is_train)
        else:
            data_paths_rural = "/home/filip/LoveDA/Test/Test/Rural"
            data_paths_urban = "/home/filip/LoveDA/Test/Test/Urban"
            dataset = LoveDADataset((data_paths_rural, data_paths_urban), is_train)
    elif args.dataset_type == "vaihingen":
        if is_train:
            data_paths_imgs_train = "/home/filip/vaihingen_dataset/img_dir/train"
            data_paths_ann_train = "/home/filip/vaihingen_dataset/ann_dir/train"
            dataset = VaihingenPotsdamDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
        else:
            data_paths_imgs_train = "/home/filip/vaihingen_dataset/img_dir/val"
            data_paths_ann_train = "/home/filip/vaihingen_dataset/ann_dir/val"
            dataset = VaihingenPotsdamDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
    elif args.dataset_type == "potsdam":
        if is_train:
            data_paths_imgs_train = "/home/filip/potsdam_dataset/img_dir/train"
            data_paths_ann_train = "/home/filip/potsdam_dataset/ann_dir/train"
            dataset = VaihingenPotsdamDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
        else:
            data_paths_imgs_train = "/home/filip/potsdam_dataset/img_dir/val"
            data_paths_ann_train = "/home/filip/potsdam_dataset/ann_dir/val"
            dataset = VaihingenPotsdamDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    return dataset  # type: ignore
