import os
import random
import warnings
from glob import glob
from typing import Any, List, Optional

import kornia.augmentation as K
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio as rio
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from PIL import Image
from rasterio import logging
from rasterio.enums import Resampling
from torch.utils.data import Dataset

import geopandas
from tqdm import tqdm
import json

from engine_finetune import sentinel2_l2a_to_rgb

log = logging.getLogger()
log.setLevel(logging.ERROR)

warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


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


class iSAIDDataset(SatelliteDataset):
    def __init__(self, img_path, mask_path, is_train, args):
        super().__init__(in_c=3)

        self.img_path = img_path
        self.mask_path = mask_path
        self.s = 896
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

        if args.dataset_split == "10" and is_train:
            self.image_filenames = self.image_filenames[:3398]
            self.mask_filenames = self.mask_filenames[:3398]

        self.transforms_train = K.AugmentationSequential(
            K.RandomResizedCrop(size=(self.s, self.s), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.transforms_val = K.AugmentationSequential(
            data_keys=["input", "mask"],
        )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_bla = transforms.Compose(
            [
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

        # print(len(self.mask_filenames))

        # print(image_path, mask_path)
        # Load image and mask
        image = F.pil_to_tensor(Image.open(image_path).convert("RGB")) / 255
        mask = F.pil_to_tensor(Image.open(mask_path))
        # if 255 in mask:
        #     print(image_path, mask_path)
        # print(np.unique(mask, return_counts=True))
        mask[mask == 255] = 0
        mask = mask.float()
        # mask_array = np.array(mask)
        # # mask_array = rgb2mask(mask_array)

        # print(np.unique(mask, return_counts=True))
        # # if 255 in np.unique(mask):
        # #     print("bla")
        # # color_list = ["white", "red", "yellow", "blue", "violet", "green", "black"]
        # # print([color_list[i] for i in np.unique(mask_array)])
        # # cmap = matplotlib.colors.ListedColormap(
        # #     [color_list[i] for i in np.unique(mask_array)]
        # # )
        # _, axarr = plt.subplots(2)
        # axarr[0].imshow(mask.permute(1, 2, 0), interpolation="none")
        # axarr[1].imshow(image.permute(1, 2, 0), interpolation="none")
        # # mask = self.transforms_test(mask)

        # plt.savefig("img.png", dpi=600)
        # plt.close()

        # mask = F.pil_to_tensor(mask).float()

        if self.is_train:
            image, mask = self.transforms_train(image, mask.unsqueeze(0))
            image = self.transforms_distort(image)
            # image, mask = self.transforms_test(image, mask)
        else:
            image, mask = self.transforms_val(image, mask.unsqueeze(0))
            # image, mask = self.transforms_test(image, mask)

        # mask = (mask * 256).to(torch.int64)

        return image.squeeze(0), mask.squeeze(0).squeeze(0).long()


class MassachusettsRoadsDataset(SatelliteDataset):
    def __init__(self, img_path, mask_path, is_train, args):
        super().__init__(in_c=3)

        self.img_path = img_path
        self.mask_path = mask_path
        self.s = 1500
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

        # if args.dataset_split == "20" and is_train:
        #     self.image_filenames = self.image_filenames[:68]
        #     self.mask_filenames = self.mask_filenames[:68]

        # if args.dataset_split == "10" and is_train:
        #     self.image_filenames = self.image_filenames[:3398]
        #     self.mask_filenames = self.mask_filenames[:3398]

        self.transforms_train = K.AugmentationSequential(
            K.RandomResizedCrop(size=(self.s, self.s), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.transforms_val = K.AugmentationSequential(
            data_keys=["input", "mask"],
        )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

        self.transforms_bla = transforms.Compose(
            [
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

        # print(len(self.mask_filenames))

        # print(image_path, mask_path)
        # Load image and mask
        image = F.pil_to_tensor(Image.open(image_path).convert("RGB")) / 255
        mask = F.pil_to_tensor(Image.open(mask_path)).float() / 255
        # if 255 in mask:
        #     print(image_path, mask_path)
        # print(np.unique(mask, return_counts=True))
        # mask[mask == 255] = 0
        # mask = mask.float()
        # mask_array = np.array(mask)
        # # mask_array = rgb2mask(mask_array)

        # print(np.unique(mask, return_counts=True))
        # # if 255 in np.unique(mask):
        # #     print("bla")
        # # color_list = ["white", "red", "yellow", "blue", "violet", "green", "black"]
        # # print([color_list[i] for i in np.unique(mask_array)])
        # # cmap = matplotlib.colors.ListedColormap(
        # #     [color_list[i] for i in np.unique(mask_array)]
        # # )
        # _, axarr = plt.subplots(2)
        # axarr[0].imshow(mask.permute(1, 2, 0), interpolation="none")
        # axarr[1].imshow(image.permute(1, 2, 0), interpolation="none")
        # # mask = self.transforms_test(mask)

        # plt.savefig("img.png", dpi=600)
        # plt.close()

        # mask = F.pil_to_tensor(mask).float()

        if self.is_train:
            image, mask = self.transforms_train(image, mask.unsqueeze(0))
            image = self.transforms_distort(image)
            # image, mask = self.transforms_test(image, mask)
        else:
            image, mask = self.transforms_val(image, mask.unsqueeze(0))
            # image, mask = self.transforms_test(image, mask)

        # mask = (mask * 256).to(torch.int64)

        return image.squeeze(0), mask.squeeze(0).squeeze(0).long()


class SpaceNetDataset(SatelliteDataset):
    def __init__(
        self,
        raster_rgb,
        raster_depth,
        mask,
        raster_list_rgb,
        raster_list_depth,
        mask_list,
        is_train,
        args,
    ):
        super().__init__(in_c=3)
        self.raster_rgb = raster_rgb
        self.raster_depth = raster_depth
        self.mask = mask
        self.raster_list_rgb = raster_list_rgb
        self.raster_list_depth = raster_list_depth
        self.mask_list = mask_list
        self.s = args.input_size

        self.is_train = is_train

        self.transforms_train = K.AugmentationSequential(
            K.RandomResizedCrop(size=(self.s, self.s), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.transforms_val = K.AugmentationSequential(
            data_keys=["input", "mask"],
        )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

    def __len__(self):
        return len(self.raster_list_rgb)

    def __getitem__(self, index):
        img_rgb = (
            rio.open(self.raster_rgb + self.raster_list_rgb[index]).read(
                out_shape=(self.s, self.s), resampling=Resampling.bilinear
            )
            / 255
        )
        # img_rgb = Image.open(self.raster_rgb + self.raster_list_rgb[index]).convert(
        #     "RGB"
        # )

        # print(self.raster_list_depth[index], self.raster_list_rgb[index])
        # img_depth = Image.open(self.raster_depth + self.raster_list_depth[index])
        # name = self.raster_list[index][:-3]
        # print("SpaceNetV1/imgs/" + name + "png")
        mask = (
            rio.open(self.mask + self.mask_list[index])
            .read(out_shape=(self.s, self.s), resampling=Resampling.bilinear)
            .squeeze()
        )
        img_rgb = torch.from_numpy(img_rgb.astype("float32"))
        # same images till here
        # save_image(img, "../SpaceNetV1/imgs/" + self.raster_list[index][:-3] + "png")
        mask = (
            torch.from_numpy(mask.astype("float32")).unsqueeze(0).unsqueeze(0).float()
        )
        if self.is_train:
            # img_rgb, img_depth, mask = self.transforms_train(img_rgb, img_depth, mask)
            # img_rgb, img_depth = self.transforms_distort(img_rgb, img_depth)
            img_rgb, mask = self.transforms_train(img_rgb, mask)
            img_rgb = self.transforms_distort(img_rgb)
            # img_rgb, img_depth, mask = self.transforms_val(img_rgb, img_depth, mask)

            # f, axarr = plt.subplots(2)
            # axarr[0].imshow(img_rgb.permute(1, 2, 0), interpolation="none")
            # axarr[1].imshow(img_depth.permute(1, 2, 0), interpolation="none")

            # plt.savefig("img.png", dpi=600)
            # plt.close()
        else:
            img_rgb, mask = self.transforms_val(img_rgb, mask)
            # f, axarr = plt.subplots(2)
            # axarr[0].imshow(img_rgb.permute(1, 2, 0), interpolation="none")
            # axarr[1].imshow(img_depth.permute(1, 2, 0), interpolation="none")

            # plt.savefig("img.png", dpi=600)
            # plt.close()
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
        return img_rgb.squeeze(0), mask.squeeze(0).squeeze(0).long()


class Sen1Floods11Dataset(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.0,
        norm_path="/home/filip/sen1floods11/v1.1/norms",
        ignore_index: int = -1,
        num_classes: int = 2,
    ):
        """Initialize the Sen1Floods11 dataset.
        Link: https://github.com/cloudtostreet/Sen1Floods11

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use.
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            partition (float, optional): Partition of the dataset to use. Defaults to 1.0.
            norm_path (str, optional): Path for normalization data. Defaults to None.
            ignore_index (int, optional): Index to ignore for metrics and loss. Defaults to -1.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.num_classes = num_classes

        self.ignore_index = ignore_index

        self.split_mapping = {"train": "train", "val": "valid", "test": "test"}

        metadata_file = os.path.join(self.path, "v1.1", "Sen1Floods11_Metadata.geojson")
        self.metadata = geopandas.read_file(metadata_file)

        # hand labeled
        split_file = os.path.join(
            self.path,
            "v1.1",
            f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv",
        )
        data_root = os.path.join(self.path, "v1.1", "data/flood_events/HandLabeled")

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        # self.s1_image_list = [
        #     os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        # ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[1]) for f in file_list
        ]

        # # weakly labeled
        # split_file_2 = os.path.join(
        #     self.path,
        #     "v1.1",
        #     f"splits/flood_alllabels/flood_{self.split_mapping[split]}_data.csv",
        # )
        # data_root_2 = os.path.join(self.path, "v1.1", "data/flood_events/WeaklyLabeled")

        # with open(split_file_2) as f:
        #     file_list_2 = f.readlines()

        # file_list_2 = [f.rstrip().split(",") for f in file_list_2]

        # # self.s1_image_list = [
        # #     os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        # # ]
        # self.s2_image_list_2 = [
        #     os.path.join(data_root_2, "S2IndexLabelWeak", f[0]) for f in file_list_2
        # ]
        # self.target_list_2 = [
        #     os.path.join(
        #         data_root_2,
        #         "S1OtsuLabelWeak",
        #         f[0].replace("S2IndexLabelWeak", "S1OtsuLabelWeak"),
        #     )
        #     for f in file_list_2
        # ]

        # self.s2_image_list = self.s2_image_list_1 + self.s2_image_list_2
        # self.target_list = self.target_list_1 + self.target_list_2

        self.collate_fn = collate_fn
        self.norm = None
        if norm_path is not None:
            norm = {}
            for modality in self.modalities:
                file_path = os.path.join(
                    norm_path, "NORM_{}_patch.json".format(modality)
                )
                if not (os.path.exists(file_path)):
                    self.compute_norm_vals(norm_path, modality)
                normvals = json.load(open(file_path))
                norm[modality] = (
                    torch.tensor(normvals["mean"]).float(),
                    torch.tensor(normvals["std"]).float(),
                )
            self.norm = norm

        if partition < 1:
            indices, _ = self.balance_seg_indices(
                strategy="stratified", label_fraction=partition, num_bins=3
            )
            self.s1_image_list = [self.s1_image_list[i] for i in indices]
            self.s2_image_list = [self.s2_image_list[i] for i in indices]
            self.target_list = [self.target_list[i] for i in indices]

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

    def __len__(self):
        return len(self.s2_image_list)

    def compute_norm_vals(self, folder, sat):
        means = []
        stds = []
        for i, b in enumerate(self.s2_image_list):
            data = self.__getitem__(i)[0]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            s2_date = pd.to_datetime("01-01-1998", dayfirst=True)
            s1_date = pd.to_datetime("01-01-1998", dayfirst=True)
        else:
            s2_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s2_date"].item()
            )
            s1_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s1_date"].item()
            )
        return torch.tensor([s2_date.dayofyear]), torch.tensor([s1_date.dayofyear])

    def __getitem__(self, index):
        with rio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        # with rasterio.open(self.s1_image_list[index]) as src:
        #     s1_image = src.read()
        #     # Convert the missing values (clouds etc.)
        #     s1_image = np.nan_to_num(s1_image)

        with rio.open(self.target_list[index]) as src:
            target = src.read(1)

        # timestamp = self._get_date(index)

        # s2_image_rgb = torch.from_numpy(s2_image).float()[[3, 2, 7]]
        s2_image = torch.from_numpy(s2_image).float()[[3, 2, 1, 4, 5, 6, 7, 8, 11, 12]]
        # s1_image = torch.from_numpy(s1_image).float()
        # ratio_band = s1_image[:1, :, :] / (s1_image[1:, :, :] + 1e-10)
        # ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
        # s1_image = torch.cat((s1_image[:2, :, :], ratio_band), dim=0)
        target = torch.from_numpy(target).long()

        output = {
            # "s2": s2_image_rgb,
            "s2": s2_image,
            # "s2_rest": s2_image_rest,
            # "s1": s1_image.unsqueeze(0),
            "label": target,
            # "s2_dates": timestamp[0],
            # "s1_dates": timestamp[1],
        }

        output["s2"], output["label"] = self.transform(
            output["s2"],
            # output["s2_rest"],
            output["label"].unsqueeze(0).unsqueeze(0).float(),
        )

        # f, axarr = plt.subplots(2)
        # axarr[0].imshow(
        #     sentinel2_l2a_to_rgb(s2_image.squeeze(0)).permute(1, 2, 0),
        #     interpolation="none",
        # )
        # axarr[1].imshow(target.squeeze(0).squeeze(0), interpolation="none")

        # plt.savefig("img.png", dpi=600)
        # plt.close()

        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (
                        output[modality] - self.norm[modality][0][None, :, None, None]
                    ) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (
                        output[modality] - self.norm[modality][0][:, None, None]
                    ) / self.norm[modality][1][:, None, None]

        # print(torch.min(output["s2"]), torch.max(output["s2"]))

        # if self.split == "train":
        #     output["s2"] = self.transforms_distort(output["s2"].clamp(min=0, max=1))

        return (
            output["s2"].squeeze(0),
            # output["s2"].squeeze(0)[4, 5, 6, 7, 8, 11, 12],
            # torch.from_numpy(s2_image).float()[[3, 2, 1]],
            output["label"].squeeze(0).squeeze(0).long() + 1,
        )

    # Calculate image-wise class distributions for segmentation
    def calculate_class_distributions(self):
        num_classes = self.num_classes
        ignore_index = self.ignore_index
        class_distributions = []

        for idx in tqdm(
            range(self.__len__()), desc="Calculating class distributions per sample"
        ):
            target = self[idx]["label"]

            if ignore_index is not None:
                target = target[(target != ignore_index)]

            total_pixels = target.numel()
            if total_pixels == 0:
                class_distributions.append([0] * num_classes)
                continue
            else:
                class_counts = [(target == i).sum().item() for i in range(num_classes)]
                class_ratios = [count / total_pixels for count in class_counts]
                class_distributions.append(class_ratios)

        return np.array(class_distributions)

    # Function to bin class distributions using ceil
    def bin_class_distributions(self, class_distributions, num_bins=3, logger=None):
        bin_edges = np.linspace(0, 1, num_bins + 1)[1]
        binned_distributions = np.ceil(class_distributions / bin_edges).astype(int) - 1
        return binned_distributions

    def balance_seg_indices(
        self, strategy, label_fraction=1.0, num_bins=3, logger=None
    ):
        """
        Balances and selects indices from a segmentation dataset based on the specified strategy.

        Args:
        dataset : GeoFMDataset | GeoFMSubset
            The dataset from which to select indices, typically containing geospatial segmentation data.

        strategy : str
            The strategy to use for selecting indices. Options include:
            - "stratified": Proportionally selects indices from each class bin based on the class distribution.
            - "oversampled": Prioritizes and selects indices from bins with lower class representation.

        label_fraction : float, optional, default=1.0
            The fraction of labels (indices) to select from each class or bin. Values should be between 0 and 1.

        num_bins : int, optional, default=3
            The number of bins to divide the class distributions into, used for stratification or oversampling.

        logger : object, optional
            A logger object for tracking progress or logging messages (e.g., `logging.Logger`)

        ------

        Returns:
        selected_idx : list of int
            The indices of the selected samples based on the strategy and label fraction.

        other_idx : list of int
            The remaining indices that were not selected.

        """
        # Calculate class distributions with progress tracking
        class_distributions = self.calculate_class_distributions()

        # Bin the class distributions
        binned_distributions = self.bin_class_distributions(
            class_distributions, num_bins=num_bins, logger=logger
        )
        combined_bins = np.apply_along_axis(
            lambda row: "".join(map(str, row)), axis=1, arr=binned_distributions
        )

        indices_per_bin = {}
        for idx, bin_id in enumerate(combined_bins):
            if bin_id not in indices_per_bin:
                indices_per_bin[bin_id] = []
            indices_per_bin[bin_id].append(idx)

        if strategy == "stratified":
            # Select a proportion of indices from each bin
            selected_idx = []
            for bin_id, indices in indices_per_bin.items():
                num_to_select = int(
                    max(1, len(indices) * label_fraction)
                )  # Ensure at least one index is selected
                selected_idx.extend(
                    np.random.choice(indices, num_to_select, replace=False)
                )
        elif strategy == "oversampled":
            # Prioritize the bins with the lowest values
            sorted_indices = np.argsort(combined_bins)
            selected_idx = sorted_indices[: int(len(dataset) * label_fraction)]

        # Determine the remaining indices not selected
        other_idx = list(set(range(self.__len__())) - set(selected_idx))

        return selected_idx, other_idx


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

        self.transforms_train = K.AugmentationSequential(
            K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.transforms_val = K.AugmentationSequential(
            data_keys=["input", "mask"],
        )

        # self.transforms_train = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(self.s, scale=(0.5, 1.0)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.Compose(
        #             [
        #                 transforms.ToImage(),
        #                 transforms.ToDtype(torch.float32, scale=True),
        #             ]
        #         ),
        #         # transforms.RandomPhotometricDistort(),
        #     ]
        # )

        # self.transforms_test = transforms.Compose(
        #     [
        #         transforms.Compose(
        #             [
        #                 transforms.ToImage(),
        #                 transforms.ToDtype(torch.float32, scale=True),
        #             ]
        #         ),
        #         # transforms.RandomPhotometricDistort(),
        #     ]
        # )

        self.transforms_distort = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
            ]
        )

        # self.transforms_val = transforms.Compose(
        #     [
        #         transforms.Resize(self.s),
        #         transforms.Compose(
        #             [
        #                 transforms.ToImage(),
        #                 transforms.ToDtype(torch.float32, scale=True),
        #             ]
        #         ),
        #     ]
        # )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        mask_path = self.mask_filenames[index]
        # Load image and mask
        image = F.pil_to_tensor(Image.open(image_path).convert("RGB")) / 255
        mask = mask = F.pil_to_tensor(Image.open(mask_path)).float() / 255
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
            image, mask = self.transforms_train(image, mask.unsqueeze(0))
            image = self.transforms_distort(image)
            # image, mask = self.transforms_test(image, mask)
        else:
            image, mask = self.transforms_val(image, mask.unsqueeze(0))
            # image, mask = self.transforms_test(image, mask)

        mask = (mask * 256).to(torch.int64)

        # print(mask.unique())
        return image.squeeze(0), mask.squeeze(0).squeeze(0).long()


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
    mean = [0.368, 0.381, 0.3436]
    std = [0.2035, 0.1854, 0.1849]
    # mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    # std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

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
        single_image_label = self.label_arr[index]
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
            single_image_label,
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

        # sample = {
        #     "images": images,
        #     "labels": labels,
        #     "image_ids": selection["image_id"],
        #     "timestamps": selection["timestamp"],
        # }
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


def build_fmow_dataset(is_train: bool, data_split, args) -> SatelliteDataset:
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
        raster_rgb = DataFolder + "3band/"
        raster_depth = DataFolder + "depth/"
        mask = DataFolder + "mask/"

        raster_list_rgb = os.listdir(raster_rgb)
        raster_list_rgb.sort()
        raster_list_depth = os.listdir(raster_depth)
        raster_list_depth.sort()
        mask_list = os.listdir(mask)
        mask_list.sort()
        # r = 0.7

        if is_train:
            # train_raster_list = raster_list[: int(0.1 * len(raster_list))]
            # train_mask_list = mask_list[: int(0.1 * len(mask_list))]
            # train_raster_list = raster_list[: int(r * len(raster_list))]
            # train_mask_list = mask_list[: int(r * len(mask_list))]
            if args.dataset_split == "100":
                train_raster_list_rgb = raster_list_rgb[:4999]
                train_raster_list_depth = raster_list_depth[:4999]
                train_mask_list = mask_list[:4999]
                dataset = SpaceNetDataset(
                    raster_rgb,
                    raster_depth,
                    mask,
                    train_raster_list_rgb,
                    train_raster_list_depth,
                    train_mask_list,
                    is_train,
                    args,
                )
            elif args.dataset_split == "10":
                train_raster_list_rgb = raster_list_rgb[:499]
                train_raster_list_depth = raster_list_depth[:499]
                train_mask_list = mask_list[:499]
                dataset = SpaceNetDataset(
                    raster_rgb,
                    raster_depth,
                    mask,
                    train_raster_list_rgb,
                    train_raster_list_depth,
                    train_mask_list,
                    is_train,
                    args,
                )
        else:
            val_raster_list_rgb = raster_list_rgb[4999:]
            val_raster_list_depth = raster_list_depth[4999:]
            val_mask_list = mask_list[4999:]
            dataset = SpaceNetDataset(
                raster_rgb,
                raster_depth,
                mask,
                val_raster_list_rgb,
                val_raster_list_depth,
                val_mask_list,
                is_train,
                args,
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
    elif args.dataset_type == "sen1floods11":
        transforms_train = K.AugmentationSequential(
            K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        transforms_test = K.AugmentationSequential(
            data_keys=["input", "mask"],
        )
        if data_split == "train":
            dataset = Sen1Floods11Dataset(
                "/home/filip/sen1floods11", ["s2"], transforms_train, "train"
            )
        elif data_split == "val":
            dataset = Sen1Floods11Dataset(
                "/home/filip/sen1floods11", ["s2"], transforms_test, "val"
            )
        elif data_split == "test":
            dataset = Sen1Floods11Dataset(
                "/home/filip/sen1floods11", ["s2"], transforms_test, "test"
            )
    elif args.dataset_type == "isaid":

        if is_train:
            data_paths_imgs_train = "/home/filip/iSAID_converted/img_dir/train"
            data_paths_ann_train = "/home/filip/iSAID_converted/ann_dir/train"
            dataset = iSAIDDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
        else:
            data_paths_imgs_train = "/home/filip/iSAID_converted/img_dir/val"
            data_paths_ann_train = "/home/filip/iSAID_converted/ann_dir/val"
            dataset = iSAIDDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
    elif args.dataset_type == "mass_roads":

        if is_train:
            data_paths_imgs_train = "/home/filip/massachusetts_roads/train"
            data_paths_ann_train = "/home/filip/massachusetts_roads/train_labels"
            dataset = MassachusettsRoadsDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
        else:
            data_paths_imgs_train = "/home/filip/massachusetts_roads/test"
            data_paths_ann_train = "/home/filip/massachusetts_roads/test_labels"
            dataset = MassachusettsRoadsDataset(
                data_paths_imgs_train, data_paths_ann_train, is_train, args
            )
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    return dataset  # type: ignore
