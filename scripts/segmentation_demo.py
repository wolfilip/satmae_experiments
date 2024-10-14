import os

import rasterio as rio
import torch
import torch.nn as nn
from rasterio.enums import Resampling
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DataFolder = "/home/filip/SpaceNetV1/"
Raster = DataFolder + "3band/"
Vector = DataFolder + "geojson/"
Mask = DataFolder + "mask/"

raster_list = os.listdir(Raster)
raster_list.sort()
mask_list = os.listdir(Mask)
mask_list.sort()

r = 0.7
train_raster_list = raster_list[: int(r * len(raster_list))]
train_mask_list = mask_list[: int(r * len(mask_list))]
val_raster_list = raster_list[int(r * len(raster_list)) :]
val_mask_list = mask_list[int(r * len(mask_list)) :]

s = 224

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpaceNetDataset(Dataset):
    def __init__(self, raster_list, mask_list):
        super().__init__()
        self.raster_list = raster_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.raster_list)

    def __getitem__(self, index):
        img = (
            rio.open(Raster + raster_list[index]).read(
                out_shape=(s, s), resampling=Resampling.bilinear
            )
            / 255
        )
        mask = (
            rio.open(Mask + mask_list[index])
            .read(out_shape=(s, s), resampling=Resampling.bilinear)
            .squeeze()
        )
        img = torch.from_numpy(img.astype("float32"))
        mask = torch.from_numpy(mask.astype("int64"))
        mask = F.one_hot(mask, num_classes=2).permute(2, 0, 1)
        return img, mask


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def get_bce_loss(pred, mask):
    bce = nn.BCELoss()
    m = nn.Sigmoid()
    loss = bce(m(pred), mask)
    return loss


def get_iou(pred, target, nclass):
    pred = torch.sigmoid(pred)
    pred = torch.argmax(pred, 1)
    pred = F.one_hot(pred, num_classes=nclass).permute(0, 3, 1, 2)
    pred = pred.cpu().detach().numpy().astype(int)
    target = target.cpu().detach().numpy().astype(int)
    iou = (pred & target).sum() / ((pred | target).sum() + 1e-6)
    return iou


def train(
    model,
    nclass,
    batch_size,
    epochs,
    train_raster_list,
    train_mask_list,
    val_raster_list,
    val_mask_list,
    device,
    save_path,
):
    model.train()
    log = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, "max")
    val_dataset = SpaceNetDataset(val_raster_list, val_mask_list)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_img = []
    test_mask = []
    test_pred = []
    for idx in [160, 200, 400, 650, 900]:
        img, mask = val_dataset.__getitem__(idx)
        test_img.append(img)
        test_mask.append(mask)
    for epoch in tqdm(range(epochs)):
        train_dataset = SpaceNetDataset(train_raster_list, train_mask_list)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        model.train()
        epoch_loss, epoch_iou = calc_metrics(
            model, train_loader, optimizer, device, True, 0, 0, nclass
        )
        epoch_loss = epoch_loss / (len(train_loader))
        epoch_iou = epoch_iou / (len(train_loader))
        model.eval()
        with torch.no_grad():
            val_loss, val_iou = calc_metrics(
                model, val_loader, optimizer, device, False, 0, 0, nclass
            )
            val_loss = val_loss / len(val_loader)
            val_iou = val_iou / len(val_loader)
            scheduler.step(val_loss)
        print(f"train_loss = {epoch_loss:.3f}, train_iou = {epoch_iou:.3f}")
        print(f"val_loss = {val_loss:.3f}, val_iou = {val_iou:.3f}")
        log["train_loss"].append(epoch_loss)
        log["train_iou"].append(epoch_iou)
        log["val_loss"].append(val_loss)
        log["val_iou"].append(val_iou)
        torch.save(
            model.state_dict(),
            f"{save_path}/BATCH={batch_size}_EPOCH={epoch}_IOU={val_iou:.3f}_LOSS={val_loss:.3f}.pt",
        )
    return log


def calc_metrics(
    model, data_loader, optimizer, device, grad, epoch_loss, epoch_iou, nclass
):
    epoch_loss = epoch_loss
    epoch_iou = epoch_iou
    for data, mask in data_loader:
        mask = mask.to(torch.float32)
        data, mask = data.to(device), mask.to(device)
        pred = model(data)
        loss = get_bce_loss(pred, mask)
        IoU = get_iou(pred, mask, nclass)
        if grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += float(loss.item())
        epoch_iou += float(IoU.item())
    return epoch_loss, epoch_iou


model = UNet(3, 2).to(device)
log = train(
    model,
    2,
    16,
    30,
    train_raster_list,
    train_mask_list,
    val_raster_list,
    val_mask_list,
    device,
    DataFolder + "Weight/",
)
