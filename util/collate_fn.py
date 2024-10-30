import torch


class TransformCollateFn:
    def __init__(self, transforms, base_resolution=2.5):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        # paths = list(zip(*samples))[2]
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res), None
