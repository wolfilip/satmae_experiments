import torch
from einops import rearrange
from torchvision.transforms.functional import resize


def visualize_features(features, is_conv):
    x = features.cpu()

    if is_conv:
        x = torch.permute(x, (0, 2, 3, 1))
        E_patch_norm = rearrange(x, "B L1 L2 E -> (B L1 L2) E").to(torch.float64)
    else:
        E_patch_norm = rearrange(x, "B L E -> (B L) E").to(torch.float64)

    _, _, V = torch.pca_lowrank(E_patch_norm)
    E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])
    E_pca_1_norm = minmax_norm(E_pca_1)
    M_fg = E_pca_1_norm.squeeze() > 0.5

    _, _, V = torch.pca_lowrank(E_patch_norm[M_fg])
    E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V[:, :3])
    E_pca_3_fg = minmax_norm(E_pca_3_fg)

    if is_conv:
        B, L1, L2, _ = x.shape
        Z = B * L1 * L2
    else:
        B, L, _ = x.shape
        Z = B * L

    I_draw = torch.zeros(Z, 3).to(torch.float64)
    I_draw[M_fg] = E_pca_3_fg
    I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)

    if is_conv:
        I_draw = rearrange(
            I_draw,
            "B (h w) C -> B h w C",
            h=144,
            w=144,
        )
    else:
        I_draw = rearrange(
            I_draw,
            "B (h w) C -> B h w C",
            h=512 // 16,
            w=512 // 16,
        )

    image_1_pca = I_draw[0]
    image_2_pca = I_draw[1]
    image_1_pca = rearrange(image_1_pca, "H W C -> C H W")
    image_2_pca = rearrange(image_2_pca, "H W C -> C H W")
    image_1_pca = resize(image_1_pca, [512, 512])
    image_2_pca = resize(image_2_pca, [512, 512])
    return image_1_pca, image_2_pca


def minmax_norm(x):
    """Min-max normalization"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
