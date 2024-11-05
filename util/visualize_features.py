import torch
from einops import rearrange
from torchvision.transforms.functional import resize


def visualize_features(features):
    # print(features.shape)
    x = features.cpu()
    E_patch_norm = rearrange(x, "B L E -> (B L) E").to(torch.float64)
    # print(E_patch_norm.shape)
    _, _, V = torch.pca_lowrank(E_patch_norm)
    # print(V)
    E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])
    # print(E_pca_1.shape)
    E_pca_1_norm = minmax_norm(E_pca_1)
    # print(E_pca_1_norm.shape)
    M_fg = E_pca_1_norm.squeeze() > 0.5
    # print(M_fg.shape)
    _, _, V = torch.pca_lowrank(E_patch_norm[M_fg])
    # print(V)
    E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V[:, :3])
    # print(E_pca_3_fg.shape)
    E_pca_3_fg = minmax_norm(E_pca_3_fg)
    # print(E_pca_3_fg.shape)
    B, L, _ = x.shape
    # print(B, L)
    Z = B * L
    # print(Z)
    I_draw = torch.zeros(Z, 3).to(torch.float64)
    # print(I_draw.shape)
    I_draw[M_fg] = E_pca_3_fg
    I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)
    I_draw = rearrange(
        I_draw,
        "B (h w) C -> B h w C",
        h=224 // 14,
        w=224 // 14,
    )
    image_1_pca = I_draw[0]
    # image_2_pca = I_draw[1]
    image_1_pca = rearrange(image_1_pca, "H W C -> C H W")
    # image_2_pca = rearrange(image_2_pca, "H W C -> C H W")
    image_1_pca = resize(image_1_pca, [224, 224])
    # image_2_pca = resize(image_2_pca, [224, 224])
    return image_1_pca, _


def minmax_norm(x):
    """Min-max normalization"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
