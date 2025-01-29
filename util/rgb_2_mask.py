import numpy as np


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
    _, _, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0], [1], [2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for c in values:
        mask[img_id == c] = color2index[tuple(img[img_id == c][0])]

    return mask.astype(int)
