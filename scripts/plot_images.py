

import torchvision
from PIL import Image
import torchvision.transforms.functional as F



imgs_paths = F.pil_to_tensor(Image.open(image_path))

grid = torchvision.utils.make_grid(
    tensor=imgs, nrow=3 + len(group["dirs"]), pad_value=255, padding=10
)
np_grid = total_grid.permute(1, 2, 0).numpy()
im = Image.fromarray(np_grid)
im.save("figs/qual_combined.png")