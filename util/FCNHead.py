from torch import nn

from util.HFFB import HFFB
from util.Norm2d import Norm2d


class FCNHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, target_dim) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, hidden_dim, 1)
        convs = []
        for _ in range(num_layers):
            convs.append(HFFB(hidden_dim))
        self.conv_blocks = nn.Sequential(*convs)
        self.pred = nn.Sequential(
            Norm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 4, 3, padding=1, groups=hidden_dim // 4
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 1, padding=0),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, 3, kernel_size=2, stride=2),
        )

    def forward(self, xp):
        """
        InputL List[B X C X H X W], FPN features
        """
        out = []
        for x in xp:
            x = self.proj(x)
            x = self.conv_blocks(x)
            out.append(self.pred(x))

        return out
