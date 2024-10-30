from torch import nn

from util.Norm2d import Norm2d


class FPNHead(nn.Module):
    def __init__(self, embed_dim, share_weights=False) -> None:
        super().__init__()
        self.share_weights = share_weights
        if self.share_weights:
            self.fpn1 = nn.Sequential(
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.do_fpn1 = lambda x: self.fpn1(self.fpn2(x))
        else:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        )

        # self.fpn3 = nn.Identity()

        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        InputL B X C X H X W
        """
        features = []
        if self.share_weights:
            ops = [
                self.do_fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        else:
            ops = [
                self.fpn1,
                self.fpn2,
                # self.fpn3, self.fpn4
            ]
        for i in range(len(ops)):
            features.append(ops[i](x))

        return tuple(features)
