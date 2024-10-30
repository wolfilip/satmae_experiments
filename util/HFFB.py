from torch import nn


class HFFB(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim // 2, 3, padding=1, groups=hidden_dim // 2
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1, padding=0),
        )
        self.residual = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        return self.convs(x) + self.residual(x)
