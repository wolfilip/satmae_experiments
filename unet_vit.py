import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip_con = nn.Identity()

    def forward(self, x):
        return self.skip_con(x)


class ResidualBlock(nn.Module):  # TODO: Change this for a basic UNet
    def __init__(
        self,
        in_channels,
        out_channels,
        n_groups=32,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = n_channels // n_heads
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool,
    ):
        super().__init__()

        self.has_attn = has_attn
        self.res = ResidualBlock(in_channels, out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool,
    ):
        super().__init__()
        add = out_channels
        self.has_attn = has_attn
        self.res = ResidualBlock(in_channels + add, out_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, parameters: dict):
        super().__init__()

        n_channels = parameters["n_channels"]
        ch_mults = parameters["ch_mults"]
        is_attn = parameters["is_attn"]
        n_blocks = parameters["n_blocks"]
        image_channels = int(parameters["image_channels"])

        self.start_img_channels = image_channels
        self.n_layers = len(ch_mults)
        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        down = []
        skip_con = []
        self.cond_embs = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        is_attn[i],
                    )
                )
                skip_con.append(SkipConnection())
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
                skip_con.append(SkipConnection())

        self.skip_con = nn.ModuleList(skip_con)
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        is_attn[i],
                    )
                )
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    is_attn[i],
                )
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(
            in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def prepare_start(self, x):
        x = self.image_proj(x)
        return x

    def encode(self, x):
        h = [x]
        for m, sc in zip(self.down, self.skip_con):
            x = m(x)
            h.append(sc(x))
        return x, h

    def middle_block(self, x):
        x = self.middle(x)
        return x

    def decode(self, x, h):
        for m in self.up:
            if not isinstance(m, Upsample):
                s = h.pop()
                x = torch.cat((x, s), dim=1)
            x = m(x)
        return x

    def final_output(self, x):
        c = x.shape[1]
        f = self.act(self.norm(x))
        f = self.final(f)
        return f

    def forward(self, x: torch.Tensor):
        x = self.prepare_start(x)
        x, h = self.encode(x)
        x = self.middle_block(x)
        x = self.decode(x, h)
        return self.final_output(x)


if __name__ == "__main__":
    params = {
        "n_channels": 64,
        "ch_mults": [1, 1],
        "is_attn": [False, False],
        "n_blocks": 1,
        "image_channels": 3,
    }
    ResUNet = UNet(params).cuda()
    x = torch.rand((1, 3, 16, 16)).cuda()
    x = ResUNet(x)
    print(x.shape)
