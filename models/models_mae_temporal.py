# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from torchvision import transforms

from util.pos_embed import (
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed,
)


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed_1 = PatchEmbed(224, patch_size, in_chans, embed_dim)
        self.patch_embed_2 = PatchEmbed(224, patch_size, in_chans, embed_dim)
        self.patch_embed_3 = PatchEmbed(224, patch_size, in_chans, embed_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim - 256), requires_grad=False  # type: ignore
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim - 128),  # type: ignore
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.same_mask = True
        self.initialize_weights()
        self.counter = 0

        self.scale_1 = transforms.Resize((224, 224))
        self.scale_2 = transforms.Resize((160, 160))
        self.scale_3 = transforms.Resize((112, 112))

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),  # type: ignore
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),  # type: ignore
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  # type: ignore
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))  # type: ignore
        return imgs

    def random_masking(self, x, mask_ratio, mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        if self.same_mask:
            L2 = L // 3
            assert 3 * L2 == L
            noise = torch.rand(N, L2, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_shuffle = [ids_shuffle + i * L2 for i in range(3)]
            ids_shuffle_keep = [z[:, : int(L2 * (1 - mask_ratio))] for z in ids_shuffle]
            ids_shuffle_disc = [z[:, int(L2 * (1 - mask_ratio)) :] for z in ids_shuffle]
            ids_shuffle = []
            for z in ids_shuffle_keep:
                ids_shuffle.append(z)
            for z in ids_shuffle_disc:
                ids_shuffle.append(z)
            ids_shuffle = torch.cat(ids_shuffle, dim=1)
            # print(ids_shuffle[0])
            # assert False
        else:
            if mask is None:
                # sort noise for each sample
                ids_shuffle = torch.argsort(
                    noise, dim=1
                )  # ascend: small is keep, large is remove
            else:
                ids_shuffle = mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2(self, x, mask_ratio, mask=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = 68

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        if self.same_mask:
            L2 = L // 2
            assert 2 * L2 == L
            noise_1 = torch.rand(N, L2, device=x.device)  # noise in [0, 1]
            ids_shuffle_1 = torch.argsort(noise_1, dim=1)
            ids_shuffle_1 = [ids_shuffle_1 + i * L2 for i in range(2)]
            ids_shuffle_keep_1 = ids_shuffle_1[0][:, : int(L2 * (1 - 0.75))]
            ids_shuffle_disc_1 = ids_shuffle_1[0][:, int(L2 * (1 - 0.75)) :]
            ids_shuffle_disc_2_temp = ids_shuffle_disc_1[:, int(L2 * (1 - 0.90)) :]
            ids_shuffle_disc_2 = torch.cat(
                (ids_shuffle_keep_1, ids_shuffle_disc_2_temp), dim=1
            )
            ids_shuffle_keep_2 = ids_shuffle_disc_1[:, : int(L2 * (1 - 0.90))]

            ids_shuffle_keep = [ids_shuffle_keep_1, ids_shuffle_keep_2]
            ids_shuffle_disc = [ids_shuffle_disc_1, ids_shuffle_disc_2]

            ids_shuffle = []
            for z in ids_shuffle_keep:
                ids_shuffle.append(z)
            for z in ids_shuffle_disc:
                ids_shuffle.append(z)
            ids_shuffle = torch.cat(ids_shuffle, dim=1)
        else:
            if mask is None:
                # sort noise for each sample
                ids_shuffle = torch.argsort(
                    noise, dim=1
                )  # ascend: small is keep, large is remove
            else:
                ids_shuffle = mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_scale(self, x, resolutions, timestamps, mask_ratio, mask=None):
        # def forward_encoder(self, x, timestamps, mask_ratio, mask=None, input_res=1.0):
        # embed patches
        # x[0] = torch.squeeze(x[0])
        # x[1] = torch.squeeze(x[1])
        # x[2] = torch.squeeze(x[2])
        # _, _, _, h, w = x.shape
        x1 = self.patch_embed_1(x[0])
        x2 = self.patch_embed_2(x[1])
        x3 = self.patch_embed_3(x[2])
        # self.x1 = self.patch_embed(x[0])
        # self.x2 = self.patch_embed(x[1])
        # self.x3 = self.patch_embed(x[2])
        x = torch.cat([x1, x2, x3], dim=1)
        mock_res_1 = torch.ones(x.shape[0])
        mock_res_2 = torch.ones(x.shape[0])
        mock_res_3 = torch.ones(x.shape[0])
        # mock_res_1 = torch.ones(x.shape[0]) / 2
        # mock_res_2 = torch.ones(x.shape[0])
        # mock_res_3 = torch.ones(x.shape[0]) * 1.75
        # mock_res_1 = torch.ones(7)
        # mock_res_2 = torch.ones(7) * resolutions[1] * 1.4
        # mock_res_3 = torch.ones(7) * 2

        # print(timestamps.shape, x.shape)

        # ts_embed_1 = get_1d_sincos_pos_embed_from_grid_torch(
        #     128, timestamps.reshape(-1, 3)[:, 0].float()
        # )
        # ts_embed_2 = get_1d_sincos_pos_embed_from_grid_torch(
        #     128, timestamps.reshape(-1, 3)[:, 1].float()
        # )
        # ts_embed_3 = get_1d_sincos_pos_embed_from_grid_torch(
        #     128, timestamps.reshape(-1, 3)[:, 2].float()
        # )

        # ts_embed_1 = ts_embed_1.reshape(-1, 3, ts_embed_1.shape[-1]).unsqueeze(2)
        # ts_embed_2 = ts_embed_2.reshape(-1, 3, ts_embed_2.shape[-1]).unsqueeze(2)
        # ts_embed_3 = ts_embed_3.reshape(-1, 3, ts_embed_3.shape[-1]).unsqueeze(2)

        # ts_embed_1 = ts_embed_1.expand(-1, -1, x1.shape[1], -1)
        # ts_embed_2 = ts_embed_2.expand(-1, -1, x2.shape[1] // 3, -1).reshape(
        #     x2.shape[0], -1, ts_embed_2.shape[-1]
        # )
        # ts_embed_3 = ts_embed_3.expand(-1, -1, x3.shape[1], -1).reshape(
        #     x3.shape[0], -1, ts_embed_3.shape[-1]
        # )

        # ts_embed = torch.cat(
        #     [ts_embed_1, ts_embed_2, ts_embed_3],
        #     dim=1,
        # ).float()

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 1].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 2].float()
                ),
            ],
            dim=1,
        ).float()

        # # print(ts_embed, ts_embed.shape)

        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)

        # ts_embed_1 = ts_embed[:, 0, :, :].expand(-1, x1.shape[1], -1)
        # ts_embed_2 = ts_embed[:, 1, :, :].expand(-1, x2.shape[1], -1)
        # ts_embed_3 = ts_embed[:, 2, :, :].expand(-1, x3.shape[1], -1)

        # ts_embed = torch.cat(
        #     [ts_embed_1, ts_embed_2, ts_embed_3],
        #     dim=1,
        # ).float()

        # bla1 = ts_embed.expand(-1, -1, x.shape[1] // 3, -1)
        # bla2 = bla1.reshape(x.shape[0], -1, ts_embed.shape[-1])

        # # # print(ts_embed.shape)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )

        # print(torch.all(ts_embed_bla.eq(ts_embed)))

        # print(ts_embed.shape)
        # ts_embed = torch.zeros_like(ts_embed)

        # num_patches = int(
        #     (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        # )

        # pos_embed = torch.cat(
        #     [
        #         get_2d_sincos_pos_embed_with_resolution(
        #             640,
        #             int(196**0.5),
        #             mock_res_1,
        #             cls_token=True,
        #             device=self.x1.device,
        #         ),
        #         get_2d_sincos_pos_embed_with_resolution(
        #             640,
        #             int(100**0.5),
        #             mock_res_2,
        #             cls_token=True,
        #             device=self.x2.device,
        #         ),
        #         get_2d_sincos_pos_embed_with_resolution(
        #             640,
        #             int(49**0.5),
        #             mock_res_3,
        #             cls_token=True,
        #             device=self.x3.device,
        #         ),
        #     ],
        #     dim=1,
        # ).float()

        # bla1 = get_2d_sincos_pos_embed_with_resolution(
        #     640,
        #     int(196**0.5),
        #     mock_res_1,
        #     cls_token=True,
        #     device=x1.device,
        # )[:, 1:, :]

        # bla2 = get_2d_sincos_pos_embed_with_resolution(
        #     640,
        #     int(196**0.5),
        #     mock_res_2,
        #     cls_token=True,
        #     device=x2.device,
        # )[:, 1:, :]

        # bla3 = get_2d_sincos_pos_embed_with_resolution(
        #     640,
        #     int(196**0.5),
        #     mock_res_3,
        #     cls_token=True,
        #     device=x3.device,
        # )[:, 1:, :]

        # pos_embed = torch.cat(
        #     [
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.embed_dim - 384,
        #             int(196**0.5),
        #             mock_res_1,
        #             cls_token=True,
        #             device=x1.device,
        #         )[:, 1:, :],
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.embed_dim - 384,
        #             int(196**0.5),
        #             mock_res_2,
        #             cls_token=True,
        #             device=x2.device,
        #         )[:, 1:, :],
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.embed_dim - 384,
        #             int(196**0.5),
        #             mock_res_3,
        #             cls_token=True,
        #             device=x3.device,
        #         )[:, 1:, :],
        #     ],
        #     dim=1,
        # ).float()

        nekaj1 = self.pos_embed[:, 1:, :]
        nekaj2 = nekaj1.repeat(ts_embed.shape[0], 3, 1)

        # nekaj = (nekaj2 - pos_embed) / nekaj2
        # print(torch.max(nekaj))

        # add pos embed w/o cls token
        x = x + torch.cat([nekaj2, ts_embed], dim=-1)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)

        # append cls token
        cls_token = self.cls_token  # + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            # print(x.dtype)
            x = blk(x)
        x = self.norm(x)

        # Added back to the mask token in decoder for decoding modes != "demasking"
        # pos_embed_encoder = get_2d_sincos_pos_embed_with_resolution(
        #     self.decoder_embed_dim,
        #     int(num_patches**0.5),
        #     ratios,
        #     cls_token=True,
        #     device=x.device,
        # )

        return x, mask, ids_restore

    def forward_decoder_scale(self, x, resolutions, timestamps, ids_restore):
        ##################################################################################
        # embed tokens
        x = self.decoder_embed(x)
        # mock_res_1 = torch.ones(128) * resolutions[0]
        # mock_res_2 = torch.ones(128) * resolutions[1] * 1.4
        # mock_res_3 = torch.ones(128) * resolutions[2] * 2

        mock_res_1 = torch.ones(x.shape[0])
        mock_res_2 = torch.ones(x.shape[0])
        mock_res_3 = torch.ones(x.shape[0])

        # mock_res_1 = torch.ones(x.shape[0]) / 2
        # mock_res_2 = torch.ones(x.shape[0])
        # mock_res_3 = torch.ones(x.shape[0]) * 1.75

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        ##################################################################################

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 1].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 2].float()
                ),
            ],
            dim=1,
        ).float()

        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)

        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )

        # ts_embed_1 = ts_embed[:, 0, :, :].expand(-1, 196, -1)
        # ts_embed_2 = ts_embed[:, 1, :, :].expand(-1, 196, -1)
        # ts_embed_3 = ts_embed[:, 2, :, :].expand(-1, 196, -1)

        # ts_embed = torch.cat(
        #     [ts_embed_1, ts_embed_2, ts_embed_3],
        #     dim=1,
        # ).float()

        ts_embed = torch.cat(
            [
                torch.zeros(
                    (ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device
                ),
                ts_embed,
            ],
            dim=1,
        )

        # pos_embed = torch.cat(
        #     [
        #         get_2d_sincos_pos_embed_with_resolution(
        #             320,
        #             int(196**0.5),
        #             mock_res_1,
        #             cls_token=True,
        #             device=x.device,
        #         ),
        #         get_2d_sincos_pos_embed_with_resolution(
        #             320,
        #             int(100**0.5),
        #             mock_res_2,
        #             cls_token=True,
        #             device=x.device,
        #         ),
        #         get_2d_sincos_pos_embed_with_resolution(
        #             320,
        #             int(49**0.5),
        #             mock_res_3,
        #             cls_token=True,
        #             device=x.device,
        #         ),
        #     ],
        #     dim=1,
        # ).float()

        # pos_embed = torch.cat(
        #     [
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.decoder_embed_dim - 192,
        #             int(196**0.5),
        #             mock_res_1,
        #             cls_token=True,
        #             device=x.device,
        #         )[:, 1:, :],
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.decoder_embed_dim - 192,
        #             int(196**0.5),
        #             mock_res_2,
        #             cls_token=True,
        #             device=x.device,
        #         )[:, 1:, :],
        #         get_2d_sincos_pos_embed_with_resolution(
        #             self.decoder_embed_dim - 192,
        #             int(196**0.5),
        #             mock_res_3,
        #             cls_token=True,
        #             device=x.device,
        #         )[:, 1:, :],
        #     ],
        #     dim=1,
        # ).float()

        # nekaj2 = self.decoder_pos_embed[:, 1:, :].repeat(1, 3, 1)

        nekaj1 = self.decoder_pos_embed[:, :1, :]
        nekaj2 = self.decoder_pos_embed[:, 1:, :].repeat(1, 3, 1)
        nekaj3 = torch.cat(
            [
                nekaj1,
                nekaj2,
            ],
            dim=1,
        ).expand(ts_embed.shape[0], -1, -1)

        # print(torch.all(pos_embed.eq(nekaj2)))

        # bla = self.decoder_pos_embed[:, :1, :].repeat(x.shape[0], 1, 1)

        # pos_embed = torch.cat(
        #     [
        #         self.decoder_pos_embed[:, :1, :].repeat(x.shape[0], 1, 1),
        #         pos_embed,
        #     ],
        #     dim=1,
        # )

        # ts_embed = torch.zeros_like(ts_embed)

        # add pos embed
        x = x + torch.cat([nekaj3, ts_embed], dim=-1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_encoder(self, x, resolutions, timestamps, mask_ratio, mask=None):
        # embed patches
        x1 = self.patch_embed(x[0])
        x2 = self.patch_embed(x[1])
        x3 = self.patch_embed(x[2])
        x = torch.cat([x1, x2, x3], dim=1)
        # x = torch.cat([x1, x2], dim=1)
        # print(timestamps.shape, x.shape)
        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 1].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 2].float()
                ),
            ],
            dim=1,
        ).float()

        # print(ts_embed, ts_embed.shape)

        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        # ts_embed = ts_embed.reshape(-1, 2, ts_embed.shape[-1]).unsqueeze(2)

        # print(ts_embed.shape)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )
        # ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 2, -1).reshape(
        #     x.shape[0], -1, ts_embed.shape[-1]
        # )
        # print(ts_embed.shape)
        # ts_embed = torch.zeros_like(ts_embed)

        # nekaj1 = self.pos_embed[:, 1:, :]
        # nekaj2 = nekaj1.repeat(ts_embed.shape[0], 3, 1)

        # add pos embed w/o cls token
        x = x + torch.cat(
            [self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 3, 1), ts_embed], dim=-1
        )
        # x = x + torch.cat(
        #     [self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 2, 1), ts_embed], dim=-1
        # )

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.dtype)

        # apply Transformer blocks
        for blk in self.blocks:
            # print(x.dtype)
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_2_images(
        self, x, resolutions, timestamps, mask_ratio, mask=None
    ):
        # embed patches
        x1 = self.patch_embed(x[0])
        x2 = self.patch_embed(x[1])
        # x3 = self.patch_embed(x[2])
        x = torch.cat([x1, x2], dim=1)
        # print(timestamps.shape, x.shape)
        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 1].float()
                ),
            ],
            dim=1,
        ).float()

        # print(ts_embed, ts_embed.shape)

        ts_embed = ts_embed.reshape(-1, 2, ts_embed.shape[-1]).unsqueeze(2)

        # print(ts_embed.shape)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 2, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )
        # print(ts_embed.shape)
        # ts_embed = torch.zeros_like(ts_embed)

        # nekaj1 = self.pos_embed[:, 1:, :]
        # nekaj2 = nekaj1.repeat(ts_embed.shape[0], 3, 1)

        # add pos embed w/o cls token
        x = x + torch.cat(
            [self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], 2, 1), ts_embed], dim=-1
        )

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking_2(x, mask_ratio, mask=mask)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.dtype)

        # apply Transformer blocks
        for blk in self.blocks:
            # print(x.dtype)
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, resolutions, timestamps, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 1].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 2].float()
                ),
            ],
            dim=1,
        ).float()

        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 3, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )

        ts_embed = torch.cat(
            [
                torch.zeros(
                    (ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device
                ),
                ts_embed,
            ],
            dim=1,
        )

        # ts_embed = torch.zeros_like(ts_embed)

        # add pos embed
        x = x + torch.cat(
            [
                torch.cat(
                    [
                        self.decoder_pos_embed[:, :1, :],
                        self.decoder_pos_embed[:, 1:, :].repeat(1, 3, 1),
                    ],
                    dim=1,
                ).expand(ts_embed.shape[0], -1, -1),
                ts_embed,
            ],
            dim=-1,
        )

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_2_images(self, x, resolutions, timestamps, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    64, timestamps.reshape(-1, 3)[:, 1].float()
                ),
            ],
            dim=1,
        ).float()

        ts_embed = ts_embed.reshape(-1, 2, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // 2, -1).reshape(
            x.shape[0], -1, ts_embed.shape[-1]
        )

        ts_embed = torch.cat(
            [
                torch.zeros(
                    (ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device
                ),
                ts_embed,
            ],
            dim=1,
        )

        # ts_embed = torch.zeros_like(ts_embed)

        # add pos embed
        x = x + torch.cat(
            [
                torch.cat(
                    [
                        self.decoder_pos_embed[:, :1, :],
                        self.decoder_pos_embed[:, 1:, :].repeat(1, 2, 1),
                    ],
                    dim=1,
                ).expand(ts_embed.shape[0], -1, -1),
                ts_embed,
            ],
            dim=-1,
        )

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target1 = self.patchify(imgs[0])
        target2 = self.patchify(imgs[1])
        # target3 = self.patchify(imgs[2])
        # target = torch.cat([target1, target2, target3], dim=1)
        target = torch.cat([target1, target2], dim=1)
        # previous_target = target

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # preds = pred.reshape(pred.shape[0], 3, -1, pred.shape[-1])

        # for i in range(pred.shape[0]):
        #     pred = self.unpatchify(preds[i])
        #     pred1 = pred[0]
        #     pred2 = pred[1]
        #     pred3 = pred[2]
        #     pred2 = self.scale_2(pred2)
        #     pred2 = self.scale_1(pred2)
        #     pred3 = self.scale_3(pred3)
        #     pred3 = self.scale_1(pred3)

        #     pred = torch.stack([pred1, pred2, pred3], dim=0)

        #     if i == 0:
        #         new_preds1 = pred1.unsqueeze(0)
        #         new_preds2 = pred2.unsqueeze(0)
        #         new_preds3 = pred3.unsqueeze(0)
        #     else:
        #         new_preds1 = torch.cat([new_preds1, pred1.unsqueeze(0)], dim=0)
        #         new_preds2 = torch.cat([new_preds2, pred2.unsqueeze(0)], dim=0)
        #         new_preds3 = torch.cat([new_preds3, pred3.unsqueeze(0)], dim=0)

        # new_preds1 = self.patchify(new_preds1)
        # new_preds2 = self.patchify(new_preds2)
        # new_preds3 = self.patchify(new_preds3)

        # pred = torch.cat([new_preds1, new_preds2, new_preds3], dim=1)

        # viz code
        """
        m = torch.tensor([0.4182007312774658, 0.4214799106121063, 0.3991275727748871]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.28774282336235046, 0.27541765570640564, 0.2764017581939697]).reshape(1, 3, 1, 1)
        
        image = (pred * (var + 1.e-6)**.5) + mean
        bs = image.shape[0]
        image = image.reshape(bs, 3, -1, image.shape[-1])[0]
        image = self.unpatchify(image).detach().cpu()
        image = image * std + m

        save_image(image, f'viz1/viz_{self.counter}.png')
        masked_image = self.patchify(image)
        masked_image.reshape(-1, 768)[mask[0].bool()] = 0.5
        masked_image = self.unpatchify(masked_image.reshape(3, -1 ,768))
        save_image(masked_image, f'viz1/viz_mask_{self.counter}.png')

        previous_target = previous_target.reshape(bs, 3, -1, previous_target.shape[-1])[0]
        previous_target = self.unpatchify(previous_target).detach().cpu()
        previous_target = previous_target * std + m
        save_image(previous_target, f'viz1/target_{self.counter}.png')

        masked_image = self.patchify(previous_target)
        masked_image.reshape(-1, 768)[mask[0].bool()] = 0.5
        masked_image = self.unpatchify(masked_image.reshape(3, -1 ,768))
        save_image(masked_image, f'viz1/viz_target_mask_{self.counter}.png')
        # print(image.shape)
        # assert False
        self.counter += 1
        """

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # def forward(
    #     self,
    #     imgs,
    #     mask_ratio=0.75,
    #     input_res=None,
    #     timestamps=None,
    #     mask=None,
    # ):
    # def forward(
    #     self, imgs, timestamps, ratios, mask_ratio=0.75, mask=None, input_res=1.0
    # ):
    def forward(
        self, imgs, resolutions, timestamps, mask_ratio=0.75, mask=None, input_res=1.0
    ):
        # latent, mask, ids_restore, pos_embed_encoder = self.forward_encoder(
        #     imgs, timestamps, input_res, mask_ratio, mask
        # )
        # latent, mask, ids_restore = self.forward_encoder(
        #     imgs, timestamps, mask_ratio, mask, input_res
        # )
        latent, mask, ids_restore = self.forward_encoder_2_images(
            imgs, resolutions, timestamps, mask_ratio, mask=mask
        )
        pred = self.forward_decoder_2_images(
            latent, resolutions, timestamps, ids_restore
        )  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b_samemask(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore
        **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # type: ignore
        **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_samemask = mae_vit_large_patch16_dec512d8b_samemask
# from models_mae import mae_vit_large_patch16_dec512d8b
# mae_vit_large_patch16_nontemp = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
