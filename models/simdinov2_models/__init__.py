# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(vit_kwargs, only_teacher=False):
    teacher = vits.__dict__[vit_kwargs.model](
        drop_path_rate=0.0, num_classes=vit_kwargs.nb_classes
    )

    # student = vits.__dict__[vit_kwargs.model](
    #     drop_path_rate=0.3,
    #     attn_drop=0.0,
    #     ffn_drop=0.0,
    #     gradient_checkpointing=False,
    # )
    # embed_dim = student.embed_dim
    # logger.info(
    #     f"Model {student.__class__.__name__} {vit_kwargs.input_size}p{vit_kwargs.patch_size} built. Total params: {sum(p.numel() for p in student.parameters())}"
    # )
    return teacher, teacher.embed_dim
