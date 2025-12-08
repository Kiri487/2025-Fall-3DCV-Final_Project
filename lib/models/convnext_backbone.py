# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is adapted from https://github.com/facebookresearch/ConvNeXt
# and https://github.com/open-mmlab/mmcv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import logging
import os

logger = logging.getLogger(__name__)

# Pre-trained weights URL for ConvNeXt-Tiny (from official repo)
PRETRAINED_URL_CONVNEXT_TINY = 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth'

def get_convnext_backbone(cfg, **kwargs):
    """
    Factory function to create a ConvNeXt backbone.
    """
    model = ConvNeXt(
        in_chans=cfg.network.CONVNEXT.IN_CHANS,
        depths=cfg.network.CONVNEXT.DEPTHS,
        dims=cfg.network.CONVNEXT.DIMS,
        drop_path_rate=cfg.network.CONVNEXT.DROP_PATH_RATE,
        layer_scale_init_value=cfg.network.CONVNEXT.LAYER_SCALE_INIT_VALUE,
        head_init_scale=cfg.network.CONVNEXT.HEAD_INIT_SCALE,
    )

    # Load pre-trained weights
    if cfg.network.CONVNEXT.PRETRAINED:
        logger.info(f"Loading custom pretrained model from {cfg.network.CONVNEXT.PRETRAINED}")
        model.init_weights(pretrained=cfg.network.CONVNEXT.CONVNEXT.PRETRAINED)
    else:
        logger.info(f"Loading official pretrained model for ConvNeXt-Tiny from {PRETRAINED_URL_CONVNEXT_TINY}")
        try:
            state_dict = load_state_dict_from_url(PRETRAINED_URL_CONVNEXT_TINY, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            logger.info("Successfully loaded ConvNeXt-Tiny pretrained weights.")
        except Exception as e:
            logger.warning(f"Could not load official ConvNeXt-Tiny pretrained weights: {e}")
            logger.warning("ConvNeXt will be trained from scratch.")

    # Determine output channels for the head.
    # ConvNeXt typically outputs the last stage's dimension as the feature map.
    model.out_channels = cfg.network.CONVNEXT.DIMS[-1]
    
    return model


class Block(nn.Module):
    r""" ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # point-wise conv1
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # point-wise conv2
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init value for classifier head. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        # 0. Stem: (N, 3, H, W) -> (N, 96, H/4, W/4)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 1-3. Intermediate downsampling layers (N, C, H, W) -> (N, 2C, H/2, W/2)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
 
        # 4 feature resolution stages, each consisting of ConvNeXt blocks
        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value
                ))
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.out_channels = dims[-1] # For external use by heads
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        # We need the feature map, not the pooled features for classification.
        # So we adapt forward to return the output of the last stage.
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x # (N, C, H, W)

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # point-wise conv1
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # point-wise conv2
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
