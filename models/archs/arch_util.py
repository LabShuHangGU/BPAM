# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from utils import get_root_logger

# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class UNet(nn.Module):
    def __init__(self, ic, oc, n_scales, n_blocks, num_feat, global_ext=False):
        super().__init__()
        self.n_scales = n_scales + 1 if global_ext else n_scales
        self.first_conv = nn.Sequential(
            nn.Conv2d(ic, num_feat, 3, 1, 1),
            nn.ReLU()
        )
        self.down_blocks = nn.ModuleList()    #定义了三个空的nn.ModuleList，分别用于存储下采样块、上采样块和上采样层。
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(n_scales):    #n_scales=3; i=0,1,2
            down_block = [ResidualBlockNoBN(num_feat)]     #循环n_scales次，为每个尺度创建下采样和上采样块。
            up_block = []
            if i != n_scales - 1:
                self.upsample.append(nn.Upsample(scale_factor=2, mode='nearest'))
                up_block += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
                down_block.append(nn.Conv2d(num_feat, num_feat, 3, 2, 1))
                down_block.append(nn.ReLU())
            elif global_ext:
                up_block += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
            for _ in range(n_blocks):  #n_blocks=2; i=0,1
                up_block.append(ResidualBlockNoBN(num_feat))
            self.down_blocks.append(nn.Sequential(*down_block))
            self.up_blocks.append(nn.Sequential(*up_block))

        self.final_conv = nn.Conv2d(num_feat, oc, 3, 1, 1)

        if global_ext:
            self.down_blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                                  ResidualBlockNoBN(num_feat)))
            up_block = []
            for _ in range(n_blocks):
                up_block.append(ResidualBlockNoBN(num_feat))
            self.up_blocks.append(nn.Sequential(*up_block))

    def forward(self, x):
        x0 = self.first_conv(x)
        down_features = [x0]
        for i in range(self.n_scales):
            down_features.append(self.down_blocks[i](down_features[i]))
        out = self.up_blocks[-1](down_features[-1])
        for i in range(self.n_scales - 1):  #n_scales-1=2; 0,1
            out = self.up_blocks[self.n_scales - i - 2](
                torch.cat((self.upsample[self.n_scales - i - 2](out), 
                           down_features[self.n_scales - i - 2]), dim=1))
        return self.final_conv(out)


class UNetDecoder2(nn.Module):
    def __init__(self, ic, oc, n_scales, n_blocks, num_feat, final_layers=2, global_ext=False):
        super().__init__()
        self.n_scales = n_scales
        self.first_conv = nn.Sequential(
            nn.Conv2d(ic-1, num_feat, 3, 1, 1),
            nn.ReLU()
        )
        self.down_blocks = nn.ModuleList()
        self.up_blocks0 = nn.ModuleList()
        self.up_blocks1 = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(n_scales):
            down_block = [ResidualBlockNoBN(num_feat)]
            up_block0, up_block1 = [], []
            if i != n_scales - 1:
                self.upsample.append(nn.Upsample(scale_factor=2, mode='nearest'))
                up_block0 += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
                up_block1 += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
                down_block.append(nn.Conv2d(num_feat, num_feat, 3, 2, 1))
                down_block.append(nn.ReLU())
            elif global_ext:
                up_block0 += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
                up_block1 += [nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU()]
            for _ in range(n_blocks):
                up_block0.append(ResidualBlockNoBN(num_feat))
                up_block1.append(ResidualBlockNoBN(num_feat))
            self.down_blocks.append(nn.Sequential(*down_block))
            self.up_blocks0.append(nn.Sequential(*up_block0))
            self.up_blocks1.append(nn.Sequential(*up_block1))

        final_conv = []
        for _ in range(final_layers - 1):
            final_conv.append(ResidualBlockNoBN(num_feat))
        final_conv.append(nn.Conv2d(num_feat, oc, 3, 1, 1))
        self.final_conv = nn.Sequential(*final_conv)

        if global_ext:
            self.down_blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                                  ResidualBlockNoBN(num_feat)))
            up_block0, up_block1 = [], []
            for _ in range(n_blocks):
                up_block0.append(ResidualBlockNoBN(num_feat))
                up_block1.append(ResidualBlockNoBN(num_feat))
            self.up_blocks0.append(nn.Sequential(*up_block0))
            self.up_blocks1.append(nn.Sequential(*up_block1))

    def forward(self, x):
        mask = x[:,-4:-3,...]
        x0 = self.first_conv(torch.cat((x[:,:-4,...], x[:,-3:,...]), dim=1))
        down_features = [x0]
        for i in range(self.n_scales):
            down_features.append(self.down_blocks[i](down_features[i]))
        out0 = self.up_blocks0[-1](down_features[-1])
        out1 = self.up_blocks1[-1](down_features[-1])
        for i in range(self.n_scales - 1):
            out0 = self.up_blocks0[self.n_scales - i - 2](
                torch.cat((self.upsample[self.n_scales - i - 2](out0), 
                           down_features[self.n_scales - i - 2]), dim=1))
            out1 = self.up_blocks1[self.n_scales - i - 2](
                torch.cat((self.upsample[self.n_scales - i - 2](out1), 
                           down_features[self.n_scales - i - 2]), dim=1))
        out = mask * out0 + (1 - mask) * out1
        return self.final_conv(out)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_normalized + self.bias
    
# handle multiple input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

import time
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps