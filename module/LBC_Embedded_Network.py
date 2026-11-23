import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from .mamba2 import Mamba2_2
from .mamba_util import bulid_act
from .mamba_util import ConvFFN_DW
from einops import rearrange
import pywt
import numpy as np

# class Origin_area_Block(nn.Module):
#     def __init__(self, in_channel,out_channel,crop_size):
#         super(Origin_area_Block, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.crop_size = crop_size
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
#         self.act = nn.SiLU()
#     def center_crop(self,input_tensor, crop_size):
#
#         B, C, H, W = input_tensor.shape
#
#         if isinstance(crop_size, int):
#             crop_h = crop_w = crop_size
#         else:
#             crop_h, crop_w = crop_size
#
#         # 计算起始位置
#         start_h = (H - crop_h) // 2
#         start_w = (W - crop_w) // 2
#
#         # 执行裁剪
#         cropped = input_tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
#
#         return cropped
#     def forward(self, input):
#         origin_data = self.center_crop(input, self.crop_size)
#         out = origin_data + input
#         out = self.conv(out)
#         out = self.act(out)
#         return out
class Crop_add_origin_area(nn.Module):
    def __init__(self, in_channel,out_channel,crop_size):
        super(Crop_add_origin_area, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.crop_size = crop_size
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.act = nn.SiLU()
    def center_crop(self,input_tensor, crop_size):

        B, C, H, W = input_tensor.shape

        if isinstance(crop_size, int):
            crop_h = crop_w = crop_size
        else:
            crop_h, crop_w = crop_size

        # 计算起始位置
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2

        # 执行裁剪
        cropped = input_tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

        return cropped
    def forward(self, data1,data_final):
        origin_data = self.center_crop(data_final, self.crop_size)
        out = torch.add(origin_data,data1)
        out = self.conv(out)
        out = self.act(out)
        return out
class Cascade_SCSSD_Block(nn.Module):
    def __init__(self, in_channel_list,out_channel_list,ssd_config,split_nums,d_state=64, d_conv=3, expand=2):
        super(Cascade_SCSSD_Block, self).__init__()

        self.blocks = nn.ModuleList([Mamba2_2(d_model = in_channel_list[i] // split_nums,  # Model dimension d_model
                                              out_channel = out_channel_list[i] // split_nums,  # Model dimension d_model
                                              d_state = d_state,  # SSM state expansion factor
                                              d_conv = d_conv,  # Local convolution width
                                              expand = expand,  # Block expansion factor
                                              headdim = 8,
                                              **ssd_config
                                              )
                                     for i in range(len(in_channel_list))])
    def forward(self, x,h,w):
        for block in self.blocks:
            x = block(x,h,w)
        return x
class SCSSD(nn.Module):
    def __init__(self, in_channel_list, out_channel_list,ssd_config,split_nums=4,d_state=64, d_conv=3, expand=2,mode='encoder'):
        super().__init__()
        self.input_dim = in_channel_list
        self.output_dim = out_channel_list
        self.split_nums = split_nums
        self.d_state = d_state
        self.ssd_config = ssd_config

        self.cascade_scssd = Cascade_SCSSD_Block(self.input_dim, self.output_dim,self.ssd_config,self.split_nums)
        self.sub_mamba = nn.ModuleList([self.cascade_scssd for _ in range(self.split_nums)])

        self.norm_1 = nn.LayerNorm(self.input_dim[0])
        self.act = nn.SiLU()

        if mode == 'encoder':
            self.fnn = ConvFFN_DW(self.input_dim[-1])
        elif mode == 'decoder':
            self.fnn = ConvFFN_DW(self.input_dim[-1])
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        # assert C == self.input_dim

        n_tokens = x.shape[2] * x.shape[3]
        img_dims = x.shape[2:]
        # print("x.reshape(B, C, n_tokens): ",x.reshape(B, C, n_tokens).shape)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print("x_flat: ", x_flat.shape)

        x_norm = self.norm_1(x_flat)

        xs = [self.sub_mamba[i](x_i,int(math.sqrt(n_tokens)),int(math.sqrt(n_tokens))) + x_i * self.skip_scale
              for i, x_i in enumerate(torch.chunk(x_norm, self.split_nums, 2))]

        out = torch.cat(xs, dim=2)
        out = self.act(out)
        out = out.transpose(-1, -2).reshape(B, self.output_dim[-1], *img_dims)
        out = self.fnn(out)

        return out
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
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1,padding_mode='reflect'))
                m.append(nn.PixelShuffle(2))

        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1,padding_mode='reflect'))
            m.append(nn.PixelShuffle(3))

        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class Upsampler(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat,act):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2)),
                m.append(act)
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3)),
            m.append(act)
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsampler, self).__init__(*m)
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,act=None,norm=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,
                                                 in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 groups=in_channels,
                                                 padding_mode='reflect'),  # 深度卷积
                                       nn.SiLU())
                                       # nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                       nn.LayerNorm(out_channels),
                                       nn.SiLU())

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8,act=None,norm=None):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self,act=None,norm=None):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3,padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, ssd_config, split_nums,act=None,norm=None):
        super(CBAM, self).__init__()
        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.channel_attention = ChannelAttentionModule(channel,act=act, norm=norm)
        self.spatial_attention = SpatialAttentionModule(act=act, norm=norm)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, ssd_config, split_nums,act=None,norm=None):
        super(MSCB, self).__init__()

        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.act = act
        self.norm = norm

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   self.act)

        self.dwcovn1 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=out_channels,
                                 padding_mode='reflect')  # 深度卷积

        self.dwcovn2 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 groups=out_channels,
                                 padding_mode='reflect')  # 深度卷积
        self.dwcovn3 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=7,
                                 stride=1,
                                 padding=3,
                                 groups=out_channels,
                                 padding_mode='reflect')  # 深度卷积
        self.pointwise = nn.Sequential(nn.Conv2d(3 * out_channels,
                                                 in_channels,
                                                 kernel_size=1),
                                       self.act)

        self.cbam = CBAM(in_channels, self.ssd_config, self.split_nums,act=self.act,norm=self.norm)

    def forward(self, x):
        x_init = x
        x = self.conv1(x)
        x1 = self.dwcovn1(x)
        x2 = self.dwcovn2(x)
        x3 = self.dwcovn3(x)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.pointwise(x) + x_init
        x = self.act(x)
        x = self.cbam(x)
        return x


class SlidingWindowAttention(nn.Module):
    def __init__(self, channel, window_size=7, num_heads=4, act=None, norm=None):
        super(SlidingWindowAttention, self).__init__()
        self.channel = channel
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channel // num_heads

        self.qkv = nn.Conv2d(channel, channel * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channel, channel, kernel_size=1)

        self.scale = self.head_dim ** -0.5
        self.act = act if act is not None else nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        q, k, v = qkv.unbind(1)

        # Pad for sliding window
        pad_size = self.window_size // 2
        q = F.pad(q, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        k = F.pad(k, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        v = F.pad(v, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        # Extract sliding windows
        q_windows = self.extract_windows(q, self.window_size)
        k_windows = self.extract_windows(k, self.window_size)
        v_windows = self.extract_windows(v, self.window_size)

        # Compute attention within each window
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out_windows = attn @ v_windows

        # Merge windows back
        out = self.merge_windows(out_windows, H, W, pad_size)

        # Project
        out = self.proj(out)

        return out

class LPEmbedNet(nn.Module):

    def __init__(self,
                 input_channels=3,
                 out_channels=1,
                 rs_factor=2,
                 c_list=[],
                 res=False,
                 split_nums=8,
                 atten_config={},
                 ssd_config={},
                 act_name='silu',
                 norm_name='ln'):
        super().__init__()

        self.rs_factor = rs_factor
        self.c_list = c_list
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.split_nums = split_nums
        self.atten_config = atten_config
        self.ssd_config = ssd_config
        self.res = res
        self.act_name = act_name
        self.norm_name = norm_name

        self.crop_size = 36

        self.act,self.norm = bulid_act(self.act_name,self.norm_name)
        # encoder_stage1

        self.stage1_en = DepthwiseSeparableConv2d(input_channels, self.c_list[0],act=self.act)

        #origin_area_embedNet
        self.oringin_area_embedNet = Crop_add_origin_area(self.c_list[0], self.c_list[0], self.crop_size)

        # encoder_stage2
        self.att_1 = MSCB(self.c_list[0],3,self.atten_config,self.split_nums,act=self.act,norm=self.norm)
        self.stage2_en = SCSSD(in_channel_list=[self.c_list[0]],
                                     out_channel_list=[self.c_list[0]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode = 'encoder')


        self.gs_2 = nn.Sequential(
            nn.Conv2d(2*self.c_list[0], self.c_list[0], kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm(self.c_list[0]),
            nn.SiLU()
        )
        # encoder_stage3
        self.att_2 = MSCB(self.c_list[1],3,self.atten_config,self.split_nums,act=self.act,norm=self.norm)
        self.stage3_en = SCSSD(in_channel_list=[self.c_list[1]],
                                     out_channel_list=[self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )

        self.gs_3 = nn.Sequential(
            nn.Conv2d(2*self.c_list[1], self.c_list[1], kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm(self.c_list[1]),
            nn.SiLU()
        )
        # encoder_stage4
        self.att_3 = MSCB(self.c_list[2], 3, self.atten_config, self.split_nums, act=self.act, norm=self.norm)
        self.stage4_en = SCSSD(in_channel_list=[self.c_list[2]],
                                     out_channel_list=[self.c_list[2]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )
        self.gs_4 = nn.Sequential(
            nn.Conv2d(2*self.c_list[2], self.c_list[2], kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm(self.c_list[2]),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        out1 = self.stage1_en(x)

        attn_1 = self.att_1(out1)
        out2 = self.stage2_en(out1)
        out2 = torch.cat(out2, attn_1)
        out2 = self.gs_2(out2)

        attn_2 = self.att_2(out2)
        out3 = self.stage3_en(out2)
        out3 = torch.cat(out3, attn_2)
        out3 = self.gs_3(out3)

        attn_3 = self.att_3(out3)
        out4 = self.stage4_en(out3)
        out4 = torch.cat(out4, attn_3)
        out4 = self.gs_4(out4)

        if self.res:
            out4 = self.oringin_area_embedNet(out1,out4)
        else:
            # Crop to central 36x36 region
            start_h = (H - self.crop_size) // 2
            start_w = (W - self.crop_size) // 2
            out4 = out4[:, :, start_h:start_h + self.crop_size, start_w:start_w + self.crop_size]
        return out4

