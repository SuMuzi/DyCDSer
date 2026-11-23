import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from module.mamba2 import Mamba2_2,Mamba2_1
from module.mamba_util import bulid_act
from mamba_ssm import Mamba2
from module.mamba_util import Mlp,MLP_SCSSD,ConvFFN_SCSSD
from einops import rearrange
import pywt
from utils.utils import HaarWavelet2D,HaarInverseWavelet2D
class IntensityClassifier(nn.Module):
    """
    输入：SR 或 HR 降水场 (B,1,H,W)
    输出：logits (B,K,H,W)  K=强度等级数，逐像素分类
    """
    def __init__(self, in_c=1, K=5, base=64, act=nn.ReLU()):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, base, 3, 1, 1),
            act,
            nn.Conv2d(base, base*2, 3, 1, 1),
            act,
            nn.Conv2d(base*2, K, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x)   # (B,K,H,W)


class LowerBoundedWaveletSubbandAttention(nn.Module):
    """
    带下限约束的离散小波变换四子带注意力机制（不依赖torchwavelets库）

    功能:
    1. 将输入特征图通过Haar小波变换分解为四个子带(LL, LH, HL, HH)
    2. 学习每个子带的动态注意力权重，带下限约束
    3. 下限约束为[n1, n2, n3, n4]的反比
    4. 使用权重对子带系数进行加权
    5. 通过逆小波变换重建特征图
    """

    def __init__(self, channels, lower_bound_nums=[130064682, 25963472, 25963472, 1647723], lambda_weight=0.1):
        """
        初始化参数

        参数:
            channels: 输入特征图的通道数
            lower_bound_nums: 四个子带的下限约束基数 [n1, n2, n3, n4]
            lambda_weight: 下限约束的权重，控制约束的强度
        """
        super(LowerBoundedWaveletSubbandAttention, self).__init__()
        self.channels = channels
        self.lambda_weight = lambda_weight

        # 计算下限约束：为[n1, n2, n3, n4]的反比并归一化
        self.lower_bound_nums = torch.tensor(lower_bound_nums, dtype=torch.float32)

        # 添加数值稳定性保护：确保所有值都大于0
        inverse_bounds = 1.0 / torch.clamp(self.lower_bound_nums, min=1e-8)
        sum_inverse = inverse_bounds.sum()

        # 防止除零错误
        if sum_inverse == 0:
            self.normalized_lower_bounds = torch.ones_like(inverse_bounds) / len(inverse_bounds)
        else:
            self.normalized_lower_bounds = inverse_bounds / sum_inverse

        # 1. 初始化Haar小波变换和逆变换
        self.dwt = HaarWavelet2D()
        self.idwt = HaarInverseWavelet2D()

        # 2. 子带注意力网络 - 改进版
        # 将所有子带的信息进行融合
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),  # 通道融合
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),  # 展平为向量
            nn.Linear(channels, channels // 4),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(channels // 4, 4),  # 输出4个子带的注意力分数
            # 注意：这里不使用Softmax，因为需要在添加约束后再进行归一化
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征图，形状为 [B, C, H, W]

        返回:
            output: 注意力加权后的重构特征图，形状为 [B, C, H, W]
            attention_weights: 各子带的注意力权重，形状为 [B, 4]
            constrained_weights: 应用下限约束后的注意力权重，形状为 [B, 4]
        """
        B, C, H, W = x.shape

        # 步骤1: 小波变换分解
        LL, LH, HL, HH = self.dwt(x)

        # 步骤2: 生成动态注意力权重
        # 将四个子带的特征拼接在一起
        subbands_concat = torch.cat([LL, LH, HL, HH], dim=1)  # [B, 4*C, H/2, W/2]
        # 计算注意力分数（未归一化）
        dynamic_scores = self.attention_net(subbands_concat)  # [B, 4]

        # 步骤3: 应用下限约束
        # 将下限约束扩展到批次维度
        lower_bounds = self.normalized_lower_bounds.to(x.device).unsqueeze(0).repeat(B, 1)  # [B, 4]

        # 使用对数形式添加约束，添加数值稳定性保护
        # 确保所有值都大于0，避免log(0)或log(负数)的问题
        safe_lower_bounds = torch.clamp(lower_bounds, min=1e-8)
        constrained_scores = dynamic_scores + self.lambda_weight * torch.log(safe_lower_bounds)  # [B, 4]

        # 归一化得到最终权重
        constrained_weights = F.softmax(constrained_scores, dim=1)  # [B, 4]

        # 步骤4: 应用注意力权重
        # 将权重应用到对应的子带
        LL_weighted = LL * constrained_weights[:, 0].view(B, 1, 1, 1)
        LH_weighted = LH * constrained_weights[:, 1].view(B, 1, 1, 1)
        HL_weighted = HL * constrained_weights[:, 2].view(B, 1, 1, 1)
        HH_weighted = HH * constrained_weights[:, 3].view(B, 1, 1, 1)

        # 步骤5: 逆小波变换重建
        output = self.idwt(LL_weighted, LH_weighted, HL_weighted, HH_weighted)

        return output

class Extract_high_freq_by_walvet(nn.Module):
    """
    输入：SR 或 HR 降水场 (B,1,H,W)
    输出：logits (B,K)  K=强度等级数
    """

    def __init__(self, wavelet='db4'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self,data):
        coeffs = pywt.dwt2(data.cpu().numpy(),self.wavelet)
        cA,(cH,cV,cD) = coeffs # cA:低频, cH:水平, cV:垂直, cD:对角
        hf_codffs = (cA,(2 * cH,2 * cV,2 * cD))
        hf_img = pywt.idwt2(hf_codffs,self.wavelet)
        hf_img = torch.from_numpy(hf_img)
        return hf_img.to('cuda')

class HighPass(nn.Module):
    def __init__(self):
        super(HighPass, self).__init__()

        self.kernel = torch.tensor([[-1.,-1.,-1.],
                                    [-1.,8.,-1.],
                                    [-1.,-1.,-1.]],dtype=torch.float32)
        self.register_buffer('kernel',self.kernel.view(1,1,3,3))

    def forward(self,x):
        # x:(B,C,H,W)->高通残差
        return x-F.conv2d(x,self.kernel,padding=1)
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,act=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,
                                                 in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 groups=in_channels),  # 深度卷积
                                       act)
                                       # nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                       act)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class GlobalSeed(nn.Module):
    def __init__(self, C, ssd_config, G=8, d_state=16, d_conv=3, expand=2, split_num=1):
        super().__init__()
        self.tokenize = nn.Conv2d(C, G, 1)          # 1×1 降维
        self.mamba = Mamba2_1(
            d_model = G,          # Model dimension d_model
            d_state = d_state,    # SSM state expansion factor
            d_conv = d_conv,      # Local convolution width
            expand = expand,      # Block expansion factor
            headdim = 8,
            **ssd_config
        )              # 全局 token 扫描
        self.detokenize = nn.Conv2d(G, C, 1)        # 1×1 升维

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.tokenize(x)                        # (B,G,H,W)
        g = rearrange(g, 'b g h w -> b (h w) g')
        g = self.mamba(g,H,W) + g                       # 残差全局
        g = rearrange(g, 'b (h w) g -> b g h w', h=H, w=W)
        return x + self.detokenize(g)   # 残差回主路径

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

class SubCrossToken(nn.Module):
    def __init__(self, split_num):
        super().__init__()
        self.split_num = split_num
        self.W = nn.Parameter(torch.eye(self.split_num) + 0.01 * torch.randn(self.split_num, self.split_num))
    def forward(self, xs):
        # xs = [A1,A2,A3,A4]  每个 (B,L,C)
        B, L, C = xs[0].shape
        g = [x.mean(dim=1) for x in xs]                              # (B,C/4)
        g = torch.stack(g, dim=2)                                    # (B,C/4,4)
        weight = F.softmax(self.W, dim=1)                            # (split_num,split_num)
        g = torch.einsum('bcq,qp->bcp', g, weight)                   # 子-子交互
        g = [g[:, :, i].unsqueeze(1).expand(-1,L,-1) for i in range(self.split_num)]
        return [xs[i] + g[i] for i in range(self.split_num)]

class GlobalFusion(nn.Module):
    def __init__(self, C, G=8):
        super().__init__()
        self.low = nn.Conv2d(C, G, 1)
        self.high = nn.Conv2d(G, C, 1)

    def forward(self, x):
        return x + self.high(self.low(x))   # 残差低秩校准

class SCSSD(nn.Module):
    def __init__(self, in_channel_list, out_channel_list,ssd_config,split_nums=8,d_state=64, d_conv=3, expand=2,mode='encoder'):
        super().__init__()
        self.input_dim = in_channel_list
        self.output_dim = out_channel_list
        self.split_nums = split_nums
        self.d_state = d_state
        self.ssd_config = ssd_config
        self.global_seed = GlobalSeed(self.input_dim[0],ssd_config)
        self.cascade_scssd = Cascade_SCSSD_Block(self.input_dim, self.output_dim,self.ssd_config,self.split_nums)
        self.sub_mamba = nn.ModuleList([self.cascade_scssd for _ in range(self.split_nums)])
        self.sub_cross = SubCrossToken(self.split_nums)
        self.norm_1 = nn.LayerNorm(self.input_dim[0])
        self.norm_2 = nn.LayerNorm(self.output_dim[-1])
        self.global_fusion = GlobalFusion(self.output_dim[-1])
        if mode == 'encoder':
            self.fnn = ConvFFN_SCSSD(self.input_dim[-1],int(2*self.output_dim[-1]))
        elif mode == 'decoder':
            self.fnn = ConvFFN_SCSSD(self.input_dim[-1],int(self.output_dim[-1]/2))
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        # 1) 预埋全局
        seed = self.global_seed(x)

        B, C = x.shape[:2]
        # assert C == self.input_dim

        n_tokens = x.shape[2] * x.shape[3]
        img_dims = x.shape[2:]
        # print("x.reshape(B, C, n_tokens): ",x.reshape(B, C, n_tokens).shape)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print("x_flat: ", x_flat.shape)

        x_norm = self.norm_1(x_flat)

        # 2) 分通道 + 各自 Mamba
        xs = [self.sub_mamba[i](x_i,int(math.sqrt(n_tokens)),int(math.sqrt(n_tokens))) + x_i * self.skip_scale
              for i, x_i in enumerate(torch.chunk(x_norm, self.split_nums, 2))]

        # 3) 子-子交互
        xs = self.sub_cross(xs)

        # 4) 聚合和转换
        out = torch.cat(xs, dim=2)
        out = self.norm_2(out)
        out = out.transpose(-1, -2).reshape(B, self.output_dim[-1], *img_dims)

        # 5) 再全局校准
        out = out + seed
        # out = self.global_fusion(out)

        # 6) 增强模型表达、引入非线性以及局部空间建模
        out = self.fnn(out)

        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8,act=None):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            act,
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.act = act

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self,act=None):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = act

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, ssd_config, split_nums,act):
        super(CBAM, self).__init__()
        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.channel_attention = ChannelAttentionModule(channel,act=act)
        self.spatial_attention = SpatialAttentionModule(act=act)
        self.gs = nn.Sequential(nn.GroupNorm(self.split_nums,channel),
                                act)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return self.gs(out)


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, ssd_config, split_nums,act):
        super(MSCB, self).__init__()

        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.act = act

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   self.act)

        self.dwcovn1 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=out_channels)  # 深度卷积

        self.dwcovn2 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=out_channels)  # 深度卷积

        self.dwcovn3 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 groups=out_channels)  # 深度卷积

        self.pointwise = nn.Sequential(nn.Conv2d(3 * out_channels,
                                                 in_channels,
                                                 kernel_size=1),
                                       self.act)

        self.cbam = CBAM(in_channels, self.ssd_config, self.split_nums,act=self.act)

    def forward(self, x):
        if self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
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

        elif self.ssd_config['MSCB'] and not self.ssd_config['CBAM']:
            x_init = x
            x = self.conv1(x)
            x1 = self.dwcovn1(x)
            x2 = self.dwcovn2(x)
            x3 = self.dwcovn3(x)
            x = torch.cat((x1,x2,x3),dim=1)
            x = self.pointwise(x) + x_init
            x = self.act(x)
            return x

        elif not self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
            return self.cbam(x)
        else:
            return None



class WindGuidedAttention(nn.Module):
    """
    风场引导的空间注意力模块（轻量版）

    将 HR 风场（u/v）插值到当前特征分辨率，学习一个空间/通道注意力图，
    对 decoder 特征进行加权，从而把风场的辐合、地形引导信息注入降水超分。
    """

    def __init__(self, feat_channels, wind_channels=2, reduction=4):
        super().__init__()
        hidden_channels = max(1, feat_channels // reduction)
        self.wind_proj = nn.Sequential(
            nn.Conv2d(wind_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, feat_channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(feat_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat, wind_hr):
        if wind_hr is None:
            return feat
        # 把 HR 风场插值到特征分辨率
        wind = F.interpolate(wind_hr, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        wind = wind.to(feat.device, dtype=feat.dtype)

        # 通道级风场门控 + 空间注意力
        w_gate = self.wind_proj(wind)            # (B, C, H, W)
        att = self.spatial(feat * w_gate)        # (B, 1, H, W)
        return feat * (1 + att)


class LU_M2SR(nn.Module):

    def __init__(self,
                 input_channels=3,
                 out_channels=1,
                 rs_factor=2,
                 c_list=[],
                 res=True,
                 is_cls=False,
                 split_nums=8,
                 atten_config={},
                 ssd_config={},
                 dic_geo_lsm={},
                 act_name='silu',
                 norm_name='bn'):
        super().__init__()

        self.rs_factor = rs_factor
        self.c_list = c_list
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.split_nums = split_nums
        self.atten_config = atten_config
        self.ssd_config = ssd_config
        self.dic_geo_lsm = dic_geo_lsm
        self.res = res
        self.is_cls = is_cls
        self.act_name = act_name
        self.norm_name = norm_name

        self.act = bulid_act(self.act_name)
        self.wavelet_brand = Extract_high_freq_by_walvet()
        # encoder_stage1

        self.stage1_en = DepthwiseSeparableConv2d(input_channels, self.c_list[0],act=self.act)

        self.att_1 = MSCB(self.c_list[0],3,self.atten_config,self.split_nums,act=self.act)
        # encoder_stage2
        self.stage2_en = SCSSD(in_channel_list=[self.c_list[0],self.c_list[0]],
                                     out_channel_list=[self.c_list[0],self.c_list[0]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode = 'encoder') #1层
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gs_2 = nn.Sequential(nn.GroupNorm(self.split_nums,self.c_list[1]),
                                 self.act)
        self.att_2 = MSCB(self.c_list[1],3,self.atten_config,self.split_nums,act=self.act)
        # encoder_stage3
        self.stage3_en = SCSSD(in_channel_list=[self.c_list[1],self.c_list[1]],
                                     out_channel_list=[self.c_list[1],self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )#2层
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_3 = MSCB(self.c_list[2],3,self.atten_config,self.split_nums,act=self.act)
        self.gs_3 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[2]),
                                  self.act)
        # encoder_stage4
        self.stage4_en = SCSSD(in_channel_list=[self.c_list[2],self.c_list[2]],
                                     out_channel_list=[self.c_list[2],self.c_list[2]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )#4层
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_4 = MSCB(self.c_list[3],3,self.atten_config,self.split_nums,act=self.act)
        self.gs_4 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[3]),
                                  self.act)
        # Bottleneck Block
        self.bottleneck_block = nn.Sequential(nn.Conv2d(self.c_list[3], self.c_list[3], kernel_size=3, stride=1, padding=1,
                                              groups=self.c_list[3]),
                                              self.act)
        self.stage4_de = SCSSD(in_channel_list=[self.c_list[3],self.c_list[3]],
                                     out_channel_list=[self.c_list[3],self.c_list[3]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )#4层
        self.up_4 = nn.ConvTranspose2d(self.c_list[2], self.c_list[2], kernel_size=2, stride=2,output_padding=1)
        # self.up_4 = Upsample(2,self.c_list[2])
        self.gs_5 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[2]),
                                  self.act)
        # print("stage3_de,{0}   {1}".format(self.c_list[3],self.c_list[2]))
        self.stage3_de = SCSSD(in_channel_list=[self.c_list[2], self.c_list[2]],
                                     out_channel_list=[self.c_list[2], self.c_list[2]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )  # 2层
        # self.stage3_de = SCSSD_Layer(in_channel_list=[128, 96],
        #                              out_channel_list=[96, 64])  # 2层
        self.up_3 = Upsample(2,self.c_list[1])
        self.gs_6 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[1]),
                                  self.act)

        self.stage2_de = SCSSD(in_channel_list=[self.c_list[1],self.c_list[1]],
                                     out_channel_list=[self.c_list[1],self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )  # 1层
        self.up_2 = Upsample(2,self.c_list[0])
        self.gs_7 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[0]),
                                  self.act)
        self.reconstruct = nn.Conv2d(self.c_list[0], self.out_channels, 3, 1, 1)

        self.up_brand = Upsampler(self.rs_factor, self.out_channels,self.act)
        self.reg = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.cls_brand = IntensityClassifier(act=self.act)

        # 风场引导注意力（方案 B）
        self.wind_guided_att = WindGuidedAttention(self.c_list[0], wind_channels=2)
        self.wind_guided_att_hr = WindGuidedAttention(self.out_channels, wind_channels=2, reduction=1)

        # self.apply(self._init_weights)

        self.finally_act = nn.Softplus()

    def forward(self, x, wind_hr=None):

        out1 = self.stage1_en(x)

        wavelet = self.wavelet_brand(x[:,0,:,:].unsqueeze(1))
        attn_1 =  self.att_1(out1)
        # print(out1.shape)
        out2 = self.stage2_en(out1)
        out2 = self.max_pool_2(out2)
        out2 = self.gs_2(out2)
        attn_2 = self.att_2(out2)

        out3 = self.stage3_en(out2)
        out3 = self.max_pool_3(out3)
        out3 = self.gs_3(out3)
        attn_3 = self.att_3(out3)

        out4 = self.stage4_en(out3)
        out4 = self.max_pool_4(out4)
        out4 = self.gs_4(out4)

        attn_4 = self.att_4(out4)

        # outb = self.bottleneck_block(out4)

        # out5 = self.stage4_de(torch.add(outb , attn_4))

        out5 = self.stage4_de(torch.add(out4, attn_4))
        out5 = self.up_4(out5)
        # print(f'out5 shape is: {out5.shape}')
        out5 = self.gs_5(out5)
        # print(f'out5 shape is: {out5.shape}')
        # print(f'attn_3 shape is: {attn_3.shape}')
        out6 = self.stage3_de(torch.add(out5 , attn_3))
        out6 = self.up_3(out6)
        out6 = self.gs_6(out6)

        out7 = self.stage2_de(torch.add(out6 , attn_2))
        out7 = self.up_2(out7)
        out7 = self.gs_7(out7)

        out8 = self.reconstruct(torch.add(torch.add(out7 , attn_1),wavelet))

        # 风场引导：在 LR 特征空间做一次引导
        out8 = self.wind_guided_att(out8, wind_hr)

        #cls
        cls_result = self.cls_brand(out8)
        # up
        up_result = self.up_brand(out8)

        # 风场引导：在 HR 输出空间做一次细化
        up_result = self.wind_guided_att_hr(up_result, wind_hr)

        reg_result = self.reg(up_result)
        reg_result = self.finally_act(reg_result)

        if self.res:
            base = F.interpolate(x[:, 0, :, :].unsqueeze(1), scale_factor=self.rs_factor, mode='bicubic')
            reg = torch.add(reg_result,base) / 2
            return reg, cls_result
        else:
            return reg_result, cls_result

