import torch
import math
from torch import nn
import torch.nn.functional as F
from module.blocks import depthwise_separable_conv_block


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding_mode='reflect', bias=False)
        self.k_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding_mode='reflect', bias=False)
        self.v_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding_mode='reflect', bias=False)

        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding_mode='reflect', bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(self, *features):

        if len(features) == 1 and isinstance(features[0], list):
            features = features[0]

        batch_size = features[0].shape[0]
        H, W = features[0].shape[2:]
        num_features = len(features)

        # 计算Q、K、V
        q_list, k_list, v_list = [], [], []
        for feat in features:
            q = self.q_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            k = self.k_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            v = self.v_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        # 计算交叉注意力
        out_list = []
        for i in range(num_features):
            # 第i个特征作为查询
            q = q_list[i]
            attn_out = 0

            # 与其他所有特征交互
            for j in range(num_features):
                k = k_list[j]
                v = v_list[j]

                # 计算注意力权重
                weight = torch.matmul(q.transpose(-2, -1), k) * self.scale
                weight = F.softmax(weight, dim=-1)

                # 应用注意力
                attn_out += torch.matmul(weight, v.transpose(-2, -1))

            # 重塑并投影
            attn_out = attn_out.view(batch_size, -1, H, W)
            attn_out = self.out_proj(attn_out)
            out_list.append(attn_out)

        # 如果只有一个输入，返回单个特征
        if len(out_list) == 1:
            return out_list[0]

        return out_list


class CrossLevelAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 投影层
        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # 输出投影
        self.out_proj = nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=1)

    def forward(self, *features):
        num_levels = len(features)
        batch_size = features[0].shape[0]
        H, W = features[0].shape[2:]

        # 计算Q、K、V
        q_list, k_list, v_list = [], [], []
        for feat in features:
            q = self.q_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            k = self.k_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            v = self.v_proj(feat).view(batch_size, self.num_heads, self.head_dim, -1)
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        # 计算跨层次注意力
        out_list = []
        for i in range(num_levels):
            # 第i层作为查询，其他所有层作为键值
            q = q_list[i]
            attn_outs = []

            for j in range(num_levels):
                k = k_list[j]
                v = v_list[j]

                # 计算注意力权重
                weight = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
                weight = F.softmax(weight, dim=-1)

                # 应用注意力
                attn_out = torch.matmul(weight, v.transpose(-2, -1))
                attn_outs.append(attn_out)

            # 聚合来自所有层次的信息
            level_out = sum(attn_outs)
            out_list.append(level_out)

        # 整合结果
        out = []
        for i, level_out in enumerate(out_list):
            level_out = level_out.view(batch_size, -1, H, W)
            out.append(level_out)

        out = torch.cat(out, dim=1)
        out = self.out_proj(out)

        return out


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super(GatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 门控权重计算
        self.gate_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * num_layers, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, num_layers, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * num_layers, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, *features):
        """
        使用门控机制融合多个层次的特征

        Args:
            *features: 多个层次的特征图

        Returns:
            融合后的特征
        """
        # 拼接所有特征
        concatenated = torch.cat(features, dim=1)

        # 计算门控权重
        gates = self.gate_conv(concatenated)

        # 应用门控权重
        gated_features = []
        for i, feat in enumerate(features):
            gated = feat * gates[:, i:i + 1, :, :]
            gated_features.append(gated)

        # 再次拼接门控后的特征
        gated_concatenated = torch.cat(gated_features, dim=1)

        # 融合特征
        fused = self.fusion_conv(gated_concatenated)

        return fused


class ElementEmdNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, element_number=1):
        super(ElementEmdNet, self).__init__()
        self.embed = nn.ModuleList([
            depthwise_separable_conv_block(in_channels, out_channels)
            for _ in range(element_number)
        ])
        self.cross_attention = CrossAttention(hidden_dim=out_channels, num_heads=4)

    def forward(self, x):
        feature = []
        for i, element in enumerate(x):
            feature.append(self.embed[i](element))
        feature = self.cross_attention(feature)
        return feature


class ERA5MultiViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.circulation_inchannel = config['circulation_number']
        self.troposphere_inchannel = config['troposphere_number']
        self.boundary_inchannel = config['boundary_number']
        self.VerticalProfile_inchannel = config['VerticalProfile_number']

        # 使用ElementEmdNet处理各层次要素
        self.circulation_processor = ElementEmdNet(in_channels=1, out_channels=64,
                                                   element_number=self.circulation_inchannel)
        self.troposphere_processor = ElementEmdNet(in_channels=1, out_channels=64,
                                                   element_number=self.troposphere_inchannel)
        self.boundary_processor = ElementEmdNet(in_channels=1, out_channels=64, element_number=self.boundary_inchannel)
        self.vertical_profile_processor = ElementEmdNet(in_channels=1, out_channels=64,
                                                        element_number=self.VerticalProfile_inchannel)

        # 跨层次交叉注意力
        self.cross_attention = CrossAttention(hidden_dim=64, num_heads=4)

        # 门控融合模块
        self.gated_fusion = GatedFusion(hidden_dim=64, num_layers=4)

    def forward(self, circulation_x, troposphere_x, boundary_x, VerticalProfile_x):
        # 处理各层次要素
        feature_circulation = self.circulation_processor(circulation_x)
        feature_troposphere = self.troposphere_processor(troposphere_x)
        feature_boundary = self.boundary_processor(boundary_x)
        feature_VerticalProfile = self.vertical_profile_processor(VerticalProfile_x)

        # 跨层次交叉注意力融合
        ca_features = self.cross_attention(
            feature_circulation, feature_troposphere, feature_boundary, feature_VerticalProfile
        )

        # 门控机制融合
        fused_feature = self.gated_fusion(*ca_features)

        return fused_feature



