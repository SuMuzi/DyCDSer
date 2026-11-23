


class PG_ERA5_PrecipitationDownscaling(nn.Module):


    def __init__(self, config):
        super().__init__()

        # 多视图编码器
        self.encoder = ERA5MultiViewEncoder(dim=config['dim'])

        # 物理引导融合
        self.physics_fusion = PhysicsGuidedFusion(dim=config['dim'])

        # 低分辨率降水生成（与ERA5模式降水对比）
        self.lr_precip_head = nn.Sequential(
            nn.Conv2d(1, 1, 1),  # 直接输出或修正ERA5降水
            nn.ReLU()
        )

        # 渐进降尺度
        self.downscaler = ProgressivePrecipitationDownscaling(
            in_channels=1,
            stages=config['upscale_stages']  # [2,2,2,2] for 16×
        )

    def forward(self, era5_input, era5_precip=None, high_res_elev=None):
        """
        era5_input: dict with multi-view ERA5 data
        era5_precip: ERA5原生大尺度降水（可选，作为先验）
        high_res_elev: 高分辨率地形（可选）
        """
        # 1. 多视图编码
        multi_view = self.encoder(era5_input)

        # 2. 物理引导生成降水潜势
        lr_precip, physics_dict = self.physics_fusion(multi_view)

        # 3. 与ERA5降水融合（如果有）
        if era5_precip is not None:
            # 学习残差修正
            residual = self.lr_precip_head(era5_precip - lr_precip)
            lr_precip = era5_precip + residual

        # 4. 渐进降尺度
        hr_precip = self.downscaler(lr_precip, high_res_elev)

        return {
            'high_res_precipitation': hr_precip,
            'low_res_precipitation': lr_precip,
            'physics_variables': physics_dict
        }