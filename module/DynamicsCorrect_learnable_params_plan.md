# DynamicsCorrect.py 校正因子可学习化修改方案

> 说明：本文档仅提供修改方案，**不改动原代码**。返岗后按此文档在 `module/DynamicsCorrect.py` 中实施即可。

---

## 一、目标

将 `DynamicsCorrect.py` 中目前写死的校正相关固定参数改为 `nn.Parameter`，让网络在训练过程中自动学习这些参数，从而替代人工经验系数。

---

## 二、需要改动的固定参数

| 位置 | 原代码 | 含义 | 建议改法 |
|---|---|---|---|
| `WindErrorExtractor.__init__` | `self.div_x.weight = nn.Parameter(div_kernel_x, requires_grad=False)` | 散度/涡度计算核 | `requires_grad=True` |
| `WindErrorExtractor.__init__` | `self.div_y.weight = nn.Parameter(div_kernel_y, requires_grad=False)` | 散度/涡度计算核 | `requires_grad=True` |
| `WindGuidedPrecipitationAdjustment.forward` | `torch.exp(-div_error * 2.0)` | 散度误差到物理修正因子的映射系数 | `nn.Parameter(torch.tensor(2.0))` |
| `WindGuidedPrecipitationAdjustment.forward` | `1.0 + 0.5 * intensity_gate * intensity_delta + 0.5 * (phys_correction - 1.0)` | 数据驱动项权重 `0.5` | `nn.Parameter(torch.tensor(0.5))` |
| `WindGuidedPrecipitationAdjustment.forward` | `1.0 + 0.5 * intensity_gate * intensity_delta + 0.5 * (phys_correction - 1.0)` | 物理驱动项权重 `0.5` | `nn.Parameter(torch.tensor(0.5))` |

> 注：`torch.clamp(..., 0.5, 2.0)` 和 `torch.clamp(..., 0.3, 3.0)` 的上下界是安全边界，建议保持固定；若也想学习，需要额外约束 `min < max` 且边界本身在反向传播中几乎得不到梯度，收益不大。

---

## 三、具体修改代码

### 3.1 `WindErrorExtractor` 微分核可学习

```python
class WindErrorExtractor(nn.Module):
    def __init__(self, in_ch: int = 2, learnable_kernels: bool = True):
        super().__init__()

        # 散度计算核
        self.div_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.div_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        div_kernel_x = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]).view(1, 1, 3, 3)
        div_kernel_y = torch.tensor([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]]).view(1, 1, 3, 3)
        self.div_x.weight = nn.Parameter(div_kernel_x, requires_grad=learnable_kernels)
        self.div_y.weight = nn.Parameter(div_kernel_y, requires_grad=learnable_kernels)

        # 涡度计算核
        self.vort_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.vort_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.vort_x.weight = nn.Parameter(div_kernel_x, requires_grad=learnable_kernels)
        self.vort_y.weight = nn.Parameter(div_kernel_y, requires_grad=learnable_kernels)

        # 特征编码（保持不变）
        self.encoder = nn.Sequential(
            depthwise_separable_conv_block(4, 64),
            depthwise_separable_conv_block(64, 128),
            nn.Conv2d(128, 64, 1),
        )
```

### 3.2 `WindGuidedPrecipitationAdjustment` 增加可学习参数

在 `__init__` 末尾加入：

```python
        # 可学习校正因子（原 hard-coded 经验系数）
        # exp_coeff: 控制散度误差到物理修正因子的映射陡峭程度
        self.exp_coeff = nn.Parameter(torch.tensor(2.0))
        # data_weight / phys_weight: 数据驱动项与物理驱动项的融合权重
        self.data_weight = nn.Parameter(torch.tensor(0.5))
        self.phys_weight = nn.Parameter(torch.tensor(0.5))
```

在 `forward` 中替换对应行：

```python
        # 物理关系：负散度误差（辐合不足）→ 增加降水
        phys_correction = torch.exp(-div_error * self.exp_coeff)  # 可学习映射系数
        phys_correction = torch.clamp(phys_correction, 0.5, 2.0)

        # 门控融合：数据驱动修正 + 物理驱动修正（权重可学习）
        correction_factor = 1.0 + self.data_weight * intensity_gate * intensity_delta \
                            + self.phys_weight * (phys_correction - 1.0)
        correction_factor = torch.clamp(correction_factor, 0.3, 3.0)  # 安全范围
```

### 3.3（可选）`intensity_corrector` 中硬编码通道数修复

原代码：

```python
nn.Conv2d(64, 1, 1),  # 当 wind_feat_dim != 64 时会报错
```

建议改为：

```python
nn.Conv2d(wind_feat_dim, 1, 1),
```

当前 `MultiVariableFusionWithWindCorrection` 中传入的 `wind_feat_dim=64`，运行不受影响，但改成变量更稳健。

---

## 四、参数是否会自动被优化器收集？

会。只要这些 `nn.Parameter` 是通过 `nn.Module` 的属性挂载（即 `self.xxx`），调用 `model.parameters()` 时就会自动包含，无需在 `train_all.py` 中额外添加。

可以通过下面代码验证：

```python
model = MultiVariableFusionWithWindCorrection(...)
learnable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(learnable_names)
```

应能看到类似：

```
wind_error_extractor.div_x.weight
wind_error_extractor.div_y.weight
wind_error_extractor.vort_x.weight
wind_error_extractor.vort_y.weight
wind_adjustment.exp_coeff
wind_adjustment.data_weight
wind_adjustment.phys_weight
```

---

## 五、训练注意事项

1. **初始化保留物理意义**：`exp_coeff` 初始化为 `2.0`，`data_weight` / `phys_weight` 初始化为 `0.5`，训练初期与原版行为一致，训练稳定性更好。
2. **学习率建议**：这些标量参数对学习率比较敏感。如果主网络学习率是 `1e-3`，可以考虑给这些标量参数单独设置 `1e-2` 或 `1e-3`；或者直接用默认优化器一起训练观察效果。
3. **梯度稳定性**：`exp_coeff` 参与指数运算，数值容易爆炸。建议在训练过程中打印 `exp_coeff` 的取值，必要时在 `forward` 中做 `torch.clamp(self.exp_coeff, 0.1, 5.0)`。
4. **微分核可学习化**：Sobel 核初始值具有明确的物理意义，开启 `requires_grad=True` 后网络会在其基础上微调。如果发现训练发散，可先关闭微分核学习（`learnable_kernels=False`），只学习 `exp_coeff` 和融合权重。

---

## 六、最小验证脚本

返岗后修改完代码，可用以下脚本快速验证：

```python
import torch
from module.DynamicsCorrect import MultiVariableFusionWithWindCorrection

model = MultiVariableFusionWithWindCorrection(
    precip_channels=1,
    other_channels=17,
    wind_channels=2,
    base_dim=64
)

print("新增可学习参数：")
for name, param in model.named_parameters():
    if 'exp_coeff' in name or 'data_weight' in name or 'phys_weight' in name or 'div_' in name or 'vort_' in name:
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 简单前向
B, H, W = 2, 64, 64
precip = torch.rand(B, 1, H, W)
other = torch.rand(B, 17, H, W)
wind_model = torch.rand(B, 2, H, W)
wind_obs = torch.rand(B, 2, H, W)

out = model(precip, other, wind_model, wind_obs, return_adjustments=True)
print("precipitation shape:", out['precipitation'].shape)
print("adjustments keys:", out['adjustments'].keys())
```

---

## 七、改动范围总结

- 仅修改 `module/DynamicsCorrect.py`
- 不需要改 `trainer/train_all.py` 或 `main_all_elements.py`
- 不需要改损失函数、数据加载器或配置文件

---

*文档生成时间：2026-06-29*  
*对应源码：`module/DynamicsCorrect.py`*
