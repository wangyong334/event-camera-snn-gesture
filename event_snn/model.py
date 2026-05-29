"""
model.py — 基于 PLIF 神经元的脉冲卷积网络 (PLIF Spiking CNN)

================== 核心知识点 ==================

1) 脉冲神经元 (spiking neuron)
   普通 CNN 的神经元输出一个连续值；脉冲神经元有“膜电位 (membrane potential) V”，
   每个时间步累积输入电流，当 V 超过阈值就“发放 (fire)”一个脉冲 (输出 1)，
   然后膜电位复位 (reset)。本项目用 LIF 家族里的 PLIF。

2) LIF vs PLIF —— 这是我们的“创新点 1”
   - LIF (Leaky Integrate-and-Fire): 膜电位会按固定的“时间常数 tau”泄漏。
   - PLIF (Parametric LIF): 把 tau 变成“可学习参数”，让网络自己学“记忆该衰减多快”。
     不同层可以学到不同的 tau —— 报告里画出各层学到的 tau，就是一个直观的创新点。

3) 替代梯度 (surrogate gradient)
   “是否发放脉冲”是一个阶跃函数 (0/1)，它的导数几乎处处为 0，没法直接反向传播。
   解决办法：前向照常用阶跃函数，反向时用一个形状相近的“平滑函数的导数”来近似。
   这里用 ATan (反正切) 替代梯度，由 surrogate.ATan() 提供。

4) 多步模式 (multi-step mode, step_mode='m')
   我们一次把整段 [T, N, ...] 输入喂进网络，框架内部沿时间维 T 逐步推进神经元状态。
   这比“写 for 循环逐步喂”更快、更简洁。注意输入要把时间维 T 放在最前面。
"""

import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional


class PLIFSpikingCNN(nn.Module):
    """
    结构概览 (输入 [T, N, 2, 128, 128]):

        5 × [Conv3x3 - BN - PLIF - MaxPool2x2]      # 空间特征提取 + 时序积分
            特征图边长: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        Flatten -> Dropout -> FC -> PLIF -> Dropout -> FC -> PLIF
        输出每个时间步的脉冲 [T, N, num_classes]，再对时间维求平均 = 发放率
    """

    def __init__(self, T=16, channels=64, num_classes=11, init_tau=2.0):
        super().__init__()
        self.T = T

        # ---- 一个“造 PLIF 神经元”的小工厂，避免重复写参数 ----
        def plif():
            return neuron.ParametricLIFNode(
                init_tau=init_tau,                    # tau 的初始值
                surrogate_function=surrogate.ATan(),  # 替代梯度
                detach_reset=True,                    # 复位操作不参与反传，训练更稳
            )

        # ---- 特征提取：5 个脉冲卷积块 ----
        # 注意这里用的都是 spikingjelly.activation_based.layer 里的层，
        # 它们支持多步模式 (能正确处理 [T, N, ...] 形状)。
        # 卷积用 bias=False，因为后面紧跟 BatchNorm，bias 是多余的。
        self.conv = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels), plif(), layer.MaxPool2d(2),       # 128 -> 64

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels), plif(), layer.MaxPool2d(2),       # 64 -> 32

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels), plif(), layer.MaxPool2d(2),       # 32 -> 16

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels), plif(), layer.MaxPool2d(2),       # 16 -> 8

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels), plif(), layer.MaxPool2d(2),       # 8  -> 4
        )

        # ---- 分类头 ----
        # 用 layer.Dropout (不是 nn.Dropout)：它在所有时间步上用“同一个”随机掩码，
        # 这对 SNN 的时序一致性很重要。
        self.fc = nn.Sequential(
            layer.Flatten(),                                    # [T,N,C,4,4] -> [T,N,C*16]
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512, bias=False),
            plif(),
            layer.Dropout(0.5),
            layer.Linear(512, num_classes, bias=False),
            plif(),                                             # 输出层也用脉冲神经元
        )

        # ---- 把整个网络切到“多步模式”----
        # 这样所有层都会按 [T, N, ...] 的输入沿时间维 T 自动展开计算。
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # x: [T, N, 2, 128, 128]  (时间维在最前)
        x = self.conv(x)        # -> [T, N, channels, 4, 4]
        x = self.fc(x)          # -> [T, N, num_classes]  每个时间步发放的脉冲(0/1)
        # 对时间维 T 求平均 = 平均发放率 (firing rate)，作为分类得分。
        # 哪一类的神经元在 T 步里发放得最频繁，就预测成哪一类。
        return x.mean(dim=0)    # -> [N, num_classes]
