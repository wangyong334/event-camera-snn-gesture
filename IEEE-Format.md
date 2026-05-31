# 基于可学习时间常数脉冲神经元的事件相机图像识别

**摘要**——事件相机以异步、稀疏的事件流编码动态场景，与以离散脉冲进行时间积分的脉冲神经网络
（Spiking Neural Network, SNN）在计算范式上高度契合。本文面向 DVS128 Gesture 手势数据集，
设计了一个以参数化漏积分发放（Parametric Leaky Integrate-and-Fire, PLIF）神经元为核心的
脉冲卷积网络，采用替代梯度与时空反向传播进行端到端训练。在 11 类手势识别任务上取得 78.82%
的测试准确率，网络整体脉冲发放稀疏度达 88.07%，按简化能耗模型估计其单位计算能耗约为同规模
人工神经网络（ANN）的 1/43。实验进一步观察到各层学习到的膜时间常数随网络深度单调减小，
表明 PLIF 能够自适应地为不同层分配时间尺度。

**关键词**——事件相机；脉冲神经网络；PLIF；替代梯度；手势识别；低功耗

---

## 一、引言

传统帧式相机以固定帧率同步采样，在高速运动与高动态范围场景下存在运动模糊、数据冗余和功耗
偏高等问题。事件相机（event camera / Dynamic Vision Sensor, DVS）则在每个像素独立、异步地
响应亮度对数变化，输出形如 (x, y, t, p) 的事件：x、y 为像素坐标，t 为微秒级时间戳，
p∈{+1,−1} 为亮度增减的极性。这种成像方式具有微秒级时间分辨率、高动态范围与低数据冗余的
优势，特别适合捕捉随时间展开的动作。

与事件数据天然匹配的计算模型是脉冲神经网络。SNN 的神经元通过膜电位对输入电流进行时间积分，
当膜电位越过阈值时发放离散脉冲，其事件驱动、稀疏的特性使其在神经形态硬件上能以极低功耗运行。
然而，脉冲发放是不可微的阶跃过程，长期制约了 SNN 的高效训练；近年来替代梯度（surrogate
gradient）方法使 SNN 可以像 ANN 一样进行端到端的梯度训练，显著提升了其性能。

本文工作可概括为：（1）在 DVS128 Gesture 手势数据集上，将异步事件积分为多时间步帧表示，
构建以 PLIF 神经元为核心的脉冲卷积网络；（2）采用 ATan 替代梯度结合时空反向传播
（Backpropagation Through Time, BPTT）实现端到端训练；（3）对模型的识别精度、脉冲发放
稀疏度、各层学习到的时间常数以及理论能耗进行定量分析，验证“事件相机 + SNN”在时序建模与
低功耗方面的优势。

## 二、技术方法

### 2.1 事件表示

设一段手势记录的事件集合为 {(x_i, y_i, t_i, p_i)}。本文按事件数量将其沿时间均分为 T 个
互不重叠的时间片，并在每个时间片内按极性把事件累积到 2 通道的二维网格上，得到第 t 个时间步的
帧 X[t]∈R^{2×128×128}。最终每条样本表示为张量序列 {X[1],…,X[T]}（本文取 T=16）。
该表示在保留事件时序信息的同时，使数据可被卷积层高效处理。

### 2.2 PLIF 脉冲神经元

普通的漏积分发放（LIF）神经元在离散时间下的动力学可写为：

- 充电： H[t] = V[t−1] + (1/τ)·( X[t] − (V[t−1] − V_reset) )
- 发放： S[t] = Θ( H[t] − V_th )，其中 Θ(·) 为阶跃函数
- 复位： V[t] = H[t]·(1 − S[t]) + V_reset·S[t]

其中 V 为膜电位，V_th 为发放阈值，τ 为膜时间常数，控制膜电位“遗忘”历史的速度。

在标准 LIF 中 τ 为人工设定的固定超参数。本文采用参数化 LIF（PLIF），将衰减系数表示为
可学习参数 w 的 Sigmoid 映射，即 1/τ = sigmoid(w)，从而 τ = 1/sigmoid(w)。如此 τ 被纳入
反向传播一同优化，使每一层都能自适应地学习最合适的时间尺度。

### 2.3 网络架构

整体为多步（multi-step）模式下的脉冲卷积网络，输入形状为 [T, 2, 128, 128]：

```
输入 [T,2,128,128]
 → 5 × ( Conv3×3 → BatchNorm → PLIF → MaxPool2×2 )   特征图 128→64→32→16→8→4
 → Flatten → Dropout → FC(→512) → PLIF → Dropout → FC(→11) → PLIF
 → 输出脉冲序列 [T,11]
```
五个卷积块逐级提取空间特征并通过 PLIF 完成时间积分；批归一化（BatchNorm）稳定训练；
最大池化逐步降低空间分辨率。分类头将特征展平后经两层全连接与 PLIF 输出 11 个类别对应的
脉冲序列。卷积层不使用偏置（其作用由后接的 BatchNorm 吸收），Dropout 在所有时间步共享同一
随机掩码以保持时序一致性。

## 三、训练方法

### 3.1 读出与损失函数

设输出层在 T 个时间步上的脉冲为 S_out[t]∈R^{11}。对时间维取平均得到发放率向量
o = (1/T)·Σ_{t=1}^{T} S_out[t]，以其作为各类别的得分，使用交叉熵损失：

L = CrossEntropy( o, y )

其中 y 为真实类别标签。该读出方式鼓励正确类别神经元在整段时间内更频繁地发放。

### 3.2 替代梯度与时空反向传播

发放函数 Θ(·) 的导数几乎处处为零，无法直接反传。训练时前向仍使用阶跃函数，反向时以平滑函数
的导数近似其梯度，本文采用反正切（ATan）替代梯度，其形式为

∂S/∂H ≈ (α/2) / ( 1 + ( (π/2)·α·(H − V_th) )² )

α 为控制陡峭程度的超参数。整个网络沿时间步展开为计算图，梯度同时沿“层”与“时间”两个方向回传，
即时空反向传播（BPTT），由此可一并优化卷积/全连接权重以及各 PLIF 层的时间常数参数 w。

### 3.3 优化设置

使用 Adam 优化器（初始学习率 1e−3）配合余弦退火学习率调度；为节省显存并加速，启用自动混合
精度（AMP）。每个 mini-batch 前向后调用膜电位复位，避免神经元状态在样本间泄漏。本文配置为
通道数 64、时间步 T=16、训练 64 个 epoch。

## 四、实验

### 4.1 数据集

DVS128 Gesture（IBM）由 DVS128 事件相机录制，包含来自 29 位受试者、3 种光照条件下的 11 类
手势，官方按受试者划分为 1176 个训练样本与 288 个测试样本（训练与测试受试者不重叠，更能反映
泛化能力）。每条样本经 2.1 节方法积分为 [16, 2, 128, 128] 的帧序列。

### 4.2 实验环境

实现基于 PyTorch 与开源 SNN 框架 SpikingJelly。硬件为 NVIDIA RTX 5060（8GB 显存）；
由于该显卡属 Blackwell 架构，需使用 CUDA 12.8 版本的 PyTorch。

### 4.3 实验结果

在测试集上取得 **78.82%** 的 11 分类准确率。脉冲稀疏性与时间常数分析结果如下：

| 指标 | 数值 |
|------|------|
| 测试准确率 | 78.82% |
| 各 PLIF 层 τ | 1.97 / 2.01 / 1.95 / 1.61 / 1.48 / 1.59 / 1.19 |
| 各层平均发放率 | 0.021 / 0.041 / 0.040 / 0.049 / 0.130 / 0.418 / 0.136 |
| 整体平均发放率 | 0.1193（稀疏度 88.07%） |
| 能耗比 E(SNN)/E(ANN) | 0.0233（≈ 1/43） |

两点观察：（1）各层学习到的膜时间常数 τ 随深度大体单调减小（约 2.0 降至 1.19），说明浅层
倾向于长时积累缓慢变化、深层倾向于快速响应，体现了 PLIF 自适应分配时间尺度的能力；
（2）卷积层发放率仅 2%~5%，网络整体保持约 88% 的稀疏度。基于 45nm CMOS 工艺的简化能耗模型
（一次乘加 MAC≈4.6 pJ，一次脉冲累加 AC≈0.9 pJ），SNN 仅在脉冲发放时触发计算，其单位
连接·时间步的能耗约为同规模 ANN 的 1/43，定量印证了低功耗优势。

代码、配置与结果图已开源：
**https://github.com/wangyong334/event-camera-snn-gesture**

## 五、结论与未来工作

本文构建并训练了一个面向事件相机手势识别的 PLIF 脉冲卷积网络，在 DVS128 Gesture 上达到
78.82% 的准确率，同时保持约 88% 的发放稀疏度与约 1/43 的相对能耗，验证了事件相机与脉冲神经
网络结合在时序建模和低功耗上的潜力；各层自适应学习到的时间常数也为 PLIF 的有效性提供了直观
证据。

受单卡显存与训练时长限制，本文采用了较小的通道数与较少的训练轮数。未来可从以下方向改进：
（1）增大网络通道数、延长训练并引入数据增广以提升精度；（2）引入时空注意力或 TET 等
更先进的训练目标；（3）探索更具信息量的事件表示（如时间表面、体素网格）；（4）在神经形态
硬件上部署并实测真实能耗，以进一步量化能效优势。

## 参考文献

[1] G. Gallego et al., "Event-based Vision: A Survey," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 1, pp. 154–180, 2022.

[2] A. Amir et al., "A Low Power, Fully Event-Based Gesture Recognition System," in *Proc. IEEE CVPR*, 2017, pp. 7243–7252.

[3] W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, and Y. Tian, "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks," in *Proc. IEEE/CVF ICCV*, 2021, pp. 2661–2671.

[4] E. O. Neftci, H. Mostafa, and F. Zenke, "Surrogate Gradient Learning in Spiking Neural Networks," *IEEE Signal Process. Mag.*, vol. 36, no. 6, pp. 51–63, 2019.

[5] Y. Wu, L. Deng, G. Li, J. Zhu, and L. Shi, "Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks," *Front. Neurosci.*, vol. 12, art. 331, 2018.

[6] W. Fang et al., "SpikingJelly: An Open-Source Machine Learning Infrastructure Platform for Spike-Based Intelligence," *Sci. Adv.*, vol. 9, no. 40, eadi1480, 2023.

[7] M. Horowitz, "1.1 Computing's Energy Problem (and What We Can Do About It)," in *Proc. IEEE Int. Solid-State Circuits Conf. (ISSCC)*, 2014, pp. 10–14.
