# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述 (Project)

机器视觉课程作业「视觉算法应用」。用事件相机数据集 **DVS128 Gesture**，设计并训练一个
基于 **PLIF 神经元**的脉冲神经网络 (Spiking CNN)，完成 11 类手势识别，并分析其稀疏性 /
能耗优势。与作者研究方向（事件相机的脉冲神经网络）相关，最终产出包括代码和一份两页 PDF
（按创新性 + 实操性评分）。

代码主体在 `event_snn/` 目录。框架使用 **SpikingJelly** (PyTorch 生态)。

## 约定 (Conventions)

- **代码注释一律使用中文**（关键技术名词可中英并列，如「替代梯度 (surrogate gradient)」）。
  作者是第一次写完整 SNN 项目，注释偏教学风格，解释每个模块的 SNN 概念。
- 目标硬件：Windows + NVIDIA RTX 5060 8GB + 32GB 内存。

## 架构 (Architecture)

数据流：
```
事件流 (x,y,t,p)
  → 切成 T=16 个时间片，每片累积成 [2,128,128] 帧        (data.py，事件表示)
  → 5 × [Conv3x3 - BN - PLIF - MaxPool]                  (model.py，脉冲卷积特征提取)
  → Flatten → FC → PLIF → FC → PLIF
  → 对 T 步发放率取平均 → 11 类分类
训练：替代梯度 (surrogate gradient) + 时空反向传播 (BPTT)
```

| 文件 | 作用 |
|------|------|
| `event_snn/data.py` | 加载 DVS128 Gesture，把异步事件积分成 `[T,2,128,128]` 帧 |
| `event_snn/model.py` | `PLIFSpikingCNN`，全网络在 `__init__` 末尾切到多步模式 `step_mode='m'` |
| `event_snn/train.py` | 训练循环（替代梯度+BPTT、AMP 混合精度、按测试准确率存 `best.pth`） |
| `event_snn/analyze.py` | 发放率/能耗/tau 分析，生成 `analysis.png` |

**两个创新点**：① PLIF 可学习膜时间常数 `tau`；② 稀疏度 / 能耗分析。

**SNN 易错点（改代码时务必注意）**：
- 多步模式下网络输入形状是 `[T, N, ...]`（时间维在前）；DataLoader 给的是 `[N, T, ...]`，
  训练/分析里都要 `transpose(0, 1)`。
- **每个 batch 前向后必须调用 `functional.reset_net(net)`** 复位膜电位，否则状态会跨 batch 泄漏。
- PLIF 的 `tau` 不是直接存的：可学习参数是 `w`，`tau = 1 / sigmoid(w)`。

## 环境与命令 (Setup & Commands)

```powershell
# RTX 50 系(Blackwell)显卡必须装 CUDA 12.8 版 PyTorch，否则跑不起来
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r event_snn/requirements.txt

# 训练（8GB 显存务必加 --amp；OOM 时调小 --batch-size / --T / --channels）
python event_snn/train.py --data-root <数据根目录> --amp

# 分析（生成报告用的 analysis.png）
python event_snn/analyze.py --data-root <数据根目录> --ckpt ./checkpoints/best.pth
```

**数据集需手动下载**：DVS128 Gesture 因版权无法自动下载。从 IBM 官网下载
`DvsGesture.tar.gz`，放到 `<数据根目录>/download/` 下（不用解压，首次运行 `train.py`
会自动解压并积分成帧，首次较慢并缓存）。

目前没有测试框架；如需添加，遵循上述中文注释约定。
