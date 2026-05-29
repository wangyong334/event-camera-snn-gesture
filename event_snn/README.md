# 事件相机手势识别 · PLIF 脉冲神经网络 (SNN)

机器视觉课程作业 —— **视觉算法应用**。
用事件相机数据集 **DVS128 Gesture**，设计并训练一个基于 **PLIF 神经元**的脉冲卷积网络
(Spiking CNN)，完成 11 类手势识别，并分析其稀疏性 / 能耗优势。

---

## 1. 算法脉络（对应两页 PDF）

```
事件流 (x,y,t,p)
  → 按时间切成 T=16 个时间片，每片累积成 [2,128,128] 帧   ← 事件表示
  → 5 × [Conv3x3 - BN - PLIF - MaxPool]                    ← 脉冲卷积特征提取
  → Flatten → FC → PLIF → FC → PLIF
  → 对 T 步发放率取平均 → 11 类分类
训练: 替代梯度 (surrogate gradient) + 时空反向传播 (BPTT)
```

**两个创新点：**
1. **PLIF 可学习神经元**：膜时间常数 `tau` 作为可训练参数，各层自动学习不同的时间尺度。
2. **稀疏度 / 能耗分析**：统计各层发放率，估算 SNN 相对等价 ANN 的能耗优势。

| 文件 | 作用 |
|------|------|
| `data.py` | 加载 DVS128 Gesture，把异步事件积分成 `[T,2,128,128]` 帧 |
| `model.py` | PLIF 脉冲卷积网络结构 |
| `train.py` | 训练循环（替代梯度 + BPTT、混合精度、保存最优模型） |
| `analyze.py` | 发放率/能耗/tau 分析，生成 `analysis.png` |

---

## 2. 环境安装（Windows + RTX 5060 8GB）

> ⚠️ **RTX 50 系列(Blackwell)显卡必须用 CUDA 12.8 版 PyTorch**，否则会报
> `no kernel image is available` 之类的错误。

```powershell
# 1) 建议建一个独立环境 (用 conda 或 venv 都行)
conda create -n snn python=3.11 -y
conda activate snn

# 2) 装 CUDA 12.8 版 PyTorch (RTX 5060 关键步骤！)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) 装其余依赖
pip install -r requirements.txt
```

验证 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
应输出类似 `2.7.x True NVIDIA GeForce RTX 5060`。

---

## 3. 下载数据集（必须手动）

DVS128 Gesture 因版权无法自动下载，需手动操作：

1. 打开 IBM 官方页面 <https://research.ibm.com/interactive/dvsgesture/>
   （或搜索 “DvsGesture dataset”），下载 **`DvsGesture.tar.gz`**。
2. 在你选定的数据根目录下新建 `download` 子目录，把压缩包放进去：

   ```
   D:\datasets\DVS128Gesture\
   └── download\
       └── DvsGesture.tar.gz
   ```
3. 不用自己解压。第一次运行 `train.py` 时，SpikingJelly 会自动解压并把事件积分成帧
   （会生成 `frames_number_16_split_by_number` 缓存目录）。
   **首次运行较慢**（几分钟到十几分钟），之后再跑就很快。

---

## 4. 训练

```powershell
python train.py --data-root D:\datasets\DVS128Gesture --amp
```

- `--amp` 开启混合精度，**8GB 显存强烈建议加上**（省显存、提速）。
- 若仍报显存不足 (CUDA out of memory)，依次尝试：
  `--batch-size 4`，再不行 `--T 12`，或 `--channels 32`。
- Windows 上若 DataLoader 报多进程错误，加 `--num-workers 0`。

训练完成后，最优权重保存在 `./checkpoints/best.pth`。
预期测试准确率约 **90%~95%+**（与 epochs、channels 有关）。

---

## 5. 分析（生成报告用的图）

```powershell
python analyze.py --data-root D:\datasets\DVS128Gesture --ckpt ./checkpoints/best.pth
```

输出：
- 终端打印各层 **tau**、各层 **平均发放率**、整体 **稀疏度**、**粗略能耗比**。
- 生成 `analysis.png`（tau 分布 + 各层发放率柱状图），可直接放进两页 PDF。

---

## 6. 常见名词速查（方便看代码注释）

| 术语 | 含义 |
|------|------|
| LIF | Leaky Integrate-and-Fire，漏积分发放神经元 |
| PLIF | Parametric LIF，时间常数可学习的 LIF |
| tau | 膜时间常数，决定膜电位“遗忘”得多快 |
| surrogate gradient | 替代梯度，让不可导的发放函数能反向传播 |
| BPTT | 时空反向传播，沿时间步展开求梯度 |
| firing rate | 发放率，神经元输出脉冲的平均频率 |
| step_mode='m' | 多步模式，一次处理整段 `[T,N,...]` 输入 |
