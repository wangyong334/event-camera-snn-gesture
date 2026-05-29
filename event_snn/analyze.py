"""
analyze.py — SNN 特性分析 (创新点 2: 稀疏度 / 能耗 / 可学习 tau)

================== 为什么要做这个分析 ==================
SNN + 事件相机最有说服力的卖点是“稀疏、低功耗”：
  - 大多数时间步、大多数神经元都“不发放”(输出 0)，
  - 只有真正有信息的地方才发放脉冲 (输出 1)。
这种稀疏性让 SNN 在专用神经形态芯片上极其省电。

这个脚本做三件事，正好对应你报告里的“实验结果”：
  1) 统计每层 PLIF 神经元的平均发放率 (firing rate) 和整体稀疏度。
  2) 用简化能耗模型，粗略估计 SNN 相对等价 ANN 的能耗优势。
  3) 取出每层学到的 tau，画图 —— 直观展示 PLIF 的“可学习时间常数”。

运行示例：
    python analyze.py --data-root D:\datasets\DVS128Gesture --ckpt ./checkpoints/best.pth
"""

import argparse

import torch
import matplotlib.pyplot as plt

from spikingjelly.activation_based import functional, neuron

from data import get_dvs128_loaders
from model import PLIFSpikingCNN


def parse_args():
    p = argparse.ArgumentParser(description='Analyze a trained PLIF Spiking CNN')
    p.add_argument('--data-root', required=True)
    p.add_argument('--ckpt', default='./checkpoints/best.pth', help='训练好的权重')
    p.add_argument('--T', type=int, default=16)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--num-batches', type=int, default=20, help='用多少个 batch 来统计发放率')
    p.add_argument('--out', default='analysis.png', help='分析图保存文件名')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 加载模型与权重 ----
    net = PLIFSpikingCNN(T=args.T, channels=args.channels).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # ---- (3) 取出每层学到的 tau ----
    # PLIF 的可学习参数叫 w，膜电位衰减系数 = sigmoid(w)，而 tau = 1 / sigmoid(w)。
    taus = []
    for m in net.modules():
        if isinstance(m, neuron.ParametricLIFNode):
            tau = (1.0 / torch.sigmoid(m.w.detach())).item()
            taus.append(tau)
    print('各 PLIF 层学到的 tau:', [round(t, 3) for t in taus])

    # ---- (1) 用前向钩子 (forward hook) 记录每层的发放率 ----
    # 钩子会在每层 PLIF 输出脉冲后被调用，我们把脉冲张量的均值(=发放率)记下来。
    firing_rates = {}                      # 层名 -> [每个batch的发放率...]
    hook_handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            # out 是脉冲张量 (0/1)，形状 [T, N, ...]，均值就是该层平均发放率
            firing_rates.setdefault(name, []).append(out.detach().float().mean().item())
        return hook

    idx = 0
    for m in net.modules():
        if isinstance(m, neuron.ParametricLIFNode):
            hook_handles.append(m.register_forward_hook(make_hook(f'PLIF_{idx}')))
            idx += 1

    # ---- 跑整个测试集: 顺便统计准确率, 同时由钩子收集发放率 ----
    _, test_loader = get_dvs128_loaders(
        args.data_root, T=args.T, batch_size=args.batch_size, num_workers=args.num_workers)

    correct, total = 0, 0
    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(device).transpose(0, 1)   # [N,T,...] -> [T,N,...]
            label = label.to(device)
            out_fr = net(frame)
            correct += (out_fr.argmax(dim=1) == label).sum().item()
            total += label.size(0)
            functional.reset_net(net)                  # 别忘了复位

    for h in hook_handles:                             # 用完移除钩子
        h.remove()

    test_acc = correct / total
    print(f'\n测试集准确率 = {test_acc:.4f}  ({correct}/{total})')

    # 每层平均发放率
    layer_fr = {k: sum(v) / len(v) for k, v in firing_rates.items()}
    overall_fr = sum(layer_fr.values()) / len(layer_fr)
    print('\n各层平均发放率 (越小越稀疏):')
    for k, v in layer_fr.items():
        print(f'  {k}: {v:.4f}')
    print(f'整体平均发放率 = {overall_fr:.4f}  ->  稀疏度 = {1 - overall_fr:.4f}')

    # ---- (2) 简化能耗估计 ----
    # 文献常用的 45nm CMOS 单次操作能耗 (单位 pJ):
    #   ANN: 每次乘加运算 (MAC) 约 4.6 pJ
    #   SNN: 每次脉冲触发的累加 (AC) 约 0.9 pJ
    # 关键差别: ANN 每个连接每步都要算; SNN 只有“发放”时才触发计算 (∝ 发放率)。
    # 这里只做“每连接每步”的粗略比值，严谨估计需按各层 FLOPs 加权 (报告里可注明)。
    E_MAC, E_AC = 4.6, 0.9
    energy_ratio = (E_AC * overall_fr) / E_MAC
    print(f'\n粗略能耗比 (SNN / ANN, 每连接每步) ≈ {energy_ratio:.4f}  '
          f'(越小越省电)')

    # ---- 画图：tau 分布 + 各层发放率 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(range(len(taus)), taus, color='#4C72B0')
    axes[0].set_title('Learned tau per PLIF layer')
    axes[0].set_xlabel('PLIF layer index')
    axes[0].set_ylabel('tau')

    names = list(layer_fr.keys())
    axes[1].bar(range(len(names)), [layer_fr[n] for n in names], color='#C44E52')
    axes[1].set_title('Average firing rate per layer')
    axes[1].set_xlabel('PLIF layer index')
    axes[1].set_ylabel('firing rate')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'\n图已保存到 {args.out} (可直接放进两页 PDF)')


if __name__ == '__main__':
    main()
