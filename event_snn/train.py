"""
train.py — 训练脚本 (替代梯度 + 时空反向传播 BPTT)

================== 训练流程在讲什么 ==================
SNN 的训练本质是“按时间展开的反向传播”，叫做 BPTT
(Backpropagation Through Time, 时空反向传播)：
  - 前向：输入 [T,N,...] 沿 T 步推进，神经元逐步积分、发放，得到各步脉冲。
  - 反向：把 T 步看成一个“展开的计算图”，梯度沿时间和层两个方向回传；
          遇到不可导的发放函数时，用替代梯度 (surrogate gradient) 近似。
PyTorch 的 autograd + SpikingJelly 的替代梯度神经元会自动完成这一切，
我们只需照常 loss.backward()。

两个 SNN 专属的“坑”，代码里都标了出来：
  (1) 每个 batch 训练完，必须 reset_net() 复位所有神经元膜电位，
      否则上一批样本的状态会泄漏到下一批。
  (2) DataLoader 给的形状是 [N, T, ...]，要 transpose 成 [T, N, ...] 再喂网络。

运行示例 (Windows PowerShell)：
    python train.py --data-root D:\datasets\DVS128Gesture --amp
显存只有 8GB 时建议加 --amp (混合精度，省显存)；若仍 OOM，把 --batch-size 调到 4。
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from spikingjelly.activation_based import functional

from data import get_dvs128_loaders
from model import PLIFSpikingCNN


def parse_args():
    p = argparse.ArgumentParser(description='Train PLIF Spiking CNN on DVS128 Gesture')
    p.add_argument('--data-root', required=True, help='数据集根目录(含 download/DvsGesture.tar.gz)')
    p.add_argument('--T', type=int, default=16, help='时间步数 (事件切成几片)')
    p.add_argument('--batch-size', type=int, default=8, help='批大小; 8GB 显存建议 8, 不够改 4')
    p.add_argument('--channels', type=int, default=64, help='卷积通道数; 想要更高精度可调到 128')
    p.add_argument('--epochs', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num-workers', type=int, default=4, help='读数据进程数; Windows 报错就设 0')
    p.add_argument('--out-dir', default='./checkpoints', help='模型保存目录')
    p.add_argument('--amp', action='store_true', help='开启混合精度训练 (省显存、提速)')
    return p.parse_args()


def run_one_epoch(net, loader, optimizer, scaler, device, amp, train):
    """跑一个 epoch。train=True 时训练并更新参数，否则只评估。返回 (平均loss, 准确率)。"""
    net.train(train)
    total_loss, total_correct, total_num = 0.0, 0, 0

    # 评估时不需要梯度，用 no_grad 省显存提速
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for frame, label in tqdm(loader, desc='train' if train else 'eval ', leave=False):
            # frame: [N, T, 2, 128, 128]  ->  [T, N, 2, 128, 128] (多步模式要时间维在前)
            frame = frame.to(device, non_blocking=True).transpose(0, 1)
            label = label.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            # autocast: 前向用 float16 算，省显存提速 (开了 --amp 才生效)
            with autocast(device_type='cuda', enabled=amp):
                out_fr = net(frame)                      # [N, num_classes] 发放率
                loss = F.cross_entropy(out_fr, label)    # 交叉熵分类损失

            if train:
                # ===== BPTT 反向传播 + 替代梯度，由框架自动完成 =====
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # ===== 关键(1): 每个 batch 后复位所有神经元的膜电位 =====
            functional.reset_net(net)

            total_loss += loss.item() * label.size(0)
            total_correct += (out_fr.argmax(dim=1) == label).sum().item()
            total_num += label.size(0)

    return total_loss / total_num, total_correct / total_num


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'设备: {device} | T={args.T} batch={args.batch_size} channels={args.channels}')

    # ---- 数据 ----
    train_loader, test_loader = get_dvs128_loaders(
        args.data_root, T=args.T, batch_size=args.batch_size, num_workers=args.num_workers)

    # ---- 模型 / 优化器 / 学习率调度 / 混合精度 ----
    net = PLIFSpikingCNN(T=args.T, channels=args.channels).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # 余弦退火: 学习率随 epoch 平滑下降到接近 0，收尾更稳
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device='cuda', enabled=args.amp)

    # ---- 训练主循环 ----
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = run_one_epoch(
            net, train_loader, optimizer, scaler, device, args.amp, train=True)
        scheduler.step()
        test_loss, test_acc = run_one_epoch(
            net, test_loader, optimizer, scaler, device, args.amp, train=False)

        print(f'epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} '
              f'| test_loss {test_loss:.4f} acc {test_acc:.4f}')

        # 保存当前最优模型 (按测试集准确率)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.out_dir, 'best.pth'))
            print(f'    >> 新最优 test_acc={best_acc:.4f}, 已保存 best.pth')

    print(f'训练结束，最佳测试准确率 = {best_acc:.4f}')


# Windows 上用多进程 DataLoader 必须有这个保护，否则会报错 / 反复启动
if __name__ == '__main__':
    main()
