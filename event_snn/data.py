"""
data.py — DVS128 Gesture 数据集加载 + “异步事件 -> 帧”转换

================== 这个模块在讲什么 ==================
事件相机 (event camera / DVS) 输出的不是普通图像帧，而是一连串异步事件：
    每个事件 = (x, y, t, polarity)
    - x, y      : 像素坐标
    - t         : 该像素亮度变化发生的精确时间 (微秒级)
    - polarity  : 极性，+1 表示变亮，-1 表示变暗

SNN 是“按时间步 (time step) 一步步”处理输入的，所以我们要把一段连续的
事件流，沿时间轴切成 T 个时间片 (time slice)。每个时间片内的所有事件，
按 (极性, y, x) 累积到一张 2 通道的“帧”里：
    通道 0 = 这段时间内该像素的“变亮”事件个数
    通道 1 = 这段时间内该像素的“变暗”事件个数

于是每个样本最终变成张量：  [T, 2, 128, 128]
其中 T 就是 SNN 仿真的总时间步数。

这一步“事件 -> 帧”的积分由 SpikingJelly 的 data_type='frame' 自动完成，
我们不用自己写积分循环。

================== 注意：数据集要手动下载 ==================
DVS128 Gesture 因为版权原因 *不能* 自动下载，需要你手动下载后放到指定目录。
详细步骤见 README.md。
"""

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader


def get_dvs128_loaders(root, T=16, batch_size=8, num_workers=4):
    """
    构建 DVS128 Gesture 的训练 / 测试 DataLoader。

    参数：
        root        : 数据集根目录 (里面要有 download/DvsGesture.tar.gz)
        T           : 时间步数，也就是把事件切成几片 (= SNN 的仿真步数)
        batch_size  : 批大小。8GB 显存建议 8；若显存不足改 4
        num_workers : 读数据的进程数 (Windows 上若报错可设为 0)

    返回： (train_loader, test_loader)
    """

    # data_type='frame'  : 把异步事件积分成“帧”
    # frames_number=T    : 切成 T 个时间片
    # split_by='number'  : 按“事件数量”均分 —— 每个时间片包含的事件数大致相等。
    #                      另一种选择是 'time'(按时间均分)，但运动快慢不均时
    #                      'number' 通常更稳定。
    #
    # 第一次运行会比较慢：SpikingJelly 要把全部录制的事件积分成帧并缓存到磁盘
    # (缓存目录形如 frames_number_16_split_by_number)，之后再跑就很快了。
    train_set = DVS128Gesture(root, train=True, data_type='frame',
                              frames_number=T, split_by='number')
    test_set = DVS128Gesture(root, train=False, data_type='frame',
                             frames_number=T, split_by='number')

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True)

    return train_loader, test_loader
