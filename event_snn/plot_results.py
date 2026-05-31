"""
plot_results.py 
# =====   任务  =====
用实测结果数据重画分析图

数据来自对最终模型 (channels=64, T=16, epochs=64, 测试准确率 78.82%) 跑 analyze.py
得到的输出，这里把数值固化下来，方便随时无需模型即可重绘报告用图。
"""
import matplotlib.pyplot as plt

# ===== 实测结果 (channels=64, T=16, epochs=64) =====
ACC = 0.7882            # 测试准确率
SPARSITY = 0.8807       # 整体稀疏度 (= 1 - 平均发放率)
ENERGY_RATIO = 0.0233   # 粗略能耗比 SNN/ANN (每连接每步)
TAUS = [1.968, 2.005, 1.947, 1.610, 1.481, 1.585, 1.191]          # 各 PLIF 层学到的 tau
FIRING = [0.0212, 0.0405, 0.0397, 0.0490, 0.1304, 0.4182, 0.1361]  # 各 PLIF 层平均发放率

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(range(len(TAUS)), TAUS, color='#4C72B0')
axes[0].set_title('Learned tau per PLIF layer')
axes[0].set_xlabel('PLIF layer index')
axes[0].set_ylabel('tau')
axes[0].set_xticks(range(len(TAUS)))

axes[1].bar(range(len(FIRING)), FIRING, color='#C44E52')
axes[1].set_title('Average firing rate per layer')
axes[1].set_xlabel('PLIF layer index')
axes[1].set_ylabel('firing rate')
axes[1].set_xticks(range(len(FIRING)))

fig.suptitle(f'DVS128 Gesture (PLIF Spiking CNN)  |  '
             f'Acc={ACC:.4f}   Sparsity={SPARSITY:.4f}   E(SNN/ANN)={ENERGY_RATIO:.4f}')
plt.tight_layout()
plt.savefig('analysis_c64.png', dpi=150)
print('saved analysis_c64.png')
