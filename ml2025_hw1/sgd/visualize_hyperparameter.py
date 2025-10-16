"""
可视化超参数搜索结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取结果
results_df = pd.read_csv('hyperparameter_search_results.csv')

# 创建热力图
fig, ax = plt.subplots(figsize=(10, 6))

# 准备数据（去除NaN和inf）
pivot_data = results_df.pivot(index='lambda', columns='eta', values='avg_mse')

# 将inf和过大的值设为NaN，以便可视化
pivot_data_clean = pivot_data.copy()
pivot_data_clean[pivot_data_clean > 100] = np.nan

# 创建热力图
im = ax.imshow(pivot_data_clean.values, cmap='RdYlGn_r', aspect='auto', 
               vmin=2.5, vmax=7, interpolation='nearest')

# 设置坐标轴
ax.set_xticks(np.arange(len(pivot_data.columns)))
ax.set_yticks(np.arange(len(pivot_data.index)))
ax.set_xticklabels([f'{x:.2f}' for x in pivot_data.columns])
ax.set_yticklabels([f'{y:.0e}' for y in pivot_data.index])

ax.set_xlabel('学习率 η', fontsize=12, fontweight='bold')
ax.set_ylabel('正则化系数 λ', fontsize=12, fontweight='bold')
ax.set_title('K折交叉验证超参数搜索热力图\n(颜色越绿表示MSE越小)', 
             fontsize=14, fontweight='bold', pad=20)

# 在每个格子中显示数值
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        value = pivot_data.iloc[i, j]
        if np.isnan(value):
            text = 'NaN'
            color = 'black'
        elif np.isinf(value):
            text = 'inf'
            color = 'black'
        elif value > 100:
            text = f'{value:.1e}'
            color = 'black'
        else:
            text = f'{value:.3f}'
            # 根据数值选择文字颜色
            if value < 3.5:
                color = 'white'
            else:
                color = 'black'
        
        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

# 标记最优组合
best_idx = results_df['avg_mse'].idxmin()
best_eta = results_df.loc[best_idx, 'eta']
best_lambda = results_df.loc[best_idx, 'lambda']
best_mse = results_df.loc[best_idx, 'avg_mse']

# 找到最优组合在热力图中的位置
best_j = list(pivot_data.columns).index(best_eta)
best_i = list(pivot_data.index).index(best_lambda)

# 在最优位置画一个星标
ax.plot(best_j, best_i, marker='*', markersize=20, color='red', 
        markeredgecolor='white', markeredgewidth=1.5)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('验证集均方误差 (MSE)', rotation=270, labelpad=20, fontsize=11)

plt.tight_layout()
plt.savefig('hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
print(f"热力图已保存至: hyperparameter_heatmap.png")
print(f"最优超参数: η*={best_eta}, λ*={best_lambda}, MSE={best_mse:.6f}")

# 创建第二个图：不同学习率下的MSE曲线
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 只绘制有效的（不发散的）结果
valid_lambdas = [1e-7, 1e-5, 1e-3, 0.1, 1.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'v', 'D']

for idx, lambda_val in enumerate(valid_lambdas):
    subset = results_df[results_df['lambda'] == lambda_val]
    # 过滤掉inf和NaN
    subset = subset[np.isfinite(subset['avg_mse'])]
    subset = subset.sort_values('eta')
    
    ax2.plot(subset['eta'], subset['avg_mse'], 
             marker=markers[idx], linewidth=2, markersize=8,
             label=f'λ = {lambda_val:.0e}', color=colors[idx])

ax2.set_xlabel('学习率 η', fontsize=12, fontweight='bold')
ax2.set_ylabel('验证集均方误差 (MSE)', fontsize=12, fontweight='bold')
ax2.set_title('不同正则化系数下学习率对MSE的影响', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([0.005, 0.055])

plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
print(f"学习率对比图已保存至: learning_rate_comparison.png")

plt.show()

print("\n可视化完成！")

