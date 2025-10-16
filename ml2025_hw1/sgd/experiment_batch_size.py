"""
实验 2.5.4: 批大小对SGD收敛性能的影响

固定λ=0，测试不同batch_size对训练过程的影响
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from start_code import (
    split_data,
    feature_normalization,
    stochastic_grad_descent,
    compute_regularized_square_loss
)

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def batch_size_experiment():
    """批大小对SGD收敛性能影响的实验"""
    
    print("=" * 80)
    print("实验 2.5.4: 批大小对SGD收敛性能的影响")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载并预处理数据...")
    df = pd.read_csv("data.csv", delimiter=",")
    X = df.values[:, :-1]
    y = df.values[:, -1]
    
    # 划分训练集和验证集（使用split_data，不使用K折）
    (X_train, X_val), (y_train, y_val) = split_data(
        X, y, split_size=[0.8, 0.2], shuffle=True, random_seed=42
    )
    
    # 特征归一化
    X_train, X_val = feature_normalization(X_train, X_val)
    
    # 添加偏置项
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
    
    print(f"   训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征（含偏置）")
    print(f"   验证集: {X_val.shape[0]} 样本")
    
    # 实验设置
    lambda_reg = 0  # 固定λ=0
    alpha = 0.03  # 根据2.4.2的结果
    num_iter = 1000
    batch_sizes = [1, 4, 8, 16, 32, 64, X_train.shape[0]]  # 不同的批大小
    
    print(f"\n2. 实验设置:")
    print(f"   正则化系数 λ: {lambda_reg}")
    print(f"   学习率 η: {alpha} (根据2.4.2小节结果选择)")
    print(f"   迭代次数: {num_iter}")
    print(f"   批大小列表: {batch_sizes}")
    
    # 运行实验
    print(f"\n3. 运行不同批大小的SGD...")
    
    results = []
    
    for batch_size in batch_sizes:
        batch_name = f"batch={batch_size}" if batch_size < X_train.shape[0] else f"Full Batch ({batch_size})"
        print(f"   运行 {batch_name}...")
        
        # 运行SGD
        theta_hist, loss_hist, val_hist = stochastic_grad_descent(
            X_train, y_train,
            X_val, y_val,
            lambda_reg=lambda_reg,
            alpha=alpha,
            num_iter=num_iter,
            batch_size=batch_size
        )
        
        # 计算验证集上的全批量损失（带正则化，用于公平比较）
        val_loss_hist = []
        for i in range(num_iter):
            theta = theta_hist[i]
            val_loss = compute_regularized_square_loss(X_val, y_val, theta, lambda_reg)
            val_loss_hist.append(val_loss)
        
        val_loss_hist = np.array(val_loss_hist)
        
        results.append({
            'batch_size': batch_size,
            'name': batch_name,
            'train_loss': loss_hist,  # 小批量训练损失（有噪声）
            'val_mse': val_hist,  # 验证集MSE（不带正则化）
            'val_loss': val_loss_hist,  # 验证集损失（带正则化）
            'final_val_mse': val_hist[-1],
            'final_val_loss': val_loss_hist[-1]
        })
        
        print(f"      最终验证MSE: {val_hist[-1]:.6f}")
        print(f"      最终验证损失: {val_loss_hist[-1]:.6f}")
    
    # 绘制对比图
    print(f"\n4. 生成对比图...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 训练损失（小批量，有噪声）
    ax1 = plt.subplot(2, 2, 1)
    for result in results:
        if result['batch_size'] < X_train.shape[0]:  # 只显示SGD的训练损失
            ax1.plot(result['train_loss'], label=result['name'], alpha=0.7)
    ax1.set_xlabel('迭代次数', fontsize=11)
    ax1.set_ylabel('训练损失（小批量）', fontsize=11)
    ax1.set_title('训练损失曲线（有噪声）', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 验证集MSE（不带正则化）
    ax2 = plt.subplot(2, 2, 2)
    for result in results:
        ax2.plot(result['val_mse'], label=result['name'], alpha=0.8, linewidth=2)
    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('验证集MSE', fontsize=11)
    ax2.set_title('验证集均方误差（用于评估收敛）', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 验证集损失（带正则化，对数尺度）
    ax3 = plt.subplot(2, 2, 3)
    for result in results:
        ax3.semilogy(result['val_loss'], label=result['name'], alpha=0.8, linewidth=2)
    ax3.set_xlabel('迭代次数', fontsize=11)
    ax3.set_ylabel('验证集损失（对数尺度）', fontsize=11)
    ax3.set_title('验证集损失曲线（对数尺度）', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 收敛速度对比（前200次迭代）
    ax4 = plt.subplot(2, 2, 4)
    for result in results:
        ax4.plot(result['val_mse'][:200], label=result['name'], alpha=0.8, linewidth=2)
    ax4.set_xlabel('迭代次数', fontsize=11)
    ax4.set_ylabel('验证集MSE', fontsize=11)
    ax4.set_title('收敛速度对比（前200次迭代）', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('批大小对SGD收敛性能的影响 (λ=0, η=0.05)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('batch_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   对比图已保存至: batch_size_comparison.png")
    
    # 输出结果表格
    print(f"\n5. 实验结果:")
    print("=" * 80)
    print(f"{'批大小':>10} {'最终验证MSE':>20} {'最终验证损失':>20} {'收敛速度':>15}")
    print("-" * 80)
    
    for result in results:
        # 计算收敛速度：前100次迭代的损失下降幅度
        loss_decrease = (result['val_loss'][0] - result['val_loss'][min(99, num_iter-1)]) / result['val_loss'][0] * 100
        speed = f"{loss_decrease:.1f}%"
        
        print(f"{result['batch_size']:>10} {result['final_val_mse']:>20.6f} {result['final_val_loss']:>20.6f} {speed:>15}")
    
    # 分析和结论
    print(f"\n6. 分析与结论:")
    print("=" * 80)
    
    # 找出最快收敛的batch_size（前100次迭代损失下降最多）
    best_speed = max(results, key=lambda x: (x['val_loss'][0] - x['val_loss'][min(99, num_iter-1)]) / x['val_loss'][0])
    print(f"\n收敛最快的批大小: {best_speed['batch_size']}")
    
    # 找出最终性能最好的batch_size
    best_final = min(results, key=lambda x: x['final_val_mse'])
    print(f"最终性能最佳的批大小: {best_final['batch_size']} (验证MSE: {best_final['final_val_mse']:.6f})")
    
    # 噪声分析
    print(f"\n训练损失噪声分析（标准差）:")
    for result in results:
        if result['batch_size'] < X_train.shape[0]:
            # 计算后500次迭代的训练损失标准差（衡量噪声）
            noise_std = np.std(result['train_loss'][500:])
            print(f"   batch_size={result['batch_size']:3d}: 噪声标准差 = {noise_std:.6f}")
    
    print(f"\n关键观察:")
    print("1. 批大小越小，训练损失噪声越大（曲线越震荡）")
    print("2. 批大小越大，收敛曲线越平滑，但单次迭代计算量越大")
    print("3. 适中的批大小（如8-32）能在收敛速度和稳定性之间取得平衡")
    print("4. 验证集损失比训练损失更能准确反映模型收敛情况")
    
    print(f"\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    
    plt.show()
    
    return results

if __name__ == "__main__":
    results = batch_size_experiment()

