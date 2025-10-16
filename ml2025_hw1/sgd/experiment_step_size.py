"""
实验 2.3.3: 步长选择对梯度下降收敛的影响

固定正则化系数λ=0，测试不同步长下的收敛情况
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from start_code import (
    split_data,
    feature_normalization,
    grad_descent,
    compute_regularized_square_loss
)

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def run_experiment():
    """运行步长选择实验"""
    
    print("=" * 70)
    print("实验 2.3.3: 步长选择对梯度下降收敛的影响")
    print("=" * 70)
    
    # 加载数据
    print("\n1. 加载并预处理数据...")
    df = pd.read_csv("data.csv", delimiter=",")
    X = df.values[:, :-1]
    y = df.values[:, -1]
    
    print(f"   数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 划分数据集
    (X_train, X_test), (y_train, y_test) = split_data(
        X, y, split_size=[0.8, 0.2], shuffle=True, random_seed=0
    )
    
    # 特征归一化
    X_train, X_test = feature_normalization(X_train, X_test)
    
    # 添加偏置项
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 实验设置
    lambda_reg = 0.0  # 固定正则化系数为0
    num_iter = 1000    # 迭代次数
    
    # 测试不同的步长
    step_sizes = [0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]
    
    print(f"\n2. 实验参数设置:")
    print(f"   正则化系数 λ = {lambda_reg}")
    print(f"   迭代次数 = {num_iter}")
    print(f"   测试步长 η = {step_sizes}")
    
    # 存储结果
    results = {}
    
    print(f"\n3. 运行梯度下降实验...\n")
    print(f"{'步长 η':>10} {'初始损失':>12} {'最终损失':>12} {'是否收敛':>12} {'收敛速度':>12}")
    print("-" * 70)
    
    for alpha in step_sizes:
        try:
            theta_hist, loss_hist = grad_descent(
                X_train, y_train,
                lambda_reg=lambda_reg,
                alpha=alpha,
                num_iter=num_iter
            )
            
            initial_loss = loss_hist[0]
            final_loss = loss_hist[-1]
            
            # 判断是否收敛（损失下降且最终损失有限）
            converged = not np.isnan(final_loss) and not np.isinf(final_loss) and final_loss < initial_loss
            
            # 计算收敛速度（到达初始损失10%所需的迭代次数）
            target_loss = initial_loss * 0.1
            converged_iter = num_iter
            for i, loss in enumerate(loss_hist):
                if loss <= target_loss:
                    converged_iter = i
                    break
            
            convergence_speed = "快速" if converged_iter < 100 else "中速" if converged_iter < 300 else "慢速"
            if not converged:
                convergence_speed = "发散"
            
            status = "收敛" if converged else "发散"
            
            print(f"{alpha:10.2f} {initial_loss:12.6f} {final_loss:12.6f} {status:>12} {convergence_speed:>12}")
            
            results[alpha] = {
                'theta_hist': theta_hist,
                'loss_hist': loss_hist,
                'converged': converged,
                'convergence_speed': convergence_speed,
                'converged_iter': converged_iter
            }
            
        except Exception as e:
            print(f"{alpha:10.2f} {'错误':>12} {'错误':>12} {'发散':>12} {'发散':>12}")
            results[alpha] = {
                'theta_hist': None,
                'loss_hist': None,
                'converged': False,
                'convergence_speed': '发散',
                'converged_iter': num_iter
            }
    
    # 分析结果
    print(f"\n4. 实验结果分析:")
    print("-" * 70)
    
    # 找出收敛最快的步长
    converged_alphas = {alpha: res for alpha, res in results.items() if res['converged']}
    if converged_alphas:
        fastest_alpha = min(converged_alphas.keys(), 
                           key=lambda a: converged_alphas[a]['converged_iter'])
        print(f"   ✓ 收敛最快的步长: η = {fastest_alpha}")
        print(f"     (仅需 {converged_alphas[fastest_alpha]['converged_iter']} 次迭代)")
    
    # 找出发散的步长
    diverged_alphas = [alpha for alpha, res in results.items() if not res['converged']]
    if diverged_alphas:
        print(f"   ✗ 导致发散的步长: η = {diverged_alphas}")
    
    # 绘制收敛曲线
    plot_convergence_curves(results, step_sizes, num_iter)
    
    return results


def plot_convergence_curves(results, step_sizes, num_iter):
    """绘制目标函数随迭代次数变化的曲线"""
    
    print(f"\n5. 绘制收敛曲线...")
    
    plt.figure(figsize=(14, 5))
    
    # 子图1: 线性尺度
    plt.subplot(1, 2, 1)
    
    for alpha in step_sizes:
        if results[alpha]['loss_hist'] is not None:
            loss_hist = results[alpha]['loss_hist']
            
            # 限制显示范围，避免发散时的极大值影响可视化
            if results[alpha]['converged']:
                linestyle = '-'
                linewidth = 2
            else:
                linestyle = '--'
                linewidth = 1.5
                # 对发散的情况，只显示前面一部分
                loss_hist = loss_hist[:min(len(loss_hist), 100)]
            
            plt.plot(loss_hist, 
                    label=f'η={alpha}' + (' (发散)' if not results[alpha]['converged'] else ''),
                    linestyle=linestyle,
                    linewidth=linewidth)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('目标函数 J(θ)', fontsize=12)
    plt.title('不同步长下的收敛曲线 (线性尺度)', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)  # 从0开始显示
    
    # 子图2: 对数尺度（仅显示收敛的曲线）
    plt.subplot(1, 2, 2)
    
    for alpha in step_sizes:
        if results[alpha]['loss_hist'] is not None and results[alpha]['converged']:
            loss_hist = results[alpha]['loss_hist']
            # 避免log(0)，添加小的epsilon
            loss_hist_log = np.maximum(loss_hist, 1e-10)
            plt.semilogy(loss_hist_log, 
                        label=f'η={alpha}',
                        linewidth=2)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('目标函数 J(θ) (对数尺度)', fontsize=12)
    plt.title('收敛曲线对比 (对数尺度)', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'step_size_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 收敛曲线已保存至: {output_path}")
    
    plt.show()


def detailed_analysis(results):
    """详细分析不同步长的表现"""
    
    print(f"\n6. 详细分析:")
    print("=" * 70)
    
    for alpha in sorted(results.keys()):
        print(f"\n步长 η = {alpha}:")
        print("-" * 70)
        
        if results[alpha]['loss_hist'] is not None:
            loss_hist = results[alpha]['loss_hist']
            initial_loss = loss_hist[0]
            final_loss = loss_hist[-1]
            
            if results[alpha]['converged']:
                print(f"  状态: ✓ 收敛")
                print(f"  初始损失: {initial_loss:.6f}")
                print(f"  最终损失: {final_loss:.6f}")
                print(f"  损失下降: {initial_loss - final_loss:.6f} "
                      f"({(1 - final_loss/initial_loss)*100:.2f}%)")
                print(f"  收敛速度: {results[alpha]['convergence_speed']}")
                
                # 分析收敛特性
                if alpha <= 0.05:
                    print(f"  特点: 步长较小，收敛稳定但速度较慢")
                elif alpha <= 0.1:
                    print(f"  特点: 步长适中，收敛速度和稳定性平衡较好")
                elif alpha <= 0.5:
                    print(f"  特点: 步长较大，收敛快速但可能有振荡")
            else:
                print(f"  状态: ✗ 发散")
                print(f"  初始损失: {initial_loss:.6f}")
                print(f"  最终损失: {final_loss}")
                print(f"  原因: 步长过大，超过了理论上界，导致参数更新过度")
        else:
            print(f"  状态: ✗ 严重发散（无法完成计算）")


if __name__ == "__main__":
    # 运行实验
    results = run_experiment()
    
    # 详细分析
    detailed_analysis(results)
    
    print("\n" + "=" * 70)
    print("实验完成！请查看生成的图表进行分析。")
    print("=" * 70)

