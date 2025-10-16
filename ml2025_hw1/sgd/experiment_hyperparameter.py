"""
实验 2.4.2: K折交叉验证超参数搜索

搜索最优的学习率和正则化系数组合
"""

import numpy as np
import pandas as pd
from start_code import (
    split_data,
    feature_normalization,
    K_fold_cross_validation,
    grad_descent  # 用于最终在完整训练集上训练
)


def run_hyperparameter_search():
    """运行超参数搜索实验"""
    
    print("=" * 80)
    print("实验 2.4.2: K折交叉验证超参数搜索")
    print("=" * 80)
    
    # 加载数据
    print("\n1. 加载并预处理数据...")
    df = pd.read_csv("data.csv", delimiter=",")
    X = df.values[:, :-1]
    y = df.values[:, -1]
    
    # 划分训练集和测试集
    (X_train, X_test), (y_train, y_test) = split_data(
        X, y, split_size=[0.8, 0.2], shuffle=True, random_seed=0
    )
    
    # 特征归一化
    X_train, X_test = feature_normalization(X_train, X_test)
    
    # 添加偏置项
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    
    print(f"   训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征（含偏置）")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 定义超参数搜索空间
    alphas = [0.05, 0.04, 0.03, 0.02, 0.01]
    lambdas = [1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100]
    K = 5
    num_iter = 1000
    
    print(f"\n2. 超参数搜索设置:")
    print(f"   步长 η: {alphas}")
    print(f"   正则化系数 λ: {lambdas}")
    print(f"   K折交叉验证: K = {K}")
    print(f"   迭代次数: {num_iter}")
    
    # 进行K折交叉验证搜索
    print(f"\n3. 进行K折交叉验证...")
    print(f"   总共需要测试 {len(alphas)} × {len(lambdas)} = {len(alphas) * len(lambdas)} 个超参数组合")
    print(f"   每个组合需要训练 {K} 次（K折）")
    print(f"   总计训练次数: {len(alphas) * len(lambdas) * K}")
    
    # 调用start_code.py中实现的K折交叉验证函数
    alpha_best, lambda_best, results = K_fold_cross_validation(
        X_train, y_train,
        alphas=alphas,
        lambdas=lambdas,
        num_iter=num_iter,
        K=K,
        shuffle=True,
        random_seed=42,
        return_all_results=True  # 返回所有超参数组合的详细结果
    )
    
    # 整理结果为DataFrame
    results_df = pd.DataFrame(results)
    best_avg_error = results_df['avg_mse'].min()
    
    # 生成结果透视表
    pivot_table = results_df.pivot(index='lambda', columns='eta', values='avg_mse')
    
    print(f"\n4. 交叉验证结果（验证集均方误差MSE）:")
    print("=" * 80)
    print(pivot_table.to_string())
    
    print(f"\n5. 最优超参数:")
    print("=" * 80)
    print(f"   最优学习率 η* = {alpha_best}")
    print(f"   最优正则化系数 λ* = {lambda_best}")
    print(f"   交叉验证均方误差 = {best_avg_error:.6f}")
    
    # 使用最优超参数在整个训练集上训练
    print(f"\n6. 使用最优超参数在完整训练集上训练...")
    theta_hist_final, loss_hist_final = grad_descent(
        X_train, y_train,
        lambda_reg=lambda_best,
        alpha=alpha_best,
        num_iter=num_iter,
        check_gradient=False
    )
    
    theta_final = theta_hist_final[-1]
    
    # 在测试集上评估
    predictions_test = np.dot(X_test, theta_final)
    test_mse = np.mean((predictions_test - y_test) ** 2)
    
    print(f"   训练完成")
    print(f"   训练集最终损失: {loss_hist_final[-1]:.6f}")
    
    print(f"\n7. 测试集性能:")
    print("=" * 80)
    print(f"   测试集均方误差 MSE = {test_mse:.6f}")
    print(f"   测试集均方根误差 RMSE = {np.sqrt(test_mse):.6f}")
    
    # 生成LaTeX格式的表格
    print(f"\n8. LaTeX格式表格（可直接用于论文）:")
    print("=" * 80)
    print(pivot_table.to_latex(float_format="%.6f"))
    
    # 保存结果
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    pivot_table.to_csv('hyperparameter_search_pivot.csv')
    
    print(f"\n结果已保存至:")
    print(f"   - hyperparameter_search_results.csv (详细结果)")
    print(f"   - hyperparameter_search_pivot.csv (透视表)")
    
    # 找出每个λ对应的最优η
    print(f"\n9. 每个正则化系数对应的最优学习率:")
    print("=" * 80)
    print(f"{'λ':>10} {'最优η':>10} {'验证MSE':>15}")
    print("-" * 80)
    
    for lambda_val in lambdas:
        subset = results_df[results_df['lambda'] == lambda_val]
        best_row = subset.loc[subset['avg_mse'].idxmin()]
        print(f"{lambda_val:>10.0e} {best_row['eta']:>10.2f} {best_row['avg_mse']:>15.6f}")
    
    # 找出每个η对应的最优λ
    print(f"\n10. 每个学习率对应的最优正则化系数:")
    print("=" * 80)
    print(f"{'η':>10} {'最优λ':>10} {'验证MSE':>15}")
    print("-" * 80)
    
    for alpha_val in alphas:
        subset = results_df[results_df['eta'] == alpha_val]
        best_row = subset.loc[subset['avg_mse'].idxmin()]
        print(f"{alpha_val:>10.2f} {best_row['lambda']:>10.0e} {best_row['avg_mse']:>15.6f}")
    
    print(f"\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    
    return alpha_best, lambda_best, test_mse, results_df


if __name__ == "__main__":
    alpha_best, lambda_best, test_mse, results_df = run_hyperparameter_search()

