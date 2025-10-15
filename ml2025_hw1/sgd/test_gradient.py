"""
梯度检验测试脚本
用于验证岭回归的损失函数和梯度计算是否正确
"""

import numpy as np
from start_code import (
    compute_regularized_square_loss,
    compute_regularized_square_loss_gradient,
    grad_checker
)


def test_gradient():
    """测试梯度计算的正确性"""
    
    print("=" * 70)
    print("岭回归梯度检验测试")
    print("=" * 70)
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 测试案例1：小规模数据
    print("\n【测试1】小规模数据 (10个样本, 5个特征)")
    print("-" * 70)
    
    X1 = np.random.randn(10, 5)
    y1 = np.random.randn(10)
    theta1 = np.random.randn(5)
    lambda1 = 0.1
    
    # 计算损失值
    loss1 = compute_regularized_square_loss(X1, y1, theta1, lambda1)
    print(f"目标函数值 J(θ): {loss1:.6f}")
    
    # 计算梯度
    grad1 = compute_regularized_square_loss_gradient(X1, y1, theta1, lambda1)
    print(f"梯度向量: {grad1}")
    print(f"梯度范数: {np.linalg.norm(grad1):.6f}")
    
    # 梯度检验
    is_correct1 = grad_checker(X1, y1, theta1, lambda1)
    result1 = "✓ 通过" if is_correct1 else "✗ 失败"
    print(f"\n梯度检验结果: {result1}")
    
    # 测试案例2：中等规模数据
    print("\n" + "=" * 70)
    print("\n【测试2】中等规模数据 (50个样本, 10个特征)")
    print("-" * 70)
    
    X2 = np.random.randn(50, 10)
    y2 = np.random.randn(50)
    theta2 = np.random.randn(10)
    lambda2 = 0.5
    
    loss2 = compute_regularized_square_loss(X2, y2, theta2, lambda2)
    print(f"目标函数值 J(θ): {loss2:.6f}")
    
    grad2 = compute_regularized_square_loss_gradient(X2, y2, theta2, lambda2)
    print(f"梯度范数: {np.linalg.norm(grad2):.6f}")
    
    is_correct2 = grad_checker(X2, y2, theta2, lambda2)
    result2 = "✓ 通过" if is_correct2 else "✗ 失败"
    print(f"\n梯度检验结果: {result2}")
    
    # 测试案例3：不同的正则化系数
    print("\n" + "=" * 70)
    print("\n【测试3】不同正则化系数的影响")
    print("-" * 70)
    
    X3 = np.random.randn(20, 8)
    y3 = np.random.randn(20)
    theta3 = np.random.randn(8)
    
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    all_passed = True
    
    print(f"{'λ':>8} {'J(θ)':>12} {'||∇J||':>12} {'梯度检验':>12}")
    print("-" * 70)
    
    for lam in lambdas:
        loss = compute_regularized_square_loss(X3, y3, theta3, lam)
        grad = compute_regularized_square_loss_gradient(X3, y3, theta3, lam)
        grad_norm = np.linalg.norm(grad)
        is_correct = grad_checker(X3, y3, theta3, lam)
        result = "✓ 通过" if is_correct else "✗ 失败"
        
        print(f"{lam:8.2f} {loss:12.6f} {grad_norm:12.6f} {result:>12}")
        
        if not is_correct:
            all_passed = False
    
    # 总结
    print("\n" + "=" * 70)
    print("\n【测试总结】")
    print("-" * 70)
    
    test_results = [
        ("测试1 (小规模数据)", is_correct1),
        ("测试2 (中等规模数据)", is_correct2),
        ("测试3 (不同正则化系数)", all_passed)
    ]
    
    all_tests_passed = all(result for _, result in test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:.<50} {status}")
    
    print("\n" + "=" * 70)
    
    if all_tests_passed:
        print("\n🎉 所有测试通过！梯度计算实现正确！")
    else:
        print("\n⚠️  部分测试失败，请检查实现！")
    
    print("=" * 70 + "\n")
    
    return all_tests_passed


def manual_gradient_check():
    """手动对比解析梯度和数值梯度（用于调试）"""
    
    print("\n" + "=" * 70)
    print("【手动梯度检验】解析梯度 vs 数值梯度")
    print("=" * 70 + "\n")
    
    np.random.seed(42)
    X = np.random.randn(5, 3)
    y = np.random.randn(5)
    theta = np.random.randn(3)
    lambda_reg = 0.1
    epsilon = 0.01
    
    # 计算解析梯度
    grad_analytical = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    
    # 手动计算数值梯度
    grad_numerical = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        
        J_plus = compute_regularized_square_loss(X, y, theta_plus, lambda_reg)
        J_minus = compute_regularized_square_loss(X, y, theta_minus, lambda_reg)
        
        grad_numerical[i] = (J_plus - J_minus) / (2 * epsilon)
    
    # 逐个分量比较
    print(f"{'参数':>8} {'解析梯度':>15} {'数值梯度':>15} {'差异':>15}")
    print("-" * 70)
    
    for i in range(len(theta)):
        diff = abs(grad_analytical[i] - grad_numerical[i])
        print(f"θ[{i}]    {grad_analytical[i]:15.8f} {grad_numerical[i]:15.8f} {diff:15.10f}")
    
    total_diff = np.linalg.norm(grad_analytical - grad_numerical)
    print("-" * 70)
    print(f"总体差异 (欧氏距离): {total_diff:.10f}")
    
    if total_diff < 1e-4:
        print("结果: ✓ 梯度计算正确\n")
    else:
        print("结果: ✗ 梯度计算可能有误\n")


if __name__ == "__main__":
    # 运行测试
    test_gradient()
    
    # 运行手动检验（可选，用于详细调试）
    print("\n是否运行手动梯度检验以查看详细对比？(y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            manual_gradient_check()
    except:
        # 如果不是交互式环境，直接运行
        manual_gradient_check()

