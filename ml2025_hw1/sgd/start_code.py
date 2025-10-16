import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_data(X, y, split_size=[0.8, 0.2], shuffle=False, random_seed=None):
    """
    对数据集进行划分

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        split_size - 划分比例，期望为一个浮点数列表，如[0.8, 0.2]表示将数据集划分为两部分，比例为80%和20%
        shuffle - 是否打乱数据集
        random_seed - 随机种子

    Return：
        X_list - 划分后的特征向量列表
        y_list - 划分后的标签向量列表
    """
    assert sum(split_size) == 1
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y = y[indices]
    X_list = []
    y_list = []

    # TODO 2.1.1
    start_idx = 0
    for ratio in split_size:
        # 计算当前划分的样本数量
        split_num = int(num_instances * ratio)
        end_idx = start_idx + split_num
        
        # 处理最后一个划分，确保包含所有剩余样本
        if end_idx > num_instances or ratio == split_size[-1]:
            end_idx = num_instances
        
        # 划分数据
        X_list.append(X[start_idx:end_idx])
        y_list.append(y[start_idx:end_idx])
        
        start_idx = end_idx
    
    return X_list, y_list


def feature_normalization(train, test):
    """将训练集中的所有特征值映射至[0,1]，对测试集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 特征归一化后的训练集
        test_normalized - 特征归一化后的测试集

    """
    # TODO 2.1.2
    # 从训练集计算每个特征的最小值和最大值
    train_min = np.min(train, axis=0)
    train_max = np.max(train, axis=0)
    
    # 避免除零错误：如果最大值等于最小值，说明该特征为常数
    # 此时将范围设为1，保持归一化后的值不变
    range_vals = train_max - train_min
    range_vals[range_vals == 0] = 1  # 避免除零
    
    # 将训练集归一化到 [0, 1]
    train_normalized = (train - train_min) / range_vals
    
    # 对测试集使用训练集的最小值和最大值进行相同的变换
    test_normalized = (test - train_min) / range_vals
    
    return train_normalized, test_normalized


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的岭回归损失函数

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)
        lambda_reg - 正则化系数

    Return：
        loss - 损失函数，标量
    """
    # TODO 2.2.2
    num_instances = X.shape[0]
    
    # 计算预测值
    predictions = np.dot(X, theta)
    
    # 计算残差
    residuals = predictions - y
    
    # 计算平方损失: (1/m) * ||Xθ - y||²
    square_loss = np.dot(residuals, residuals) / num_instances
    
    # 计算正则化项: λ * ||θ||²
    regularization = lambda_reg * np.dot(theta, theta)
    
    # 目标函数 J(θ)
    loss = square_loss + regularization
    
    return loss


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.2.4
    num_instances = X.shape[0]
    
    # 计算预测值
    predictions = np.dot(X, theta)
    
    # 计算残差
    residuals = predictions - y
    
    # 计算梯度: (2/m) * X^T(Xθ - y) + 2λθ
    grad = 2 * np.dot(X.T, residuals) / num_instances + 2 * lambda_reg * theta
    
    return grad


def grad_checker(X, y, theta, lambda_reg, epsilon=0.01, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数
        epsilon - 步长
        tolerance - 容差

    Return：
        梯度是否正确

    """
    grad_computed = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    num_features = theta.shape[0]
    grad_approx = np.zeros(num_features)

    for h in np.identity(num_features):
        J0 = compute_regularized_square_loss(X, y, theta - epsilon * h, lambda_reg)
        J1 = compute_regularized_square_loss(X, y, theta + epsilon * h, lambda_reg)
        grad_approx += (J1 - J0) / (2 * epsilon) * h
    dist = np.linalg.norm(grad_approx - grad_computed)
    return dist <= tolerance


def grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 全批量损失函数的历史，大小为 (num_iter) 的一维 numpy 数组
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 2.3.3
    for i in range(num_iter):
        # 计算当前损失
        loss_hist[i] = compute_regularized_square_loss(X, y, theta, lambda_reg)
        
        # 计算梯度
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        
        # 梯度检查
        if check_gradient:
            if not grad_checker(X, y, theta, lambda_reg):
                print(f"警告：第{i}次迭代梯度检查未通过")
        
        # 更新参数：θ := θ - α * ∇J(θ)
        theta = theta - alpha * grad
        
        # 保存参数历史
        theta_hist[i + 1] = theta
    
    return theta_hist, loss_hist


def stochastic_grad_descent(
    X_train, y_train, X_val, y_val, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1
):
    """
    随机梯度下降，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_val - 验证集特征向量，数组大小 (num_instances, num_features)
        y_val - 验证集标签向量，数组大小 (num_instances)
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量正则化损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist

    # TODO 2.4.3


cross_validation_K = 5


def K_fold_split_data(X, y, K=cross_validation_K, shuffle=False, random_seed=None):
    """
    K 折划分数据集

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        K - 折数
        shuffle - 是否打乱数据集
        random_seed - 随机种子

    Return：
        X_train_list - 划分后的训练集特征向量列表
        y_train_list - 划分后的训练集标签向量列表
        X_valid_list - 划分后的验证集特征向量列表
        y_valid_list - 划分后的验证集标签向量列表
    """
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y = y[indices]
    X_train_list, y_train_list = [], []
    X_valid_list, y_valid_list = [], []

    # TODO 2.5.1


def K_fold_cross_validation(
    X,
    y,
    alphas,
    lambdas,
    num_iter=1000,
    K=cross_validation_K,
    shuffle=False,
    random_seed=None,
):
    """
    K 折交叉验证

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        alphas - 搜索的步长列表
        lambdas - 搜索的正则化系数列表
        num_iter - 要运行的迭代次数
        K - 折数
        shuffle - 是否打乱数据集
        random_seed - 随机种子

    Return：
        alpha_best - 最佳步长
        lambda_best - 最佳正则化系数
    """
    alpha_best, lambda_best = None, None
    X_train_list, y_train_list, X_valid_list, y_valid_list = K_fold_split_data(
        X, y, K, shuffle, random_seed
    )

    # TODO 2.5.2


def analytical_solution(X, y, lambda_reg):
    """
    岭回归解析解

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        lambda_reg - 正则化系数

    Return：
        theta - 参数向量
    """
    assert lambda_reg > 0
    # TODO 2.6.1


def main():
    # 加载数据集
    print("loading the dataset")

    df = pd.read_csv("data.csv", delimiter=",")
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print("Split into Train and Test")
    (X_train, X_test), (y_train, y_test) = split_data(
        X, y, split_size=[0.8, 0.2], shuffle=True, random_seed=0
    )

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项

    # TODO


if __name__ == "__main__":
    main()
