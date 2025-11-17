"""
胎心数据回归分析
使用最小二乘法和梯度下降法进行二次多项式回归
Author: Jinyong-Li
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import FuncFormatter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 数据读取与预处理
# =============================================================================
def load_and_preprocess_data(file_path):
    """读取并预处理数据"""
    df = pd.read_excel(file_path)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values

    print("=" * 70)
    print("数据信息")
    print("=" * 70)
    print(f"数据共 {len(y)} 条记录")
    print(f"特征: {df.columns[0]}, 标签: {df.columns[1]}")
    print(f"数据范围: X ∈ [{X.min():.4f}, {X.max():.4f}], y ∈ [{y.min():.4f}, {y.max():.4f}]")

    return X, y, df.columns[0], df.columns[1]


def create_polynomial_features(X, degree=2):
    """创建多项式特征"""
    n_samples = len(X)
    X_poly = np.c_[np.ones((n_samples, 1)), X, X ** degree]
    print(f"\n使用 {degree} 次多项式特征扩展: [1, x, x²]")
    return X_poly


# =============================================================================
# 最小二乘法
# =============================================================================
def ordinary_least_squares(X_poly, y):
    """
    最小二乘法求解线性回归
    使用正规方程: w = (X^T * X)^(-1) * X^T * y
    """
    w = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return w


# =============================================================================
# 梯度下降法
# =============================================================================
def gradient_descent(X_poly, y, learning_rate=0.1, n_iterations=50000,
                     tolerance=1e-6, print_interval=5000):
    """
    梯度下降法求解线性回归

    参数:
        X_poly: 多项式特征矩阵
        y: 标签
        learning_rate: 学习率
        n_iterations: 最大迭代次数
        tolerance: 收敛阈值
        print_interval: 打印间隔

    返回:
        w: 回归系数
        cost_history: 损失函数历史
        final_iter: 实际迭代次数
        converged: 是否收敛
    """
    m = len(y)
    n_features = X_poly.shape[1]

    # 初始化参数
    np.random.seed(42)
    w = np.zeros(n_features)

    cost_history = []
    converged = False
    final_iter = n_iterations - 1

    for iteration in range(n_iterations):
        # 前向传播
        y_pred = X_poly.dot(w)

        # 计算损失
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)

        # 计算梯度
        gradients = (1 / m) * X_poly.T.dot(y_pred - y)

        # 更新参数
        w_new = w - learning_rate * gradients

        # 打印进度
        if iteration % print_interval == 0:
            print(f"  迭代 {iteration:6d}: 损失 = {cost:.10f}, |Δw| = {np.linalg.norm(w_new - w):.10f}")

        # 检查收敛
        param_change = np.linalg.norm(w_new - w)
        if param_change < tolerance:
            converged = True
            final_iter = iteration
            print(f"\n  ✓ 在第 {iteration} 次迭代时收敛 (|Δw| = {param_change:.10f})")
            break

        w = w_new

    if not converged:
        print(f"\n  ⚠ 达到最大迭代次数 {n_iterations}，未满足收敛条件")

    return w, cost_history, final_iter, converged


# =============================================================================
# 模型评估
# =============================================================================
def evaluate_model(y_true, y_pred, w_name="模型"):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{w_name}:")
    print(f"  MSE = {mse:.8f}")
    print(f"  R² = {r2:.8f}")
    return mse, r2


# =============================================================================
# 可视化
# =============================================================================
def plot_results(X, y, X_plot, y_plot_ols, y_plot_gd, all_results,
                 mse_ols, r2_ols, best_result, final_config,
                 feature_name, label_name):
    """绘制完整的分析结果图"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 最小二乘法
    ax1 = axes[0, 0]
    ax1.scatter(X, y, color='blue', alpha=0.6, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5)
    ax1.plot(X_plot, y_plot_ols, color='red', linewidth=3, label='最小二乘法拟合')
    ax1.set_xlabel(f'{feature_name} (归一化)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{label_name} (归一化)', fontsize=12, fontweight='bold')
    ax1.set_title(f'最小二乘法\n$R^2$ = {r2_ols:.6f}, MSE = {mse_ols:.6f}',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: 梯度下降法
    ax2 = axes[0, 1]
    ax2.scatter(X, y, color='blue', alpha=0.6, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5)
    ax2.plot(X_plot, y_plot_gd, color='green', linewidth=3, label='梯度下降法拟合')
    ax2.set_xlabel(f'{feature_name} (归一化)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{label_name} (归一化)', fontsize=12, fontweight='bold')
    status = '已收敛' if best_result['converged'] else '未收敛'
    ax2.set_title(
        f'梯度下降法 (lr={final_config["lr"]}, iter={best_result["final_iter"]})\n'
        f'{status}, $R^2$ = {best_result["r2"]:.6f}, MSE = {best_result["mse"]:.6f}',
        fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3: 两种方法对比
    ax3 = axes[1, 0]
    ax3.scatter(X, y, color='blue', alpha=0.5, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5)
    ax3.plot(X_plot, y_plot_ols, color='red', linewidth=2.5, linestyle='--',
             label='最小二乘法')
    ax3.plot(X_plot, y_plot_gd, color='green', linewidth=2.5, linestyle='-.',
             label='梯度下降法')
    ax3.set_xlabel(f'{feature_name} (归一化)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'{label_name} (归一化)', fontsize=12, fontweight='bold')
    ax3.set_title('两种方法对比', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4: 收敛曲线对比
    ax4 = axes[1, 1]
    colors = ['purple', 'orange', 'brown']
    for i, result in enumerate(all_results):
        lr = result['config']['lr']
        label = f'lr={lr}'
        if result['converged']:
            label += ' (已收敛)'
        ax4.plot(result['cost_history'], color=colors[i], linewidth=2,
                 label=label, alpha=0.8)

    ax4.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
    ax4.set_ylabel('损失函数', fontsize=12, fontweight='bold')
    ax4.set_title('不同学习率的收敛对比', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0e}'))
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存: regression_analysis_results.png")
    plt.show()


def plot_scatter(X, y, feature_name, label_name):
    """绘制数据散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'{feature_name} (归一化)', fontsize=13, fontweight='bold')
    plt.ylabel(f'{label_name} (归一化)', fontsize=13, fontweight='bold')
    plt.title('胎心数据散点图', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("散点图已保存: data_scatter_plot.png")
    plt.show()


# =============================================================================
# 主程序
# =============================================================================
def main():
    """主函数"""

    # 1. 读取数据
    file_path = r'D:\ljy_course\rgzn\hw\zctxgyh.xlsx'
    X, y, feature_name, label_name = load_and_preprocess_data(file_path)

    # 2. 绘制散点图
    plot_scatter(X, y, feature_name, label_name)

    # 3. 创建多项式特征
    X_poly = create_polynomial_features(X, degree=2)

    # 4. 最小二乘法
    print("\n" + "=" * 70)
    print("最小二乘法")
    print("=" * 70)
    w_ols = ordinary_least_squares(X_poly, y)
    y_pred_ols = X_poly.dot(w_ols)
    print(f"回归系数: w = [{w_ols[0]:.8f}, {w_ols[1]:.8f}, {w_ols[2]:.8f}]")
    mse_ols, r2_ols = evaluate_model(y, y_pred_ols, "最小二乘法")

    # 5. 梯度下降法 - 测试多组学习率
    print("\n" + "=" * 70)
    print("梯度下降法 - 多组学习率对比")
    print("=" * 70)

    configs = [
        {"lr": 0.1, "max_iter": 50000, "tol": 1e-6, "name": "学习率 0.1"},
        {"lr": 0.3, "max_iter": 50000, "tol": 1e-6, "name": "学习率 0.3"},
        {"lr": 0.5, "max_iter": 50000, "tol": 1e-6, "name": "学习率 0.5"},
    ]

    all_results = []
    for i, config in enumerate(configs):
        print(f"\n{'─' * 70}")
        print(f"测试配置 {i + 1}: {config['name']}")
        print(f"{'─' * 70}")

        w_gd, cost_history, final_iter, converged = gradient_descent(
            X_poly, y,
            learning_rate=config['lr'],
            n_iterations=config['max_iter'],
            tolerance=config['tol'],
            print_interval=5000
        )

        y_pred_gd = X_poly.dot(w_gd)
        mse_gd, r2_gd = evaluate_model(y, y_pred_gd, f"梯度下降 (lr={config['lr']})")

        # 计算与OLS的差异
        mse_vs_ols = mean_squared_error(w_ols, w_gd)

        result = {
            'config': config,
            'w': w_gd,
            'mse': mse_gd,
            'r2': r2_gd,
            'final_iter': final_iter,
            'converged': converged,
            'cost_history': cost_history,
            'mse_vs_ols': mse_vs_ols
        }
        all_results.append(result)

        print(f"回归系数: w = [{w_gd[0]:.8f}, {w_gd[1]:.8f}, {w_gd[2]:.8f}]")
        print(f"与OLS系数差异 (MSE): {mse_vs_ols:.10f}")

    # 6. 选择最佳配置
    best_result = next((r for r in all_results if r['converged']), all_results[0])
    w_gd_final = best_result['w']
    final_config = best_result['config']

    # 7. 结果对比
    print("\n" + "=" * 70)
    print("最终结果对比")
    print("=" * 70)
    print(f"\n最小二乘法:")
    print(f"  w = [{w_ols[0]:.8f}, {w_ols[1]:.8f}, {w_ols[2]:.8f}]")
    print(f"  MSE = {mse_ols:.8f}, R² = {r2_ols:.8f}")

    print(f"\n梯度下降法 (学习率={final_config['lr']}):")
    print(f"  w = [{w_gd_final[0]:.8f}, {w_gd_final[1]:.8f}, {w_gd_final[2]:.8f}]")
    print(f"  迭代次数 = {best_result['final_iter']}")
    print(f"  收敛状态 = {'已收敛' if best_result['converged'] else '未收敛'}")
    print(f"  MSE = {best_result['mse']:.8f}, R² = {best_result['r2']:.8f}")
    print(f"  与OLS系数差异 (MSE) = {best_result['mse_vs_ols']:.10f}")

    # 8. 生成可视化
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_plot_poly = np.c_[np.ones((len(X_plot), 1)), X_plot, X_plot ** 2]
    y_plot_ols = X_plot_poly.dot(w_ols)
    y_plot_gd = X_plot_poly.dot(w_gd_final)

    plot_results(X, y, X_plot, y_plot_ols, y_plot_gd, all_results,
                 mse_ols, r2_ols, best_result, final_config,
                 feature_name, label_name)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()