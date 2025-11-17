"""
Linear Regression vs Polynomial Regression Comparison
线性回归与多项式回归对比分析

This script compares simple linear regression with polynomial regression
on fetal heart rate data, demonstrating why polynomial features are necessary
for non-linear data patterns.

Author: Jinyong-Li
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """
    读取并返回数据

    Parameters:
        file_path (str): Excel文件路径

    Returns:
        tuple: (X, y, feature_name, label_name)
    """
    df = pd.read_excel(file_path)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    feature_name = df.columns[0]
    label_name = df.columns[1]

    print("=" * 70)
    print("数据信息")
    print("=" * 70)
    print(f"数据共 {len(y)} 条记录")
    print(f"特征: {feature_name}, 标签: {label_name}")

    return X, y, feature_name, label_name


def least_squares_fit(X_features, y):
    """
    最小二乘法求解
    使用正规方程: w = (X^T * X)^(-1) * X^T * y

    Parameters:
        X_features (ndarray): 特征矩阵
        y (ndarray): 标签向量

    Returns:
        ndarray: 回归系数
    """
    w = np.linalg.inv(X_features.T.dot(X_features)).dot(X_features.T).dot(y)
    return w


def simple_linear_regression(X, y):
    """
    简单线性回归: y = w0 + w1*x

    Parameters:
        X (ndarray): 输入特征
        y (ndarray): 标签

    Returns:
        tuple: (w, y_pred, mse, r2, X_linear)
    """
    n_samples = len(X)
    X_linear = np.c_[np.ones((n_samples, 1)), X]  # [1, x]

    w = least_squares_fit(X_linear, y)
    y_pred = X_linear.dot(w)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return w, y_pred, mse, r2, X_linear


def polynomial_regression(X, y, degree=2):
    """
    多项式回归: y = w0 + w1*x + w2*x²

    Parameters:
        X (ndarray): 输入特征
        y (ndarray): 标签
        degree (int): 多项式次数

    Returns:
        tuple: (w, y_pred, mse, r2, X_poly)
    """
    n_samples = len(X)
    X_poly = np.c_[np.ones((n_samples, 1)), X, X ** degree]  # [1, x, x²]

    w = least_squares_fit(X_poly, y)
    y_pred = X_poly.dot(w)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return w, y_pred, mse, r2, X_poly


def plot_comparison(X, y, X_plot, y_linear, y_poly,
                    mse_linear, r2_linear, mse_poly, r2_poly,
                    feature_name, label_name):
    """绘制线性回归和多项式回归的对比图"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1: 简单线性回归
    ax1 = axes[0]
    ax1.scatter(X, y, color='blue', alpha=0.6, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5, zorder=3)
    ax1.plot(X_plot, y_linear, color='red', linewidth=3, label='线性拟合', zorder=2)
    ax1.set_xlabel(f'{feature_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{label_name}', fontsize=12, fontweight='bold')
    ax1.set_title(f'简单线性回归\n$R^2$ = {r2_linear:.4f}, MSE = {mse_linear:.6f}',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.05, '直线无法完全拟合曲线数据', transform=ax1.transAxes,
             fontsize=10, color='red', ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 子图2: 多项式回归
    ax2 = axes[1]
    ax2.scatter(X, y, color='blue', alpha=0.6, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5, zorder=3)
    ax2.plot(X_plot, y_poly, color='green', linewidth=3, label='多项式拟合', zorder=2)
    ax2.set_xlabel(f'{feature_name}', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{label_name}', fontsize=12, fontweight='bold')
    ax2.set_title(f'二次多项式回归\n$R^2$ = {r2_poly:.4f}, MSE = {mse_poly:.6f}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.05, '曲线更好地捕捉数据特征', transform=ax2.transAxes,
             fontsize=10, color='green', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 子图3: 两种方法对比
    ax3 = axes[2]
    ax3.scatter(X, y, color='blue', alpha=0.5, s=60, label='实际数据',
                edgecolors='black', linewidth=0.5, zorder=4)
    ax3.plot(X_plot, y_linear, color='red', linewidth=2.5, linestyle='--',
             label=f'线性回归 ($R^2$={r2_linear:.4f})', zorder=2)
    ax3.plot(X_plot, y_poly, color='green', linewidth=2.5, linestyle='-',
             label=f'多项式回归 ($R^2$={r2_poly:.4f})', zorder=3)
    ax3.set_xlabel(f'{feature_name}', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'{label_name}', fontsize=12, fontweight='bold')
    ax3.set_title('两种方法对比', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)

    # 标注差异最大的区域
    diff = np.abs(y_linear - y_poly)
    max_diff_idx = np.argmax(diff)
    ax3.annotate('最大差异区域',
                 xy=(X_plot[max_diff_idx], y_linear[max_diff_idx]),
                 xytext=(X_plot[max_diff_idx] + 0.15, y_linear[max_diff_idx] + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('linear_vs_polynomial_comparison.png', dpi=300, bbox_inches='tight')
    print("\n对比图已保存: linear_vs_polynomial_comparison.png")
    plt.show()


def plot_residuals(X, y_true, y_linear, y_poly, feature_name):
    """绘制残差对比图"""

    residuals_linear = y_true - y_linear
    residuals_poly = y_true - y_poly

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 线性回归残差
    ax1 = axes[0]
    sort_idx = np.argsort(X.ravel())
    X_sorted = X[sort_idx]
    residuals_linear_sorted = residuals_linear[sort_idx]

    ax1.scatter(X_sorted, residuals_linear_sorted, color='red', alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel(f'{feature_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('残差', fontsize=12, fontweight='bold')
    ax1.set_title(f'简单线性回归残差分析\nMSE={mean_squared_error(y_true, y_linear):.6f}',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 拟合残差趋势线
    z = np.polyfit(X_sorted.ravel(), residuals_linear_sorted, 2)
    p = np.poly1d(z)
    ax1.plot(X_sorted, p(X_sorted), "r--", alpha=0.8, linewidth=2, label='残差趋势')
    ax1.legend(fontsize=9)
    ax1.text(0.5, 0.95, '残差呈现二次曲线模式', transform=ax1.transAxes,
             fontsize=10, color='red', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 多项式回归残差
    ax2 = axes[1]
    residuals_poly_sorted = residuals_poly[sort_idx]

    ax2.scatter(X_sorted, residuals_poly_sorted, color='green', alpha=0.6, s=50,
                edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel(f'{feature_name}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('残差', fontsize=12, fontweight='bold')
    ax2.set_title(f'多项式回归残差分析\nMSE={mean_squared_error(y_true, y_poly):.6f}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.95, '残差更接近随机分布', transform=ax2.transAxes,
             fontsize=10, color='green', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig('residuals_comparison.png', dpi=300, bbox_inches='tight')
    print("残差对比图已保存: residuals_comparison.png")
    plt.show()


def main():
    """主函数"""

    # 读取数据
    file_path = r'D:\ljy_course\rgzn\hw\zctxgyh.xlsx'  # 修改为你的数据路径
    X, y, feature_name, label_name = load_data(file_path)

    # 简单线性回归
    print("\n" + "=" * 70)
    print("简单线性回归 (y = w0 + w1*x)")
    print("=" * 70)
    w_linear, y_pred_linear, mse_linear, r2_linear, _ = simple_linear_regression(X, y)
    print(f"回归系数: w = [{w_linear[0]:.8f}, {w_linear[1]:.8f}]")
    print(f"回归方程: y = {w_linear[0]:.6f} + {w_linear[1]:.6f}*x")
    print(f"MSE = {mse_linear:.8f}")
    print(f"R² = {r2_linear:.8f}")

    # 多项式回归
    print("\n" + "=" * 70)
    print("二次多项式回归 (y = w0 + w1*x + w2*x²)")
    print("=" * 70)
    w_poly, y_pred_poly, mse_poly, r2_poly, _ = polynomial_regression(X, y, degree=2)
    print(f"回归系数: w = [{w_poly[0]:.8f}, {w_poly[1]:.8f}, {w_poly[2]:.8f}]")
    print(f"回归方程: y = {w_poly[0]:.6f} + {w_poly[1]:.6f}*x + {w_poly[2]:.6f}*x²")
    print(f"MSE = {mse_poly:.8f}")
    print(f"R² = {r2_poly:.8f}")

    # 性能对比
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)
    print(f"{'方法':<20} {'R²':<15} {'MSE':<15} {'相对改善':<15}")
    print("-" * 70)
    print(f"{'简单线性回归':<20} {r2_linear:<15.6f} {mse_linear:<15.6f} {'-':<15}")
    print(f"{'多项式回归':<20} {r2_poly:<15.6f} {mse_poly:<15.6f}", end="")

    r2_improvement = ((r2_poly - r2_linear) / r2_linear) * 100
    mse_improvement = ((mse_linear - mse_poly) / mse_linear) * 100
    print(f" {f'↑{r2_improvement:.1f}%':<15}")
    print(f"{'MSE降低':<20} {'':<15} {'':<15} {f'↓{mse_improvement:.1f}%':<15}")

    # 生成绘图数据
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    X_plot_linear = np.c_[np.ones((len(X_plot), 1)), X_plot]
    y_plot_linear = X_plot_linear.dot(w_linear)
    X_plot_poly = np.c_[np.ones((len(X_plot), 1)), X_plot, X_plot ** 2]
    y_plot_poly = X_plot_poly.dot(w_poly)

    # 可视化
    plot_comparison(X, y, X_plot, y_plot_linear, y_plot_poly,
                    mse_linear, r2_linear, mse_poly, r2_poly,
                    feature_name, label_name)

    plot_residuals(X, y, y_pred_linear, y_pred_poly, feature_name)

    # 分析总结
    print("\n" + "=" * 70)
    print("分析总结")
    print("=" * 70)
    print("\n关键发现:")
    print(f"1. 数据呈现明显的二次函数关系（倒U型曲线）")
    print(f"2. 简单线性回归R²为 {r2_linear:.4f}，虽然有一定拟合能力，但存在系统性偏差")
    print(f"3. 多项式回归R²达到 {r2_poly:.4f}，能更好地捕捉数据的非线性特征")
    print(f"4. MSE从 {mse_linear:.6f} 降低到 {mse_poly:.6f}，降低了 {mse_improvement:.1f}%")
    print(f"5. 从残差图可以看出，线性回归的残差呈现曲线模式，说明模型遗漏了二次特征")
    print(f"\n结论:")
    print(f"虽然简单线性回归在这个数据集上也能达到{r2_linear:.2f}的R²，但由于数据本质上")
    print(f"是二次关系，使用多项式特征扩展能够更准确地建模数据生成过程，")
    print(f"避免系统性偏差，得到更可靠的预测结果。")

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()