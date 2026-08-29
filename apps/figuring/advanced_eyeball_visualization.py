"""
增强版3D眼动数据可视化
提供多种着色方式和交互式分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 导入基础函数
from visualize_eyeball_gaze import (
    screen_to_eyeball_3d, load_gaze_data,
    EYEBALL_RADIUS_MM, DISTANCE_TO_SCREEN_MM,
    SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX
)


def analyze_gaze_patterns(gaze_x, gaze_y):
    """分析眼动模式"""
    # 计算眼动速度（相邻点之间的距离）
    n_points = len(gaze_x)
    if n_points > 1:
        distances = np.sqrt(
            np.diff(gaze_x)**2 + np.diff(gaze_y)**2
        )
        mean_velocity = distances.mean()
        max_velocity = distances.max()
        std_velocity = distances.std()
    else:
        mean_velocity = max_velocity = std_velocity = 0

    # 计算注视中心（凸包中心）
    center_x = gaze_x.mean()
    center_y = gaze_y.mean()

    # 计算散布度（标准差）
    spread_x = gaze_x.std()
    spread_y = gaze_y.std()

    return {
        'mean_velocity': mean_velocity,
        'max_velocity': max_velocity,
        'std_velocity': std_velocity,
        'center_x': center_x,
        'center_y': center_y,
        'spread_x': spread_x,
        'spread_y': spread_y,
    }


def draw_eyeball_sphere(ax, radius=EYEBALL_RADIUS_MM, alpha=0.1, color='gray'):
    """
    在3D坐标轴上绘制眼球球体

    Args:
        ax: 3D坐标轴
        radius: 球体半径
        alpha: 透明度
        color: 颜色
    """
    # 生成球体网格
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 25)  # 只画上半球
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # 绘制球体表面
    ax.plot_surface(x, y, z, alpha=alpha, color=color, linewidth=0, shade=True)


def create_enhanced_visualization(gaze_x, gaze_y, output_path=None,
                                 color_mode='sequence', point_size=5,
                                 alpha=0.8, figsize=(20, 8)):
    """
    创建增强版3D眼动数据可视化

    Args:
        gaze_x: 屏幕x坐标数组
        gaze_y: 屏幕y坐标数组
        output_path: 输出图片路径（可选）
        color_mode: 着色模式
            - 'sequence': 按数据序列（时间顺序）
            - 'velocity': 按眼动速度
            - 'position': 按空间位置（径向距离）
            - 'quadrant': 按象限分类
        point_size: 点的大小
        alpha: 透明度
        figsize: 图形尺寸
    """
    # 转换到3D眼球坐标
    eyeball_coords = np.array([
        screen_to_eyeball_3d(x, y) for x, y in zip(gaze_x, gaze_y)
    ])
    eyeball_x = eyeball_coords[:, 0]
    eyeball_y = eyeball_coords[:, 1]
    eyeball_z = eyeball_coords[:, 2]

    n_points = len(gaze_x)

    # 根据模式设置颜色
    if color_mode == 'sequence':
        # 按序列着色（时间顺序）
        colors = np.arange(n_points)
        cmap_name = 'jet'
        colorbar_label = 'Sequence Index'

    elif color_mode == 'velocity':
        # 按眼动速度着色
        velocities = np.zeros(n_points)
        if n_points > 1:
            distances = np.sqrt(
                np.diff(gaze_x)**2 + np.diff(gaze_y)**2
            )
            velocities[1:] = distances
            velocities[0] = distances[0]  # 第一个点使用第二个点的速度
        colors = velocities
        cmap_name = 'hot'
        colorbar_label = 'Velocity (px/frame)'

    elif color_mode == 'position':
        # 按径向距离着色（距离注视中心的距离）
        center_x = gaze_x.mean()
        center_y = gaze_y.mean()
        radial_distances = np.sqrt(
            (gaze_x - center_x)**2 + (gaze_y - center_y)**2
        )
        colors = radial_distances
        cmap_name = 'viridis'
        colorbar_label = 'Radial Distance (px)'

    elif color_mode == 'quadrant':
        # 按象限分类
        center_x = SCREEN_WIDTH_PX / 2
        center_y = SCREEN_HEIGHT_PX / 2
        quadrants = np.zeros(n_points)
        for i in range(n_points):
            if gaze_x[i] >= center_x and gaze_y[i] >= center_y:
                quadrants[i] = 0  # 右上
            elif gaze_x[i] < center_x and gaze_y[i] >= center_y:
                quadrants[i] = 1  # 左上
            elif gaze_x[i] < center_x and gaze_y[i] < center_y:
                quadrants[i] = 2  # 左下
            else:
                quadrants[i] = 3  # 右下
        colors = quadrants
        cmap_name = 'tab10'
        colorbar_label = 'Quadrant'

    else:
        colors = np.arange(n_points)
        cmap_name = 'jet'
        colorbar_label = 'Sequence Index'

    # 创建图形（2x3布局）
    fig = plt.figure(figsize=figsize)

    # 1. 主3D视图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # 绘制眼球球体
    draw_eyeball_sphere(ax1, alpha=0.15, color='lightblue')
    # 绘制gaze点
    scatter1 = ax1.scatter(eyeball_x, eyeball_y, eyeball_z,
                          c=colors, s=point_size, alpha=alpha,
                          cmap=cmap_name, edgecolors='none')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D View', fontsize=12, fontweight='bold')
    limit = EYEBALL_RADIUS_MM * 1.2
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.set_zlim([0, limit])

    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    cbar1.set_label(colorbar_label, fontsize=9)

    # 2. 俯视图
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    draw_eyeball_sphere(ax2, alpha=0.15, color='lightblue')
    scatter2 = ax2.scatter(eyeball_x, eyeball_y, eyeball_z,
                          c=colors, s=point_size, alpha=alpha,
                          cmap=cmap_name, edgecolors='none')
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('Top View', fontsize=12, fontweight='bold')
    ax2.set_xlim([-limit, limit])
    ax2.set_ylim([-limit, limit])
    ax2.set_zlim([0, limit])

    # 3. 侧视图
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    draw_eyeball_sphere(ax3, alpha=0.15, color='lightblue')
    scatter3 = ax3.scatter(eyeball_x, eyeball_y, eyeball_z,
                          c=colors, s=point_size, alpha=alpha,
                          cmap=cmap_name, edgecolors='none')
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('Side View', fontsize=12, fontweight='bold')
    ax3.set_xlim([-limit, limit])
    ax3.set_ylim([-limit, limit])
    ax3.set_zlim([0, limit])

    # 4. 2D屏幕坐标散点图
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(gaze_x, gaze_y,
                          c=colors, s=point_size, alpha=alpha,
                          cmap=cmap_name, edgecolors='none')
    ax4.set_xlabel('X (px)')
    ax4.set_ylabel('Y (px)')
    ax4.set_title('2D Screen Coordinates', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, SCREEN_WIDTH_PX])
    ax4.set_ylim([SCREEN_HEIGHT_PX, 0])  # 反转y轴
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # 添加颜色条
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label(colorbar_label, fontsize=9)

    # 5. X坐标时间序列
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(gaze_x, 'b-', alpha=0.6, linewidth=0.5)
    ax5.scatter(np.arange(n_points), gaze_x, c=colors,
               s=point_size/2, alpha=alpha, cmap=cmap_name, edgecolors='none')
    ax5.set_xlabel('Sequence Index')
    ax5.set_ylabel('X (px)')
    ax5.set_title('X Coordinate Over Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Y坐标时间序列
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(gaze_y, 'r-', alpha=0.6, linewidth=0.5)
    ax6.scatter(np.arange(n_points), gaze_y, c=colors,
               s=point_size/2, alpha=alpha, cmap=cmap_name, edgecolors='none')
    ax6.set_xlabel('Sequence Index')
    ax6.set_ylabel('Y (px)')
    ax6.set_title('Y Coordinate Over Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'可视化结果已保存到: {output_path}')

    return fig


def downsample_data(gaze_x, gaze_y, method='random', target_n=500, seed=42):
    """
    对数据进行下采样

    Args:
        gaze_x, gaze_y: 原始数据
        method: 下采样方法
            - 'random': 随机采样
            - 'uniform': 均匀间隔采样
        target_n: 目标点数
        seed: 随机种子

    Returns:
        下采样后的数据
    """
    n_points = len(gaze_x)

    if n_points <= target_n:
        print(f'  数据点数({n_points})已小于目标数({target_n})，不进行下采样')
        return gaze_x, gaze_y

    if method == 'random':
        np.random.seed(seed)
        indices = np.random.choice(n_points, target_n, replace=False)
    elif method == 'uniform':
        step = n_points // target_n
        indices = np.arange(0, n_points, step)[:target_n]
    else:
        raise ValueError(f'未知的下采样方法: {method}')

    return gaze_x[indices], gaze_y[indices]


def main():
    print('=' * 80)
    print('增强版3D眼动数据可视化')
    print('=' * 80)
    print()

    # 数据路径
    data_path = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'all' / 'all_trials_model_predictions_0111.csv'

    # 加载数据
    print(f'加载数据: {data_path}')
    gaze_x, gaze_y = load_gaze_data(data_path)
    print(f'原始数据点数量: {len(gaze_x)}')

    # 下采样
    print()
    print('进行随机下采样...')
    gaze_x, gaze_y = downsample_data(gaze_x, gaze_y, method='random', target_n=300)
    print(f'下采样后数据点数量: {len(gaze_x)}')

    # 分析眼动模式
    print()
    print('眼动模式分析:')
    stats = analyze_gaze_patterns(gaze_x, gaze_y)
    print(f'  平均眼动速度: {stats["mean_velocity"]:.2f} px/frame')
    print(f'  最大眼动速度: {stats["max_velocity"]:.2f} px/frame')
    print(f'  速度标准差: {stats["std_velocity"]:.2f} px/frame')
    print(f'  注视中心: ({stats["center_x"]:.1f}, {stats["center_y"]:.1f}) px')
    print(f'  水平散布: {stats["spread_x"]:.1f} px')
    print(f'  垂直散布: {stats["spread_y"]:.1f} px')

    # 生成不同模式的可视化
    output_dir = Path(__file__).resolve().parent
    modes = ['sequence', 'velocity', 'position', 'quadrant']

    print()
    print('生成可视化...')

    for mode in modes:
        print(f'  创建 {mode} 模式可视化...')
        output_path = output_dir / f'eyeball_gaze_{mode}.png'
        fig = create_enhanced_visualization(
            gaze_x, gaze_y,
            output_path=output_path,
            color_mode=mode,
            point_size=12,
            alpha=0.7
        )
        plt.close(fig)  # 关闭图形以节省内存

    print()
    print('=' * 80)
    print('完成！')
    print('=' * 80)
    print(f'输出目录: {output_dir}')
    print()
    print('生成的文件:')
    for mode in modes:
        print(f'  - eyeball_gaze_{mode}.png')


if __name__ == '__main__':
    main()
