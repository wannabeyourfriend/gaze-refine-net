"""
将2D屏幕上的gaze point数据逆映射到3D眼球体表面并可视化

眼球建模：
- 直径24mm的球体
- 眼球到平面距离为70cm
- 显示器尺度: x,y ∈ (0,2000) * (0,1000)像素
- 对应27寸显示器
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 参数设置
EYEBALL_RADIUS_MM = 12.0  # 眼球半径12mm（直径24mm)
DISTANCE_TO_SCREEN_MM = 700.0  # 眼球到屏幕距离70cm
SCREEN_WIDTH_PX = 2000  # 屏幕宽度（像素）
SCREEN_HEIGHT_PX = 1000  # 屏幕高度（像素）

# 27寸显示器的物理尺寸（约597mm x 336mm）
SCREEN_WIDTH_MM = 597.0
SCREEN_HEIGHT_MM = 336.0

# 像素到毫米的转换比例
PX_TO_MM_X = SCREEN_WIDTH_MM / SCREEN_WIDTH_PX
PX_TO_MM_Y = SCREEN_HEIGHT_MM / SCREEN_HEIGHT_PX


def screen_to_eyeball_3d(screen_x_px, screen_y_px):
    """
    将2D屏幕坐标逆映射到3D眼球表面

    假设：
    - 眼球位于原点(0, 0, 0)
    - 屏幕在z = DISTANCE_TO_SCREEN_MM平面上
    - 屏幕中心对应(0, 0, DISTANCE_TO_SCREEN_MM)
    - 使用简单的透视投影模型

    Args:
        screen_x_px: 屏幕x坐标（像素）
        screen_y_px: 屏幕y坐标（像素）

    Returns:
        eyeball_x, eyeball_y, eyeball_z: 眼球表面的3D坐标（毫米）
    """
    # 转换为以屏幕中心为原点的坐标（毫米）
    screen_x_mm = (screen_x_px - SCREEN_WIDTH_PX / 2) * PX_TO_MM_X
    screen_y_mm = (screen_y_px - SCREEN_HEIGHT_PX / 2) * PX_TO_MM_Y

    # 计算从眼球原点到屏幕点的方向向量
    # 屏幕点坐标：(screen_x_mm, screen_y_mm, DISTANCE_TO_SCREEN_MM)
    direction_x = screen_x_mm
    direction_y = screen_y_mm
    direction_z = DISTANCE_TO_SCREEN_MM

    # 归一化方向向量
    direction_length = np.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
    unit_x = direction_x / direction_length
    unit_y = direction_y / direction_length
    unit_z = direction_z / direction_length

    # 眼球表面上的点（沿着方向向量延伸半径距离）
    eyeball_x = unit_x * EYEBALL_RADIUS_MM
    eyeball_y = unit_y * EYEBALL_RADIUS_MM
    eyeball_z = unit_z * EYEBALL_RADIUS_MM

    return eyeball_x, eyeball_y, eyeball_z


def load_gaze_data(csv_path):
    """加载眼动数据"""
    df = pd.read_csv(csv_path)

    # 提取original gaze points
    gaze_x = df['origin_gaze_x'].values
    gaze_y = df['origin_gaze_y'].values

    return gaze_x, gaze_y


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


def create_3d_visualization(gaze_x, gaze_y, output_path=None,
                           color_by='sequence', point_size=5,
                           alpha=0.8, figsize=(20, 6)):
    """
    创建3D眼动数据可视化

    Args:
        gaze_x: 屏幕x坐标数组
        gaze_y: 屏幕y坐标数组
        output_path: 输出图片路径（可选）
        color_by: 着色方式 ('sequence' 按序列, 'random' 随机)
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

    # 设置颜色
    n_points = len(gaze_x)
    if color_by == 'sequence':
        # 按序列着色（红色→蓝色）
        colors = plt.cm.jet(np.linspace(0, 1, n_points))
    elif color_by == 'random':
        colors = np.random.rand(n_points)
    else:
        colors = 'blue'

    # 创建图形
    fig = plt.figure(figsize=figsize)

    # 主3D视图
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # 绘制眼球球体
    draw_eyeball_sphere(ax1, alpha=0.15, color='lightblue')
    # 绘制gaze点
    if color_by == 'sequence':
        scatter1 = ax1.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=np.arange(n_points), s=point_size, alpha=alpha,
                              cmap='jet', edgecolors='none')
    else:
        scatter1 = ax1.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=colors, s=point_size, alpha=alpha,
                              edgecolors='none')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Gaze Label - 3D View')

    # 设置3D坐标轴范围
    limit = EYEBALL_RADIUS_MM * 1.2
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.set_zlim([0, limit])

    # 添加3D坐标系指示器
    ax1.quiver(0, 0, 0, limit/3, 0, 0, color='r', arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, limit/3, 0, color='g', arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, 0, limit/3, color='b', arrow_length_ratio=0.1)

    # 俯视图
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_eyeball_sphere(ax2, alpha=0.15, color='lightblue')
    if color_by == 'sequence':
        scatter2 = ax2.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=np.arange(n_points), s=point_size, alpha=alpha,
                              cmap='jet', edgecolors='none')
    else:
        scatter2 = ax2.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=colors, s=point_size, alpha=alpha,
                              edgecolors='none')
    ax2.view_init(elev=90, azim=0)  # 俯视视角
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('Top View')
    ax2.set_xlim([-limit, limit])
    ax2.set_ylim([-limit, limit])
    ax2.set_zlim([0, limit])

    # 侧视图
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_eyeball_sphere(ax3, alpha=0.15, color='lightblue')
    if color_by == 'sequence':
        scatter3 = ax3.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=np.arange(n_points), s=point_size, alpha=alpha,
                              cmap='jet', edgecolors='none')
    else:
        scatter3 = ax3.scatter(eyeball_x, eyeball_y, eyeball_z,
                              c=colors, s=point_size, alpha=alpha,
                              edgecolors='none')
    ax3.view_init(elev=0, azim=0)  # 侧视视角
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('Side View')
    ax3.set_xlim([-limit, limit])
    ax3.set_ylim([-limit, limit])
    ax3.set_zlim([0, limit])

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
    print('2D屏幕gaze点 → 3D眼球表面映射及可视化')
    print('=' * 80)
    print()

    # 数据路径
    repo = Path(__file__).resolve().parents[2]
    data_path = repo / 'data' / 'raw' / 'all' / 'all_trials_model_predictions_0111.csv'
    output_path = Path(__file__).resolve().parent / 'eyeball_gaze_visualization.png'

    # 加载数据
    print(f'加载数据: {data_path}')
    gaze_x, gaze_y = load_gaze_data(data_path)
    print(f'原始数据点数量: {len(gaze_x)}')

    # 下采样
    print()
    print('进行随机下采样...')
    gaze_x, gaze_y = downsample_data(gaze_x, gaze_y, method='random', target_n=800)
    print(f'下采样后数据点数量: {len(gaze_x)}')

    # 统计信息
    print()
    print('屏幕坐标统计:')
    print(f'  X: {gaze_x.min():.1f} ~ {gaze_x.max():.1f} px (mean: {gaze_x.mean():.1f})')
    print(f'  Y: {gaze_y.min():.1f} ~ {gaze_y.max():.1f} px (mean: {gaze_y.mean():.1f})')

    # 转换到3D眼球坐标
    print()
    print('转换到3D眼球坐标...')
    eyeball_coords = np.array([
        screen_to_eyeball_3d(x, y) for x, y in zip(gaze_x, gaze_y)
    ])

    print()
    print('3D眼球坐标统计 (毫米):')
    print(f'  X: {eyeball_coords[:, 0].min():.2f} ~ {eyeball_coords[:, 0].max():.2f} mm')
    print(f'  Y: {eyeball_coords[:, 1].min():.2f} ~ {eyeball_coords[:, 1].max():.2f} mm')
    print(f'  Z: {eyeball_coords[:, 2].min():.2f} ~ {eyeball_coords[:, 2].max():.2f} mm')

    # 验证所有点都在眼球表面
    distances = np.sqrt(np.sum(eyeball_coords**2, axis=1))
    print()
    print(f'验证 - 所有点到原点的距离:')
    print(f'  平均: {distances.mean():.2f} mm')
    print(f'  标准差: {distances.std():.4f} mm')
    print(f'  理论值: {EYEBALL_RADIUS_MM:.2f} mm (眼球半径)')

    # 创建可视化
    print()
    print('创建3D可视化...')
    fig = create_3d_visualization(
        gaze_x, gaze_y,
        output_path=output_path,
        color_by='sequence',
        point_size=15,
        alpha=0.7
    )

    print()
    print('=' * 80)
    print('完成！')
    print('=' * 80)
    print(f'输出文件: {output_path}')
    print()
    print('参数设置:')
    print(f'  眼球直径: {EYEBALL_RADIUS_MM * 2} mm')
    print(f'  眼球半径: {EYEBALL_RADIUS_MM} mm')
    print(f'  眼球到屏幕距离: {DISTANCE_TO_SCREEN_MM} mm ({DISTANCE_TO_SCREEN_MM/10:.1f} cm)')
    print(f'  屏幕尺寸: {SCREEN_WIDTH_PX} x {SCREEN_HEIGHT_PX} px')
    print(f'  屏幕物理尺寸: {SCREEN_WIDTH_MM} x {SCREEN_HEIGHT_MM} mm')


if __name__ == '__main__':
    main()
