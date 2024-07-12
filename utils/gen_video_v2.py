'''
Gneerate video for 3D animation of camera trajectory joint points by 3Blue1Brown style.
'''
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
from matplotlib import rcParams

# 设置3Blue1Brown风格
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['text.color'] = 'white'
rcParams['axes.labelcolor'] = 'white'
rcParams['xtick.color'] = 'white'
rcParams['ytick.color'] = 'white'
rcParams['axes.facecolor'] = '#2E2E2E'
rcParams['figure.facecolor'] = '#2E2E2E'
rcParams['axes.edgecolor'] = '#FFFFFF'
rcParams['grid.color'] = '#555555'
rcParams['lines.color'] = '#007ACC'

# 读取CSV文件
data = pd.read_csv('/home/haku/work/umi_data_deal1/data/demo_session/demo_C3461324973256_2024.06.21_19.26.02.375817/camera_trajectory_aloha_v3.csv')

# 初始化图形
fig = plt.figure(figsize=(10, 8), facecolor='#2E2E2E')
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴范围
ax.set_xlim([0.0, 0.3])  # 根据数据调整范围以放大显示
ax.set_ylim([0.0, 0.2])
ax.set_zlim([-0.08, 0])

# 设置坐标轴标签
ax.set_xlabel('X axis', fontsize=12, color='white', labelpad=20)
ax.set_ylabel('Y axis', fontsize=12, color='white', labelpad=20)
ax.set_zlabel('Z axis', fontsize=12, color='white', labelpad=20)

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=10, colors='white')

# 添加标题
ax.set_title('3D Point Animation', fontsize=16, color='white')

# 初始化点
points_true = ax.scatter([], [], [], color='blue', s=10, label='True')  # 初始化True点
points_false = ax.scatter([], [], [], color='red', s=10, label='False')  # 初始化False点

# 存储所有点的列表
x_data_true, y_data_true, z_data_true = [], [], []
x_data_false, y_data_false, z_data_false = [], [], []

# 添加显示 joint_angles 的文本框
joint_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='white')

def format_joint_angles(joint_angles):
    # 将 joint_angles 从字符串转换为列表，并格式化为4位小数
    joint_angles_list = eval(joint_angles)
    formatted_angles = [f'{angle:.4f}' for angle in joint_angles_list]
    return f"Joint Angles: [{' '.join(formatted_angles)}]"

def init():
    points_true._offsets3d = ([], [], [])
    points_false._offsets3d = ([], [], [])
    joint_text.set_text("")
    return points_true, points_false, joint_text

def update(frame):
    x = data.iloc[frame]['x']
    y = data.iloc[frame]['y']
    z = data.iloc[frame]['z']
    success = data.iloc[frame]['success']
    
    if success:
        x_data_true.append(z)
        y_data_true.append(y)
        z_data_true.append(x)
    else:
        x_data_false.append(z)
        y_data_false.append(y)
        z_data_false.append(x)
    
    points_true._offsets3d = (x_data_true, y_data_true, z_data_true)
    points_false._offsets3d = (x_data_false, y_data_false, z_data_false)
    
    joint_angles = data.iloc[frame]['joint_angles']
    joint_text.set_text(format_joint_angles(joint_angles))
    
    return points_true, points_false, joint_text

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

# 添加图例
ax.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', labelcolor='white', bbox_to_anchor=(0.85, 0.85))

# 添加网格线
ax.grid(True)

# 保存动画
ani.save('animation.mp4', writer='ffmpeg', fps=60, dpi=300)  # 增加dpi参数以提高分辨率

plt.show()
