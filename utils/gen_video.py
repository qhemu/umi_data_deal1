'''
This file is used to generate a 3D animation video of the camera trajectory. Not including the code of joint array.
'''
import pandas as pdnot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation

# 读取CSV文件
data = pd.read_csv('/home/haku/work/umi_data_deal1/data/demo_session/demo_C3461324973256_2024.06.21_19.26.02.375817/camera_trajectory_aloha_v3.csv')

# 初始化图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴范围
ax.set_xlim([0.0, 0.3])  # 根据数据调整范围以放大显示
ax.set_ylim([0.0, 0.2])
ax.set_zlim([-0.08, 0])

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加标题
ax.set_title('3D Point Animation')

# 初始化点
points, = ax.plot([], [], [], 'bo', markersize=4)  # 设置点的样式和大小

# 存储所有点的列表
x_data, y_data, z_data = [], [], []

# 添加显示 joint_angles 的文本框
joint_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='red')

def init():
    points.set_data([], [])
    points.set_3d_properties([])
    joint_text.set_text("")
    return points, joint_text

def update(frame):
    x = data.iloc[frame]['x']
    y = data.iloc[frame]['y']
    z = data.iloc[frame]['z']
    x_data.append(z)
    y_data.append(y)
    z_data.append(x)
    points.set_data(x_data, y_data)
    points.set_3d_properties(z_data)
    joint_angles = data.iloc[frame]['joint_angles']
    joint_text.set_text(f"Joint Angles: {joint_angles}")
    return points, joint_text

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True)

# 添加网格线
ax.grid(True)

# 保存动画
ani.save('animation.mp4', writer='ffmpeg', fps=60)

plt.show()
