'''
This file is using calculus of differences ways to generate camera trajectory.
'''
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import modern_robotics as mr
from rich.progress import track
from rich.console import Console

console = Console()

# Functions for conversion and visualization
def quaternion_to_euler(q_x, q_y, q_z, q_w):
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def process_csv(input_csv, output_csv, arm_interface, use_differential_kinematics=False):
    df = pd.read_csv(input_csv)
    success_points = []
    failure_points = []

    # 获取第一行数据作为初始位置
    first_row = df.iloc[0]
    x, y, z = first_row['x'], first_row['y'], first_row['z']
    q_x, q_y, q_z, q_w = first_row['q_x'], first_row['q_y'], first_row['q_z'], first_row['q_w']
    roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)

    # 初始化机器人姿态
    initial_guess = np.array([x, y, z, roll, pitch, yaw], dtype=float)
    arm_interface.T_sb = poseToTransformationMatrix(initial_guess)

    # 打开输出文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件头
        writer.writerow(['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint_angles', 'success'])

        for index, row in track(df.iterrows(), total=df.shape[0], description="Processing CSV"):
            x, y, z = row['x'], row['y'], row['z']
            q_x, q_y, q_z, q_w = row['q_x'], row['q_y'], row['q_z'], row['q_w']
            roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)

            if use_differential_kinematics:
                target_position = [x, y, z]
                target_orientation = eulerAnglesToRotationMatrix([roll, pitch, yaw])
                joint_angles, success = arm_interface.move_to_target(target_position, target_orientation)
            else:
                success = arm_interface.set_ee_cartesian_trajectory(x, y, z, roll, pitch, yaw, initial_guess=initial_guess)
                joint_angles = arm_interface.joint_commands

            if success:
                success_points.append((x, y, z))
                initial_guess = joint_angles  # 更新初始猜测值为当前成功的关节角度
            else:
                failure_points.append((x, y, z))

            if index % 50 == 0:
                # console.print(f"Processed {index + 1}/{df.shape[0]} rows")
                pass

            # 写入每一行的结果
            writer.writerow([x, y, z, roll, pitch, yaw, joint_angles.tolist(), success])

    return success_points, failure_points

def visualize_points(success_points, failure_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if success_points:
        x_s, y_s, z_s = zip(*success_points)
        ax.scatter(x_s, y_s, z_s, c='blue', label='Success Points', alpha=0.6)

    if failure_points:
        x_f, y_f, z_f = zip(*failure_points)
        ax.scatter(x_f, y_f, z_f, c='red', label='Failure Points', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Robot interface and transformation utilities
class RobotDescription:
    def __init__(self):
        self.M = np.array([
            [1.0, 0.0, 0.0, 0.536494],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.42705],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.Slist = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]
        ]).T
        self.Guesses = [np.zeros(6), np.ones(6)]

robot_des = RobotDescription()
initial_guesses = robot_des.Guesses
joint_commands = [0, 0, 0, 0, 0, 0]

class InterbotixArmUXInterface:
    def __init__(self, robot_des, initial_guesses, joint_commands):
        self.robot_des = robot_des
        self.initial_guesses = initial_guesses
        self.joint_commands = joint_commands
        self.T_sb = np.identity(4)

    def command_positions(self, positions, vel, accel, mode):
        console.print(f"Commanding positions: {positions} with velocity: {vel} and acceleration: {accel} in mode: {mode}")

    def normalize_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def check_joint_limits(self, theta_list):
        normalized_thetas = [self.normalize_angle(theta) for theta in theta_list]
        console.print(f"Checking joint limits for: {normalized_thetas}")
        for idx, theta in enumerate(normalized_thetas):
            if theta < -np.pi or theta > np.pi:
                console.print(f"Joint {idx+1} out of limits: {theta}")
                return False
        return True

    def set_ee_pose_matrix(self, T_sd, custom_guess=None, execute=True, vel=1.0, accel=5.0, mode=0, initial_guess=None):
        if custom_guess is None:
            initial_guesses = self.initial_guesses
            initial_guesses[3] = self.joint_commands
        else:
            initial_guesses = [custom_guess]

        if initial_guess is not None:
            initial_guesses.insert(0, initial_guess)

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(self.robot_des.Slist, self.robot_des.M, T_sd, guess, 0.0001, 0.0001)
            if success:
                if self.check_joint_limits(theta_list):
                    if execute:
                        self.command_positions(theta_list, vel, accel, mode)
                    return theta_list, True
                else:
                    console.print(f"Joint limits check failed for theta_list: {theta_list}")
            else:
                console.print(f"IKinSpace failed for guess: {guess}")

        console.print("No valid pose could be found")
        return None, False

    def set_ee_cartesian_trajectory(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, moving_time=2.0, wp_period=0.02, initial_guess=None):
        rpy = rotationMatrixToEulerAngles(self.T_sb[:3, :3])
        T_sy = np.identity(4)
        T_sy[:2, :2] = yawToRotationMatrix(rpy[2])
        T_yb = np.dot(transInv(T_sy), self.T_sb)
        rpy = rotationMatrixToEulerAngles(T_yb[:3, :3])
        N = int(moving_time / wp_period)
        inc = 1.0 / float(N)
        joint_traj = []
        joint_positions = list(self.joint_commands)
        for i in range(N + 1):
            joint_traj.append(joint_positions)
            if i == N:
                break
            T_yb[:3, 3] += [inc * x, inc * y, inc * z]
            rpy[0] += inc * roll
            rpy[1] += inc * pitch
            rpy[2] += inc * yaw
            T_yb[:3, :3] = eulerAnglesToRotationMatrix(rpy)
            T_sd = np.dot(T_sy, T_yb)
            theta_list, success = self.set_ee_pose_matrix(T_sd, joint_positions, False)
            if success:
                joint_positions = theta_list
                self.T_sb = T_sd  # 更新姿态
            else:
                console.print(f"{i/float(N) * 100:.1f}% of trajectory successfully planned. Trajectory will not be executed.")
                break

        if success:
            mode = 1
            console.print("Executing trajectory in servo mode.")
            for cmd in joint_traj:
                console.print(f"Commanding trajectory position: {cmd}")
            self.T_sb = T_sd
            self.joint_commands = joint_positions

        return success

    def move_to_target(self, target_position, target_orientation, step_size=0.01, max_steps=1000):
        current_joint_angles = np.array(self.joint_commands, dtype=float)  # 确保为浮点数数组
        target_transform = mr.RpToTrans(target_orientation, target_position)
        
        for _ in range(max_steps):
            current_transform = mr.FKinSpace(self.robot_des.M, self.robot_des.Slist, current_joint_angles)
            transform_error = np.dot(mr.TransInv(current_transform), target_transform)
            error_twist = mr.se3ToVec(mr.MatrixLog6(transform_error))
            
            if np.linalg.norm(error_twist) < 1e-3:
                return current_joint_angles, True
            
            J = mr.JacobianSpace(self.robot_des.Slist, current_joint_angles)
            delta_theta = np.dot(np.linalg.pinv(J), error_twist) * step_size
            current_joint_angles += delta_theta
        
        return current_joint_angles, False

# Transformation matrix utilities
def transInv(T):
    R, p = T[:3, :3], T[:3, 3]
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def yawToRotationMatrix(yaw):
    R_z = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    return R_z

def poseToTransformationMatrix(pose):
    mat = np.identity(4)
    mat[:3, :3] = eulerAnglesToRotationMatrix(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat

def eulerAnglesToRotationMatrix(theta):
    return euler_matrix(theta[0], theta[1], theta[2], axes="sxyz")[:3, :3]

def rotationMatrixToEulerAngles(R):
    return list(euler_from_matrix(R, axes="sxyz"))

def euler_matrix(ai, aj, ak, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    if frame:
        M = np.dot(M, np.diag([1, 1, 1, -1]))
    return M

def euler_from_matrix(matrix, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

_EPS = np.finfo(float).eps * 4.0
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzx': (0, 0, 2, 0), 'sxzy': (0, 0, 0, 1),
    'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0), 'syzx': (1, 0, 2, 0), 'syxy': (1, 0, 0, 1),
    'szxy': (2, 0, 0, 0), 'szxz': (2, 0, 1, 0), 'szx': (2, 0, 2, 0), 'szx': (2, 0, 0, 1),
}
_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}

if __name__ == "__main__":
    # example
    use_differential_kinematics = True  # 通过这个标志位切换使用的方法

    robot_des = RobotDescription()
    initial_guesses = robot_des.Guesses
    joint_commands = [0, 0, 0, 0, 0, 0]  # 初始的关节角度位置
    arm_interface = InterbotixArmUXInterface(robot_des, initial_guesses, joint_commands)

    input_csv = '/home/haku/work/umi_data_deal1/data/demo_session/demo_C3461324973256_2024.06.21_19.26.02.375817/camera_trajectory.csv'
    output_csv = '/home/haku/work/umi_data_deal1/data/demo_session/demo_C3461324973256_2024.06.21_19.26.02.375817/camera_trajectory_aloha_v3.csv'
    success_points, failure_points = process_csv(input_csv, output_csv, arm_interface, use_differential_kinematics)

    # 可视化
    visualize_points(success_points, failure_points)

