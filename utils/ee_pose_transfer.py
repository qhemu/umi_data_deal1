'''
This file is used to process the camera trajectory data and calculate the trajectory of the end effector of the robot arm.
By using the InterbotixArmUXInterface class and related functions.
'''
import csv
import math
import numpy as np
import modern_robotics as mr
from rich.progress import track
from rich.console import Console

console = Console()

# Define the InterbotixArmUXInterface class and related functions here
class InterbotixArmUXInterface:
    def __init__(self, robot_des, initial_guesses, joint_commands):
        self.robot_des = robot_des
        self.initial_guesses = initial_guesses
        self.joint_commands = joint_commands
        self.T_sb = np.identity(4)  # Initialize transformation matrix

    def command_positions(self, positions, vel, accel, mode):
        console.print(f"Commanding positions: {positions} with velocity: {vel} and acceleration: {accel} in mode: {mode}")

    def check_joint_limits(self, theta_list):
        console.print(f"Checking joint limits for: {theta_list}")
        # Example limit check
        for theta in theta_list:
            if theta < -np.pi or theta > np.pi:
                return False
        return True

    def set_ee_pose_matrix(self, T_sd, custom_guess=None, execute=True, vel=1.0, accel=5.0, mode=0):
        if custom_guess is None:
            initial_guesses = self.initial_guesses
            initial_guesses[3] = self.joint_commands
        else:
            initial_guesses = [custom_guess]

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(self.robot_des.Slist, self.robot_des.M, T_sd, guess, 0.0001, 0.0001)
            solution_found = True

            if success:
                solution_found = self.check_joint_limits(theta_list)
            else:
                solution_found = False

            if solution_found:
                if execute:
                    self.command_positions(theta_list, vel, accel, mode)
                return theta_list, True

        console.print("[red]No valid pose could be found[/red]")
        return theta_list, False

    def set_ee_pose_components(self, x=0, y=0, z=0, roll=math.pi, pitch=0, yaw=0, custom_guess=None, execute=True, vel=1.0, accel=5.0):
        T_sd = poseToTransformationMatrix([x, y, z, roll, pitch, yaw])
        return self.set_ee_pose_matrix(T_sd, custom_guess, execute, vel, accel, 0)

    def set_ee_cartesian_trajectory(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, moving_time=2.0, wp_period=0.02):
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
            else:
                console.print(f"[yellow]{i/float(N) * 100:.1f}% of trajectory successfully planned. Trajectory will not be executed.[/yellow]")
                break

        if success:
            mode = 1  # Assuming servo mode
            console.print("[green]Executing trajectory in servo mode.[/green]")
            for cmd in joint_traj:
                console.print(f"Commanding trajectory position: {cmd}")
            self.T_sb = T_sd
            self.joint_commands = joint_positions

        return success

# Transformation matrix utilities
def transInv(T):
    """Inverts a homogeneous transformation matrix."""
    R, p = T[:3, :3], T[:3, 3]
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def yawToRotationMatrix(yaw):
    """Calculates 2D Rotation Matrix given a desired yaw angle."""
    R_z = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    return R_z

def poseToTransformationMatrix(pose):
    """Transforms a Six Element Pose vector to a Transformation Matrix."""
    mat = np.identity(4)
    mat[:3, :3] = eulerAnglesToRotationMatrix(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat

def eulerAnglesToRotationMatrix(theta):
    """Calculates rotation matrix given euler angles in 'xyz' sequence."""
    return euler_matrix(theta[0], theta[1], theta[2], axes="sxyz")[:3, :3]

def rotationMatrixToEulerAngles(R):
    """Calculates euler angles given rotation matrix in 'xyz' sequence."""
    return list(euler_from_matrix(R, axes="sxyz"))

def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    if frame:
        M = np.dot(M, np.diag([1, 1, 1, -1]))
    return M

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
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
    'szxy': (2, 0, 0, 0), 'szxz': (2, 0, 1, 0), 'szx':  (2, 0, 2, 0), 'szx':  (2, 0, 0, 1),
}
_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}


def quaternion_to_euler(q_x, q_y, q_z, q_w):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    Roll is rotation around x in radians (counterclockwise)
    Pitch is rotation around y in radians (counterclockwise)
    Yaw is rotation around z in radians (counterclockwise)
    """
    # Calculate the Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Calculate the Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Calculate the Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def process_csv(input_csv, output_csv, arm_interface):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header + ['roll', 'pitch', 'yaw'])  # 添加新的标题行

        for row in track(reader, description="Processing CSV"):
            q_x, q_y, q_z, q_w = map(float, row[-4:])
            roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)
            
            # Extracting x, y, z from the row
            x, y, z = map(float, row[5:8])

            # Calculating the trajectory
            success = arm_interface.set_ee_cartesian_trajectory(x, y, z, roll, pitch, yaw)
            if success:
                console.print(f"Trajectory successfully calculated for frame {row[0]}")
            else:
                console.print(f"[red]Trajectory calculation failed for frame {row[0]}[/red]")

            new_row = row + [roll, pitch, yaw]
            writer.writerow(new_row)

class RobotDescription:
    def __init__(self):
        # 末端执行器在基坐标系中的初始位姿矩阵
        self.M = np.array([
            [1.0, 0.0, 0.0, 0.536494],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.42705],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # 空间螺旋轴矩阵，每列表示一个关节的螺旋轴向量
        self.Slist = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]
        ]).T

        # 逆运动学求解的初始猜测值列表
        self.Guesses = [np.zeros(6), np.ones(6)]

# example
robot_des = RobotDescription()
initial_guesses = robot_des.Guesses
joint_commands = [0, 0, 0, 0, 0, 0]  # 初始的关节角度位置
arm_interface = InterbotixArmUXInterface(robot_des, initial_guesses, joint_commands)

input_csv = 'data/debug_session/demos/demo_C3461324973256_2024.05.10_21.59.28.773017/camera_trajectory.csv'
output_csv = 'data/debug_session/demos/demo_C3461324973256_2024.05.10_21.59.28.773017/camera_trajectory_aloha.csv'
process_csv(input_csv, output_csv, arm_interface)
