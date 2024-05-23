import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from ikpy.chain import Chain
register_codecs()
import modern_robotics as mr
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
import hydra
from torch.utils.data import DataLoader


Slist = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            ]
        ).T
M = np.array(
            [
                [1.0, 0.0, 0.0, 0.536494],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.42705],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

def forward_kinematics(thetalist):
    """Compute the forward kinematics for the given joint angles."""
    return mr.FKinSpace(M, Slist, thetalist)

'''
initial_guess 是一个向量，包含了机器人各关节的初始角度值。这个初始猜测用于开始逆运动学的迭代过程。逆运动学可能有多解或无解，初始猜测的选择可能会影响算法找到的解或者算法的收敛速度。
'''
def inverse_kinematics( T, initial_guess, tolerance=1e-6, max_iterations=20):
    """Compute the inverse kinematics using the Newton-Raphson method."""
    qpos_list = []
    success_list = []

    # print("Type of T:", type(T))Shape of T: (300, 4, 4)
    # print("Shape of T:", T.shape if isinstance(T, np.ndarray) else "Not an ndarray") Type of T: <class 'numpy.ndarray'>
    # print("First element type:", type(T[0, 0, 0]))  # 检查数组内元素类型First element type: <class 'numpy.float32'>
    for pose_mat in T:
        qpos, success = mr.IKinSpace(M=M, Slist=Slist, T=pose_mat, thetalist0=initial_guess, eomg = 1.9, ev = 1.9 )
        # 更新初始猜测为当前计算的关节位置，这通常可以帮助提高逆运动学的计算效率和稳定性
        if success:
            qpos_list.append(qpos)
        else:
            print("Inverse kinematics did not converge")
    # qpos_list 现在包含了对应于每个位姿矩阵的关节位置
    thetalist = np.array(qpos_list)
    # print(thetalist.shape)
    # thetalist, success = mr.IKinSpace(Slist, M, T, initial_guess, tolerance, max_iterations)
    return  thetalist


if __name__ == "__main__":
    # constriants
    DT = 0.02
    FPS = 50
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    XML_DIR = '/home/xiuxiu/interbotix_ws/src/universal_manipulation_interface/diffusion_policy/dataset/assets/' # note: absolute path
    # Left finger position limits (qpos[7]), right_finger = -1 * left_finger
    MASTER_GRIPPER_POSITION_OPEN = 0.02417
    MASTER_GRIPPER_POSITION_CLOSE = 0.01244
    PUPPET_GRIPPER_POSITION_OPEN = 0.05800
    PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
    # Gripper joint limits (qpos[6])
    MASTER_GRIPPER_JOINT_OPEN = -0.8
    MASTER_GRIPPER_JOINT_CLOSE = -1.65
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

    ############################ Helper functions ############################
    MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
    MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

    MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

    MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

    MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
    PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

    import h5py
    import numpy as np

    # 定义文件路径
    file_path = 'example.hdf5'

    # 创建新的 HDF5 文件
    with h5py.File(file_path, 'w') as file:
        # 创建一个数据集
        dataset = file.create_dataset("dataset_name", (100,), dtype='i')

        # 写入数据到数据集
        dataset[:] = np.arange(100)

        # 创建一个组
        group = file.create_group("subgroup")

        # 在组内创建另一个数据集
        dataset2 = group.create_dataset("another_dataset", (50,), dtype='f')
        dataset2[:] = np.linspace(0, 1, 50)

        # 可以设置属性
        file.attrs['Author'] = 'Your Name'
        dataset2.attrs['Description'] = 'This is a dataset example.'

    print("HDF5 文件已创建并写入数据。")
        # robot.gripper.set_single_gripper_position(gripper_position[0], gripper="left")  # 假设只有一个夹爪