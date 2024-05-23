# %%
import sys
import os
import matplotlib.pyplot as plt
# from common.normalize_util import (
#     array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
#     get_image_identity_normalizer, get_range_normalizer_from_stat)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from collections import defaultdict
# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from imagecodecs_numcodecs import register_codecs
register_codecs()
# from umi.common.pose_util import pose_to_mat, mat_to_pose10d
# from umi.umi_dataset_revise import forward_kinematics, inverse_kinematics,forward_kinematics
'''
数据集转成模型训练的格式
1. 数据存储在ReplayBuffer对象中
2 视频帧处理
2.1 转为RGB
2.2 标记检测
2.3 遮罩
2.4 鱼眼镜头
2.5 镜像处理
2.6 压缩图像
4.3.3 最终数据结构
'''


# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
# @click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-or', '--out_res', type=str, default='896,896')  # init ih, iw: 2028 2704
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level,
         no_mirror, mirror_swap, num_workers):


    out_res = tuple(int(x) for x in out_res.split(','))  # (224, 224)
    # 根据CPU核心数决定工作线程数，除非已指定
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        # 初始化相机参数转换器
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov  # 解析输出分辨率和视场角
        )


    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    num = 0
    observations = defaultdict(dict)
    actions = defaultdict(dict)
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        gripper_filename = ipath.joinpath('grippers')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue

        plan = pickle.load(plan_path.open('rb'))

        videos_dict = defaultdict(list)

        for plan_episode in plan:
            grippers = plan_episode['grippers']
            # check that all episodes have the same number of grippers
            if n_grippers is None:  # 1
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
            cameras = plan_episode['cameras']
            if n_cameras is None:  # 1
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[..., :3]
                eef_rot = eef_pose[..., 3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                episode_data['eef_pos'] = eef_pos.astype(np.float32)
                episode_data['eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data['gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data['demo_start_pose'] = demo_start_pose
                episode_data['demo_end_pose'] = demo_end_pose
                qpos, act, nor_gripper_widths = motion_convert(eef_pose.astype(np.float32), gripper['gripper_width'])
                episode_data['qpos'] = qpos
                episode_data['qvel'] = demo_end_pose
                episode_data['actions'] = act
                episode_data['gripper_width'] = nor_gripper_widths
                # 存储
                observations[num]['qpos'] = episode_data['qpos']
                observations[num]['qvel'] = episode_data['qvel']
                actions[num]['action'] = episode_data['actions']
            num = num + 1
            # aggregate video gen aguments
            n_frames = None
            # 0 {'video_path': 'demo_C3461324973256_2024.05.10_21.59.28.773017/raw_video.mp4', 'video_start_end': (2, 302)}
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()

                video_start, video_end = camera['video_start_end']  # 302  2
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)

                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
        print("0 ih, iw:", ih, iw)  # ih, iw: 2028 2704
    i = 0  # 循环不同的视频
    for mp4_path, tasks in vid_args:
        image_dict, camera_idx = video_to_zarr2(fisheye_converter, ih, iw, out_res, observations, no_mirror,  mp4_path, tasks)
        observations[i]['images'] = image_dict
        i = i+1
    load_h5file(observations, actions)
    print(f"{len(all_videos)} videos used in total!")


def plot_gripper_width(filename, gripper_widths):
    mean_width = np.mean(gripper_widths)
    std_deviation = np.std(gripper_widths)
    min_width = np.min(gripper_widths)
    max_width = np.max(gripper_widths)

    print(f"Mean width: {mean_width:.4f}")
    print(f"Standard Deviation: {std_deviation:.4f}")
    print(f"Min width: {min_width:.4f}")
    print(f"Max width: {max_width:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(gripper_widths, label='Gripper Width')
    plt.title('Gripper Width Over Time')
    plt.xlabel('Measurement Index')
    plt.ylabel('Width (meters)')
    plt.legend()
    plt.show()
    # plt.savefig(filename)
    # plt.close()  # 关闭图形，释放资源

def plot_gripper_width2(filename, widths):
    # 计算百分位数作为阈值
    tight_state_threshold = np.percentile(widths, 10)  # 取最小10%的值作为紧夹状态的上限
    partial_open_threshold = np.percentile(widths, 50)  # 取中位数作为部分开放状态的上限
    open_state_threshold = np.percentile(widths, 90)  # 取最大10%的值作为开放状态的下限
    tight_state = widths[widths <= tight_state_threshold]
    partial_open_state = widths[(widths > tight_state_threshold) & (widths < open_state_threshold)]
    open_state = widths[widths >= open_state_threshold]
    plt.figure(figsize=(10, 5))
    plt.hist(widths, bins=30, alpha=0.7, label='All Widths')
    plt.axvline(tight_state_threshold, color='r', linestyle='dashed', linewidth=1, label='Tight State Threshold')
    plt.axvline(partial_open_threshold, color='g', linestyle='dashed', linewidth=1, label='Partial Open Threshold')
    plt.axvline(open_state_threshold, color='b', linestyle='dashed', linewidth=1, label='Open State Threshold')
    plt.legend()
    plt.title('Width Distribution with Dynamic Thresholds')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.show()
    # plt.savefig(filename)
    # plt.close()  # 关闭图形，释放资源

def plot_gripper_width2(widths):
    # 计算百分位数作为阈值
    tight_state_threshold = np.percentile(widths, 10)  # 取最小10%的值作为紧夹状态的上限
    partial_open_threshold = np.percentile(widths, 50)  # 取中位数作为部分开放状态的上限
    open_state_threshold = np.percentile(widths, 90)  # 取最大10%的值作为开放状态的下限
    tight_state = widths[widths <= tight_state_threshold]
    partial_open_state = widths[(widths > tight_state_threshold) & (widths < open_state_threshold)]
    open_state = widths[widths >= open_state_threshold]
    plt.figure(figsize=(10, 5))
    # plt.plot(widths, label='Gripper Width')
    plt.hist(widths, bins=30, alpha=0.7, label='All Widths')
    plt.axvline(tight_state_threshold, color='r', linestyle='dashed', linewidth=1, label='Tight State Threshold')
    plt.axvline(partial_open_threshold, color='g', linestyle='dashed', linewidth=1, label='Partial Open Threshold')
    plt.axvline(open_state_threshold, color='b', linestyle='dashed', linewidth=1, label='Open State Threshold')
    plt.legend()
    plt.title('Width Distribution with Dynamic Thresholds')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.show()

def motion_convert(qposes, gripper_widths):
    nor_gripper_widths = process_data(np.array(gripper_widths))  # １开０关
    qpos = MASTER_POS2JOINT(qposes)  # 空间位置到关节位置的转换
    result = np.column_stack((qpos, nor_gripper_widths))
    new_array = np.zeros_like(result)
    combined_result = np.concatenate((new_array, result), axis=1)
    actions = MASTER_GRIPPER_JOINT_NORMALIZE_FN(combined_result)
    return combined_result*0.1, actions*0.1, nor_gripper_widths
def motion_convert(qposes, gripper_widths):
    # # initial_guess = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])--
    # initial_guess = [0, -0.96, 1.16, 0, -0.3, 0]
    # qpos = inverse_kinematics(T=pose_mat,initial_guess=initial_guess)
    # qpos = MASTER_POS2JOINT(eef_pose.astype(np.float32))
    # nor_gripper_widths = process_data(gripper_widths)  # １开０关
    nor_gripper_widths = process_data(np.array(gripper_widths))  # １开０关
    # unnor_gripper = MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(np.array(nor_gripper_widths))
    # 画出夹爪宽度变化轨迹
    # plot_gripper_width(filename,gripper_widths)
    # plot_gripper_width2(filename,gripper_widths)
    qpos = MASTER_POS2JOINT(qposes)
    result = np.column_stack((qpos, nor_gripper_widths))
    new_array = np.zeros_like(result)
    combined_result = np.concatenate((new_array, result), axis=1)
    # actions = np.diff(eef_pose, axis=0)  # 如果动作是定义为两个连续时间点间的位置变化，那么可以简单地通过计算相邻时间点间的差分来得到：
    # actions = MASTER_JOINT2POS(demo_end_pose)
    # actions = MASTER_GRIPPER_JOINT_NORMALIZE_FN(qpos)
    actions = MASTER_GRIPPER_JOINT_NORMALIZE_FN(combined_result)
    return combined_result*0.1, actions*0.1, nor_gripper_widths

def video_to_zarr2(fisheye_converter, ih, iw, out_res, observations, no_mirror, mp4_path, tasks):
    pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')  # '/home/xiuxiu/interbotix_ws/src/universal_manipulation_interface/doll_basket_session/demos/demo_C3461324973256_2024.05.10_21.59.28.773017/tag_detection.pkl'
    tag_detection_results = pickle.load(open(pkl_path, 'rb'))  #
    resize_tf = get_image_transform(
        in_res=(iw, ih),
        out_res=out_res
    )
    tasks = sorted(tasks, key=lambda x: x['frame_start'])
    camera_idx = None
    for task in tasks:
        if camera_idx is None:
            camera_idx = task['camera_idx']
        else:
            assert camera_idx == task['camera_idx']
    # name = f'camera{camera_idx}_rgb'
    if camera_idx == 0:
        camera_name = 'cam_right_wrist'
    curr_task_idx = 0

    # image_dict = {}  # 创建字典来存储每帧的图像
    image_dict = defaultdict(dict)
    with av.open(mp4_path) as container:
        in_stream = container.streams.video[0]
        # in_stream.thread_type = "AUTO"
        in_stream.thread_count = 1
        buffer_idx = 0
        for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
            if curr_task_idx >= len(tasks):
                # all tasks done
                break

            if frame_idx < tasks[curr_task_idx]['frame_start']:
                # current task not started
                continue
            elif frame_idx < tasks[curr_task_idx]['frame_end']:
                if frame_idx == tasks[curr_task_idx]['frame_start']:
                    buffer_idx = tasks[curr_task_idx]['buffer_start']

                # do current task
                img = frame.to_ndarray(format='rgb24')  # (2028,2704,3)
                # inpaint tags 修补标签
                this_det = tag_detection_results[frame_idx]
                all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                for corners in all_corners:
                    img = inpaint_tag(img, corners)

                # mask out gripper
                img = draw_predefined_mask(img, color=(0, 0, 0),
                                           mirror=no_mirror, gripper=True, finger=False)# (2028,2704,3)
                # resize
                if fisheye_converter is None:
                    img = resize_tf(img)
                else:
                    img = fisheye_converter.forward(img)

                # compress image
                # Convert RGB to BGR for OpenCV compatibility
                img = img[:, :, [2, 1, 0]]

                # Store the image in the dictionary
                image_dict[camera_name][buffer_idx] = img
                buffer_idx += 1

                if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                    # current task done, advance
                    curr_task_idx += 1
            else:
                assert False

    return image_dict, camera_idx

def gripper_convert(gripper_widths):
    # gripper_nor = MASTER_GRIPPER_POSITION_NORMALIZE_FN()
    # 计算最小值和最大值
    # min_val = np.min(gripper['gripper_width'])
    # max_val = np.max(gripper['gripper_width'])
    # 应用归一化公式
    # normalized_data = (gripper['gripper_width'] - min_val) / (max_val - min_val)
    # normalized_data = get_range_normalizer_from_stat(array_to_stats(gripper['gripper_width']), 1, 0)
    # max_width = np.max(gripper_widths)
    # 计算连续宽度的变化率
    width_changes = np.diff(gripper_widths) / gripper_widths[:-1]
    # # 放大变化率，例如乘以一个常数或使用非线性函数调整其范围
    normalized_widths = np.clip(width_changes * 1000, -0.01, 1)  # 举例放大并限制在-1到1之间
    # window_size = 2
    # smoothed_widths = np.convolve(gripper_widths, np.ones(window_size) / window_size, mode='valid')
    # 计算滑动窗口差分
    # smoothed_changes = np.diff(smoothed_widths)
    # 放大变化
    # normalized_widths = np.clip(smoothed_changes * 1000, -0.01, 1)

    # 归一化夹爪宽度到 0-1 范围, 转换为开和向量，其中 1 表示完全打开，0 表示完全关闭
    # normalized_widths = gripper_widths / max_width
    return normalized_widths

def gripper_convert2(gripper_widths):
    # 计算宽度的变化（差分）
    width_changes = np.diff(gripper_widths)
    # 生成0/1向量，正变化（增加）为1，负变化（减少）为0
    binary_states = (width_changes > 0).astype(int)
    return binary_states

def apply_hysteresis(widths, pickup_threshold=0.085, place_threshold=0.075):
    state = 'waiting'  # 初始状态为等待
    states = []
    for w in widths:
        if state == 'waiting' and w > pickup_threshold:
            state = 'picking'
        elif state == 'picking' and w < place_threshold:
            state = 'placing'
        elif state == 'placing' and w > pickup_threshold:
            state = 'picking'  # 可能再次进入拿起状态
        states.append(state)
    return states

def smooth_and_threshold(widths, pickup_threshold=0.085, place_threshold=0.075):
    state = 0  # 0 for closed, 1 for open
    states = []
    for w in widths:
        if state == 0 and w > pickup_threshold:
            state = 1
        elif state == 1 and w < place_threshold:
            state = 0
        states.append(state)
    return states


def delayed_state_change(widths, pickup_threshold=0.085, place_threshold=0.075):
    state = 0  # 0 for closed, 1 for open
    last_open = False
    states = []
    for w in widths:
        if state == 0 and w > pickup_threshold:
            state = 1
            last_open = True
        elif state == 1 and w < place_threshold:
            state = 0
        elif state == 0 and last_open and w > pickup_threshold:
            state = 1
            last_open = False  # Reset last open
        states.append(state)
    return states

def load_h5file(observations, actions):
    # for mp4_path, tasks in vid_args:
    import h5py
    for key, value in observations.items():  # 在单个观察中循环pos，这里默认一个机械臂带一个腕式视频
        dataset_path = f'debug_session/debug_session/episode_{key}.hdf5'
        with h5py.File(dataset_path, 'w') as root:
            # print(value)
            # 设置文件属性
            root.attrs['sim'] = True  # 或 False，根据您的需要
            root.attrs['compress'] = False  # 根据需要设置
            # 创建 observations 组
            obs_grp = root.create_group('observations')
            # 保存 qpos 和 qvel
            obs_grp.create_dataset('qpos', data=value['qpos'])
            obs_grp.create_dataset('qvel', data=value['qvel'])
            act_grp = root.create_group('action')
            act_grp.create_dataset('action', data=actions[key]['action'])

            # 创建 images 子组
            img_grp = obs_grp.create_group('images')
            # print(value['images'])
            for cam_name in value['images'].keys():
                list_img = []
                for cam_frame, img_data in value['images'][cam_name].items():
                    cam_frame = str(cam_frame)
                    # print(f"Camera name: {cam_name}, Type: {type(cam_name)}")
                    # print(img_data.shape)
                    list_img.append(img_data)
                    # 假设 img_data 是 numpy 数组; 如果不是，需要先转换成 numpy 数组
                # 堆叠图像数据
                stacked_images = np.stack(list_img, axis=0)  # 在新的轴0上堆叠
                print(stacked_images.shape)
                img_grp.create_dataset(cam_name, data=stacked_images)
            # 保存动作
        print(f'episode_{key}.hdf5')
    print("HDF5 文件已创建并写入数据。")



def calculate_thresholds(data):
    tight_state_threshold = np.percentile(data, 10)
    partial_open_threshold = np.percentile(data, 50)
    open_state_threshold = np.percentile(data, 90)
    return tight_state_threshold, partial_open_threshold, open_state_threshold


def map_width_to_state(width, tight_state_threshold, partial_open_threshold, open_state_threshold):
    if width <= tight_state_threshold:
        return 0.0
    elif width <= partial_open_threshold:
        return 0.5 * (width - tight_state_threshold) / (partial_open_threshold - tight_state_threshold)
    elif width <= open_state_threshold:
        return 0.5 + 0.5 * (width - partial_open_threshold) / (open_state_threshold - partial_open_threshold)
    else:
        return 1.0


def process_data(data):
    # 计算阈值
    tight_state_threshold, partial_open_threshold, open_state_threshold = calculate_thresholds(data)
    # 映射每个数据点到状态
    state_values = [map_width_to_state(width, tight_state_threshold, partial_open_threshold, open_state_threshold) for
                    width in data]

    return state_values


# def calculate_local_thresholds(data, window_size=30):
#     # 使用滑动窗口计算每个点的局部阈值
#     tight_state_thresholds = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
#     partial_open_thresholds = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
#     open_state_thresholds = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
#
#     return tight_state_thresholds, partial_open_thresholds, open_state_thresholds
#
#
# def map_width_to_state(width, tight_threshold, partial_threshold, open_threshold):
#     if width <= tight_threshold:
#         return 0.0
#     elif width <= partial_threshold:
#         return 0.5 * (width - tight_threshold) / (partial_threshold - tight_threshold)
#     elif width <= open_threshold:
#         return 0.5 + 0.5 * (width - partial_threshold) / (open_threshold - partial_threshold)
#     else:
#         return 1.0
#
#
# def process_data(data, window_size=30):
#     # 计算局部阈值
#     tight_thresholds, partial_thresholds, open_thresholds = calculate_local_thresholds(data, window_size)
#
#     # 映射每个数据点到状态
#     state_values = [
#         map_width_to_state(data[i + window_size // 2], tight_thresholds[i], partial_thresholds[i], open_thresholds[i])
#         for i in range(len(tight_thresholds))]
#
#     return state_values


if __name__ == "__main__":
    DT = 0.02
    FPS = 50
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    # Left finger position limits (qpos[7]), right_finger = -1 * left_finger
    MASTER_GRIPPER_POSITION_OPEN = 0.02417
    MASTER_GRIPPER_POSITION_CLOSE = 0.01244
    PUPPET_GRIPPER_POSITION_OPEN = 0.05800
    PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
    # Gripper joint limits (qpos[6])
    MASTER_GRIPPER_JOINT_OPEN = -0.493
    # MASTER_GRIPPER_JOINT_OPEN = 0.3083  # awe
    # MASTER_GRIPPER_JOINT_OPEN = 0.8  # act
    MASTER_GRIPPER_JOINT_CLOSE = -1.65
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    PUPPET_GRIPPER_JOINT_CLOSE = -0.6213
    ############################ Helper functions ############################
    MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
            MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
    MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
        MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

    MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

    MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

    MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
    PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))
    main()

'''
(umi)$ python scripts_slam_pipeline/07_generate_replay_buffer.py 
-o example_demo_session/dataset.zarr.zip example_demo_session
'''
