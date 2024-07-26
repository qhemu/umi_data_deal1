import sys
import os
import csv
import json
import pathlib
import pickle
import warnings
import multiprocessing
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import cv2
import av
import click
from rich import print
from rich.progress import Progress, BarColumn, TimeElapsedColumn, SpinnerColumn, TextColumn
import h5py

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform,
    draw_predefined_mask,
    inpaint_tag
)

warnings.filterwarnings("ignore", category=UserWarning, message="deprecated pixel format used, make sure you did set range correctly")

@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='640,640', help='Output resolution')
@click.option('-of', '--out_fov', type=float, default=None, help='Output field of view')
@click.option('-cl', '--compression_level', type=int, default=99, help='Compression level for output data')
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help='Disable mirror observation by masking them out')
@click.option('-ms', '--mirror_swap', is_flag=True, default=False, help='Enable mirror swapping')
@click.option('-n', '--num_workers', type=int, default=None, help='Number of worker threads')

def main(input, output, out_res, out_fov, compression_level,
         no_mirror, mirror_swap, num_workers):
    print("[green]Initializing bot and settings...[/green]")
    bot = 'vx300s'

    initialize_constants_and_functions()

    out_res = tuple(int(x) for x in out_res.split(','))
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    print("[green]Setting up fisheye converter if needed...[/green]")
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(input[0])).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )

    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = []
    num = 0
    observations = defaultdict(dict)
    actions = defaultdict(dict)

    print("[green]Processing each input path...[/green]")
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()

        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"[yellow]Skipping {ipath.name}: no dataset_plan.pkl[/yellow]")
            continue

        plan = pickle.load(plan_path.open('rb'))
        videos_dict = defaultdict(list)

        for plan_episode in plan:
            grippers = plan_episode['grippers']
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)

            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)

            episode_data = {}
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
                qpos, act, nor_gripper_widths = motion_convert(bot, eef_pose.astype(np.float32), gripper['gripper_width'])
                episode_data['qpos'] = qpos
                episode_data['qvel'] = demo_end_pose
                episode_data['actions'] = act
                episode_data['gripper_width'] = nor_gripper_widths

                observations[num]['qpos'] = episode_data['qpos']
                observations[num]['qvel'] = episode_data['qvel']
                actions[num]['action'] = episode_data['actions']
            num += 1

            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()

                video_start, video_end = camera['video_start_end']
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

    print("[green]Getting image size...[/green]")
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
        print(f"[blue]Initial image height and width: {ih}, {iw}[/blue]")

    print("[green]Processing videos and generating replay buffer...[/green]")
    total_frames = sum(in_stream.frames for _, tasks in vid_args)
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeElapsedColumn(),
        TextColumn("{task.fields[info]}"),
        transient=True
    ) as progress:
        task = progress.add_task("[green]Processing videos...", total=total_frames, info="Starting")
        for i, (mp4_path, tasks) in enumerate(vid_args):
            
            print(f'[green]mp4_path: {mp4_path}[\green]')
            if "calibration" in mp4_path or "mapping" in mp4_path or "demo_C3461324973256_2024.06.21_19.23.28.722317" in mp4_path:
                print(f"[yellow]Skipping {mp4_path}: folder contains 'calibration' or 'mapping'[/yellow]")
                continue
            
            image_dict, camera_idx = video_to_zarr2(fisheye_converter, ih, iw, out_res, observations, no_mirror, mp4_path, tasks, progress, task, i + 1, len(vid_args))
            observations[i]['images'] = image_dict

            # Read qpos data from CSV file in the same directory as the videos
            csv_file_path = os.path.join(os.path.dirname(mp4_path), 'camera_trajectory_aloha_v3.csv')
            qpos_data = []
            qpos_success_data = []
            try:
                with open(csv_file_path, mode='r') as file:
                    csv_reader = csv.DictReader(file)
                    for row in csv_reader:
                        joint_angles = eval(row['joint_angles'])  # Convert string to list
                        qpos = [0] * 7 + joint_angles + [0] # Set first 7 values to 0 and append joint angles
                        qpos_data.append(qpos)
                        qpos_success_data.append(row['success'])
            except FileNotFoundError:
                print(f"[red]CSV file '{csv_file_path}' not found. Skipping qpos replacement.[/red]")
                return

            # Replace qpos in observations if the lengths match
            if len(qpos_data) >= len(observations[i]['images']['cam_right_wrist']):
                length = len(observations[i]['images']['cam_right_wrist']) - len(qpos_data)
                qpos_data = qpos_data[:length]
                qpos_success_data = qpos_success_data[:length]

            observations[i]['qpos'][-7:-2] = qpos_data[-7:-2]
            # notice the observations[i]['qpos'][13] is already used for gripper width, filled in motion_convert function, so only tak place of [7:13]
            # (7, 8, 9, 10, 11, 12) of (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13)
            observations[i]['qpos_success'] = qpos_success_data
            
            # Save and clear data after processing each video
            save_data(output, observations, actions, i)
            # Memory clean
            qpos_data.clear()
            qpos_success_data.clear()
            observations[i] = []

    print("[green]All videos processed and saved successfully![/green]")

def save_data(output, observations, actions, video_index):
    """
    Save observations and actions to HDF5 files.

    Args:
        output (str): Output directory path.
        observations (dict): Observations dictionary.
        actions (dict): Actions dictionary.
        video_index (int): Index of the current video being processed.

    Returns:
        None
    """
    directory = output.rstrip('/') + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"[green]Directory '{directory}' was created.[/green]")
    else:
        # print(f"[yellow]Directory '{directory}' already exists.[/yellow]")
        pass

    if 'qpos' not in observations[video_index]:
        print(f"[red]3rd check, Skipping video {video_index} due to missing 'qpos'.[/red]")
        return

    dataset_path = directory + f'episode_{video_index}.hdf5'
    with h5py.File(dataset_path, 'w') as h5_file:
        h5_file.attrs['sim'] = True
        h5_file.attrs['compress'] = False

        obs_grp = h5_file.create_group('observations')
        obs_grp.create_dataset('qpos', data=observations[video_index].get('qpos', []))
        obs_grp.create_dataset('qpos_success', data=observations[video_index].get('qpos_success', []))
        obs_grp.create_dataset('qvel', data=observations[video_index].get('qvel', []))
        act_grp = h5_file.create_group('action')
        act_grp.create_dataset('action', data=actions[video_index].get('action', []))

        img_grp = obs_grp.create_group('images')
        for cam_name in observations[video_index].get('images', {}).keys():
            img_shape = None
            for cam_frame, img_data in observations[video_index]['images'][cam_name].items():
                if img_shape is None:
                    img_shape = img_data.shape
                    maxshape = (None,) + img_shape
                    img_dataset = img_grp.create_dataset(
                        cam_name, shape=(0,) + img_shape, maxshape=maxshape, chunks=True, dtype=img_data.dtype
                    )
                img_dataset.resize(img_dataset.shape[0] + 1, axis=0)
                img_dataset[-1] = img_data
                # print(f"[blue]Added frame {cam_frame} to {cam_name} dataset. New shape: {img_dataset.shape}[/blue]")

    print(f"[blue]Created episode_{video_index}.hdf5[/blue]")

def initialize_constants_and_functions():
    """
    Initializes constants and functions used for normalizing and un-normalizing gripper positions and joints,
    as well as other configurations for the robot's movements.

    This function sets various constants and defines lambda functions for the following purposes:
    - Normalizing and un-normalizing gripper positions.
    - Normalizing and un-normalizing gripper joints.
    - Converting between positions and joints for both master and puppet grippers.
    - Normalizing gripper velocities.

    Constants defined:
    - DT: Time step duration.
    - FPS: Frames per second.
    - JOINT_NAMES: List of joint names.
    - START_ARM_POSE: Initial arm pose configuration.
    - MASTER_GRIPPER_POSITION_OPEN: Open position for the master gripper.
    - MASTER_GRIPPER_POSITION_CLOSE: Close position for the master gripper.
    - PUPPET_GRIPPER_POSITION_OPEN: Open position for the puppet gripper.
    - PUPPET_GRIPPER_POSITION_CLOSE: Close position for the puppet gripper.
    - MASTER_GRIPPER_JOINT_OPEN: Open joint position for the master gripper.
    - MASTER_GRIPPER_JOINT_CLOSE: Close joint position for the master gripper.
    - PUPPET_GRIPPER_JOINT_OPEN: Open joint position for the puppet gripper.
    - PUPPET_GRIPPER_JOINT_CLOSE: Close joint position for the puppet gripper.

    Lambda functions defined:
    - MASTER_GRIPPER_POSITION_NORMALIZE_FN: Normalizes master gripper positions.
    - PUPPET_GRIPPER_POSITION_NORMALIZE_FN: Normalizes puppet gripper positions.
    - MASTER_GRIPPER_POSITION_UNNORMALIZE_FN: Un-normalizes master gripper positions.
    - PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN: Un-normalizes puppet gripper positions.
    - MASTER2PUPPET_POSITION_FN: Converts master gripper positions to puppet gripper positions.
    - MASTER_GRIPPER_JOINT_NORMALIZE_FN: Normalizes master gripper joints.
    - PUPPET_GRIPPER_JOINT_NORMALIZE_FN: Normalizes puppet gripper joints.
    - MASTER_GRIPPER_JOINT_UNNORMALIZE_FN: Un-normalizes master gripper joints.
    - PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN: Un-normalizes puppet gripper joints.
    - MASTER2PUPPET_JOINT_FN: Converts master gripper joints to puppet gripper joints.
    - MASTER_GRIPPER_VELOCITY_NORMALIZE_FN: Normalizes master gripper velocities.
    - PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN: Normalizes puppet gripper velocities.
    - MASTER_POS2JOINT: Converts master gripper positions to joint positions.
    - MASTER_JOINT2POS: Converts master gripper joint positions to gripper positions.
    - PUPPET_POS2JOINT: Converts puppet gripper positions to joint positions.
    - PUPPET_JOINT2POS: Converts puppet gripper joint positions to gripper positions.

    Returns:
        None
    """
    # global XML_DIR
    global DT, FPS, JOINT_NAMES, START_ARM_POSE, MASTER_GRIPPER_POSITION_OPEN, MASTER_GRIPPER_POSITION_CLOSE
    global PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSE, MASTER_GRIPPER_JOINT_OPEN, MASTER_GRIPPER_JOINT_CLOSE
    global PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE, MASTER_GRIPPER_POSITION_NORMALIZE_FN
    global PUPPET_GRIPPER_POSITION_NORMALIZE_FN, MASTER_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
    global MASTER2PUPPET_POSITION_FN, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_NORMALIZE_FN
    global MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, MASTER2PUPPET_JOINT_FN
    global MASTER_GRIPPER_VELOCITY_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN, MASTER_POS2JOINT, MASTER_JOINT2POS
    global PUPPET_POS2JOINT, PUPPET_JOINT2POS
    
    # Define constants for the main function
    DT = 0.02  # Time step duration
    FPS = 50  # Frames per second
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]  # Names of the joints
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]  # Initial arm pose

    # Directory for XML assets (commented out in this example)
    # XML_DIR = '/home/xiuxiu/interbotix_ws/src/universal_manipulation_interface/diffusion_policy/dataset/assets/'

    # Gripper position and joint limits for master and puppet configurations
    MASTER_GRIPPER_POSITION_OPEN = 0.02417
    MASTER_GRIPPER_POSITION_CLOSE = 0.01244
    PUPPET_GRIPPER_POSITION_OPEN = 0.05800
    PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
    MASTER_GRIPPER_JOINT_OPEN = -0.493
    MASTER_GRIPPER_JOINT_CLOSE = -1.65
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

    # Functions for normalizing and un-normalizing gripper positions
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

    # Functions for normalizing and un-normalizing gripper joints
    MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

    # Functions for normalizing gripper velocities
    MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

    # Functions for converting between positions and joints for master and puppet grippers
    MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
    PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

def plot_gripper_width(filename, gripper_widths):
    """
    Plot and save the gripper width over time.

    Args:
        filename (str): Filename to save the plot.
        gripper_widths (np.ndarray): Array of gripper widths.

    Returns:
        None
    """
    mean_width = np.mean(gripper_widths)
    std_deviation = np.std(gripper_widths)
    min_width = np.min(gripper_widths)
    max_width = np.max(gripper_widths)

    print(f"[blue]Mean width: {mean_width:.4f}[/blue]")
    print(f"[blue]Standard Deviation: {std_deviation:.4f}[/blue]")
    print(f"[blue]Min width: {min_width:.4f}[/blue]")
    print(f"[blue]Max width: {max_width:.4f}[/blue]")

    plt.figure(figsize=(10, 5))
    plt.plot(gripper_widths, label='Gripper Width')
    plt.title('Gripper Width Over Time')
    plt.xlabel('Measurement Index')
    plt.ylabel('Width (meters)')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_gripper_width2(filename, widths):
    """
    Plot and save the width distribution with dynamic thresholds.

    Args:
        filename (str): Filename to save the plot.
        widths (np.ndarray): Array of gripper widths.

    Returns:
        None
    """
    tight_state_threshold = np.percentile(widths, 10)
    partial_open_threshold = np.percentile(widths, 50)
    open_state_threshold = np.percentile(widths, 90)

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
    plt.savefig(filename)
    plt.close()

def motion_convert(bot, qpos, gripper_widths):
    """
    Convert motion data from poses and gripper widths to normalized values.

    Args:
        bot (str): The robot identifier.
        qpos (np.ndarray): Array of poses.
        gripper_widths (np.ndarray): Array of gripper widths.

    Returns:
        tuple: Combined result, actions, and normalized gripper widths.
    """
    nor_gripper_widths = process_data(np.array(gripper_widths))

    qpos = MASTER_POS2JOINT(qpos)
    result = np.column_stack((qpos, nor_gripper_widths))

    new_array = np.zeros_like(result)
    combined_result = np.concatenate((new_array, result), axis=1)
    actions = MASTER_GRIPPER_JOINT_NORMALIZE_FN(combined_result)
    return combined_result * 0.1, actions * 0.1, nor_gripper_widths

def video_to_zarr2(fisheye_converter, ih, iw, out_res, observations, no_mirror, mp4_path, tasks, progress, task_id, current_video, total_videos):
    """
    Convert video frames to Zarr format and save as a video file.

    Args:
        fisheye_converter (FisheyeRectConverter): Fisheye converter object.
        ih (int): Input height.
        iw (int): Input width.
        out_res (tuple): Output resolution.
        observations (dict): Observations dictionary.
        no_mirror (bool): Disable mirror observation by masking them out.
        mp4_path (str): Path to the input video file.
        tasks (list): List of tasks for processing the video frames.
        progress (Progress): Rich Progress object for updating progress bar.
        task_id (int): Task ID for the Progress bar.
        current_video (int): Current video number being processed.
        total_videos (int): Total number of videos to be processed.

    Returns:
        tuple: Image dictionary and camera index.
    """
    pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
    tag_detection_results = pickle.load(open(pkl_path, 'rb'))
    resize_tf = get_image_transform(
        in_res=(iw, ih),
        out_res=out_res
    )
    tasks = sorted(tasks, key=lambda x: x['frame_start'])
    camera_idx = tasks[0]['camera_idx'] if tasks else None

    camera_name = 'cam_right_wrist' if camera_idx == 0 else 'cam_left_wrist'
    curr_task_idx = 0

    image_dict = defaultdict(dict)
    with av.open(mp4_path) as container:
        in_stream = container.streams.video[0]
        in_stream.thread_count = 1
        total_frames = in_stream.frames

        for frame_idx, frame in enumerate(container.decode(in_stream)):
            if curr_task_idx >= len(tasks):
                break

            if frame_idx < tasks[curr_task_idx]['frame_start']:
                continue
            elif frame_idx < tasks[curr_task_idx]['frame_end']:
                if frame_idx == tasks[curr_task_idx]['frame_start']:
                    buffer_idx = tasks[curr_task_idx]['buffer_start']

                img = frame.to_ndarray(format='rgb24')
                this_det = tag_detection_results[frame_idx]
                all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                for corners in all_corners:
                    img = inpaint_tag(img, corners)

                img = draw_predefined_mask(img, color=(0, 0, 0),
                                           mirror=no_mirror, gripper=True, finger=False)
                if fisheye_converter is None:
                    img = resize_tf(img)
                else:
                    img = fisheye_converter.forward(img)

                image_dict[camera_name][buffer_idx] = img
                buffer_idx += 1

                # Update the progress bar with additional information
                progress.update(task_id, advance=1, info=f"Video {current_video}/{total_videos}, frame {frame_idx + 1}/{total_frames}")

                if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                    curr_task_idx += 1
            else:
                assert False

    return image_dict, camera_idx

def gripper_convert(gripper_widths):
    """
    Convert gripper widths to normalized values.

    Args:
        gripper_widths (np.ndarray): Array of gripper widths.

    Returns:
        np.ndarray: Normalized gripper widths.
    """
    width_changes = np.diff(gripper_widths) / gripper_widths[:-1]
    normalized_widths = np.clip(width_changes * 1000, -0.01, 1)
    return normalized_widths

def gripper_convert2(gripper_widths):
    """
    Convert gripper widths to binary states (0 or 1).

    Args:
        gripper_widths (np.ndarray): Array of gripper widths.

    Returns:
        np.ndarray: Binary states of gripper widths.
    """
    width_changes = np.diff(gripper_widths)
    binary_states = (width_changes > 0).astype(int)
    return binary_states

def apply_hysteresis(widths, pickup_threshold=0.085, place_threshold=0.075):
    """
    Apply hysteresis to the gripper widths to determine states.

    Args:
        widths (np.ndarray): Array of gripper widths.
        pickup_threshold (float): Threshold for pickup state.
        place_threshold (float): Threshold for place state.

    Returns:
        list: List of states based on the widths.
    """
    state = 'waiting'
    states = []
    for w in widths:
        if state == 'waiting' and w > pickup_threshold:
            state = 'picking'
        elif state == 'picking' and w < place_threshold:
            state = 'placing'
        elif state == 'placing' and w > pickup_threshold:
            state = 'picking'
        states.append(state)
    return states

def smooth_and_threshold(widths, pickup_threshold=0.085, place_threshold=0.075):
    """
    Smooth and threshold the gripper widths to determine states.

    Args:
        widths (np.ndarray): Array of gripper widths.
        pickup_threshold (float): Threshold for pickup state.
        place_threshold (float): Threshold for place state.

    Returns:
        list: List of states based on the widths.
    """
    state = 0
    states = []
    for w in widths:
        if state == 0 and w > pickup_threshold:
            state = 1
        elif state == 1 and w < place_threshold:
            state = 0
        states.append(state)
    return states

def delayed_state_change(widths, pickup_threshold=0.085, place_threshold=0.075):
    """
    Apply delayed state change based on the gripper widths.

    Args:
        widths (np.ndarray): Array of gripper widths.
        pickup_threshold (float): Threshold for pickup state.
        place_threshold (float): Threshold for place state.

    Returns:
        list: List of states based on the widths.
    """
    state = 0
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
            last_open = False
        states.append(state)
    return states

def load_h5file(output, observations, actions):
    """
    Save observations and actions to HDF5 files.

    Args:
        output (str): Output directory path.
        observations (dict): Observations dictionary.
        actions (dict): Actions dictionary.

    Returns:
        None
    """
    directory = output + output
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[green]Directory '{directory}' was created.[/green]")
    else:
        print(f"[yellow]Directory '{directory}' already exists.[/yellow]")

    for key, value in observations.items():
        dataset_path = directory + f'episode_{key}.hdf5'
        with h5py.File(dataset_path, 'w') as root:
            root.attrs['sim'] = True
            root.attrs['compress'] = False
            obs_grp = root.create_group('observations')
            obs_grp.create_dataset('qpos', data=value['qpos'])
            obs_grp.create_dataset('qvel', data=value['qvel'])
            act_grp = root.create_group('action')
            act_grp.create_dataset('action', data=actions[key]['action'])

            img_grp = obs_grp.create_group('images')
            for cam_name in value['images'].keys():
                list_img = []
                for cam_frame, img_data in value['images'][cam_name].items():
                    cam_frame = str(cam_frame)
                    list_img.append(img_data)
                stacked_images = np.stack(list_img, axis=0)
                print(f"[blue]Stacked images shape for camera {cam_name}: {stacked_images.shape}[/blue]")
                img_grp.create_dataset(cam_name, data=stacked_images)
        print(f"[blue]Created episode_{key}.hdf5[/blue]")

def calculate_thresholds(data):
    """
    Calculate thresholds for gripper widths.

    Args:
        data (np.ndarray): Array of gripper widths.

    Returns:
        tuple: Tight state threshold, partial open threshold, open state threshold.
    """
    tight_state_threshold = np.percentile(data, 10)
    partial_open_threshold = np.percentile(data, 50)
    open_state_threshold = np.percentile(data, 90)
    return tight_state_threshold, partial_open_threshold, open_state_threshold

def map_width_to_state(width, tight_state_threshold, partial_open_threshold, open_state_threshold):
    """
    Map gripper width to a state value based on thresholds.

    Args:
        width (float): Gripper width.
        tight_state_threshold (float): Threshold for tight state.
        partial_open_threshold (float): Threshold for partial open state.
        open_state_threshold (float): Threshold for open state.

    Returns:
        float: State value.
    """
    if width <= tight_state_threshold:
        return 0.0
    elif width <= partial_open_threshold:
        return 0.5 * (width - tight_state_threshold) / (partial_open_threshold - tight_state_threshold)
    elif width <= open_state_threshold:
        return 0.5 + 0.5 * (width - partial_open_threshold) / (open_state_threshold - partial_open_threshold)
    else:
        return 1.0

def process_data(data):
    """
    Process gripper width data and map to state values.

    Args:
        data (np.ndarray): Array of gripper widths.

    Returns:
        list: List of state values.
    """
    tight_state_threshold, partial_open_threshold, open_state_threshold = calculate_thresholds(data)
    state_values = [map_width_to_state(width, tight_state_threshold, partial_open_threshold, open_state_threshold) for width in data]
    return state_values

if __name__ == "__main__":
    main()
